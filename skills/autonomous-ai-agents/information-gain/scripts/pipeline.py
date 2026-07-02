#!/usr/bin/env python3
"""pipeline.py — the model-calling stages of the information-gain skill.

Given a PROMPT, produce the raw signals voi.py needs to rank the key questions whose
answers would most improve a RESPONSE to that prompt. Stages (each a role-specialized
Ollama model, mostly via direct /api/chat raw calls run in parallel):

    0. frame_and_plan   — restate goal/response-type/success + a baseline response
    1. generate_questions — find candidate questions that would change the response
    2. project_answers  — plausible answers + probabilities + derivability
    3. judge_plan_change — per-answer response-change and stakes vs the baseline response

Vocabulary note: the JSON keys `baseline_plan` and `delta_plan` are legacy names for the
baseline RESPONSE and the per-answer RESPONSE-change — kept stable to avoid churn; the
prompts and output speak in "response" terms.

Reuse: `build_prompt`, `resolve_alias`, `NON_ENGLISH_MODELS` come from the `ask`
skill's model_utils (resolved at runtime via HERMES_HOME / ASK_SCRIPTS_DIR). The
raw /api/chat call mirrors ask.py::dispatch_single_raw but is owned here so the
many small scoring calls parallelize without the agent-loop / reasoning-effort race.
"""

import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.request

# ── Resolve the ask skill's model_utils at runtime (soft dependency) ──────────
_ASK = os.environ.get("ASK_SCRIPTS_DIR") or os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "skills", "productivity", "ask", "scripts",
)
if _ASK not in sys.path:
    sys.path.insert(0, _ASK)
try:
    from model_utils import build_prompt, resolve_alias, NON_ENGLISH_MODELS  # noqa: E402
except ImportError as e:  # pragma: no cover - environment guard
    raise SystemExit(
        "information-gain requires the `ask` skill (model_utils.py). Looked in "
        f"{_ASK!r}. Install the ask skill or set ASK_SCRIPTS_DIR / HERMES_HOME."
    ) from e

# Sibling pure-math modules: voi (dedup/scoring), pairwise (comparative-elicitation aggregation).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voi  # noqa: E402
import pairwise  # noqa: E402

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
OLLAMA_TAGS_URL = OLLAMA_URL.replace("/api/chat", "/api/tags")
MAX_WORKERS = int(os.environ.get("INFOGAIN_MAX_WORKERS", "8"))

# ── token/time usage accounting (thread-safe; aggregated per run) ─────────────
_USAGE_LOCK = threading.Lock()
_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "model_seconds": 0.0}


def reset_usage():
    """Zero the per-run usage counters (call at the start of a run)."""
    with _USAGE_LOCK:
        _USAGE.update(calls=0, input_tokens=0, output_tokens=0, model_seconds=0.0)


def get_usage():
    """Snapshot of usage since the last reset: calls, input/output tokens (from
    Ollama's prompt_eval_count / eval_count), and summed model wall-seconds."""
    with _USAGE_LOCK:
        return dict(_USAGE)


# ── low-level: raw Ollama call + JSON extraction ─────────────────────────────


def ollama_reachable(timeout=5):
    """True if the Ollama daemon answers /api/tags (used for preflight / tests)."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=timeout):
            return True
    except Exception:
        return False


def raw_chat(model, user_content, timeout=120, temperature=0.0, num_predict=900):
    """Single direct /api/chat call. Returns {content, elapsed, error}.

    `build_prompt` handles the /no_think prefix (Qwen) and English directive
    (GLM and other NON_ENGLISH_MODELS) for us.
    """
    start = time.time()
    try:
        english_only = model in NON_ENGLISH_MODELS
        prompt = build_prompt(user_content, "", model, english_only=english_only)
        data = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = (result.get("message") or {}).get("content", "")
        itok, otok = int(result.get("prompt_eval_count") or 0), int(result.get("eval_count") or 0)
        elapsed = time.time() - start
        with _USAGE_LOCK:
            _USAGE["calls"] += 1
            _USAGE["input_tokens"] += itok
            _USAGE["output_tokens"] += otok
            _USAGE["model_seconds"] += elapsed
        return {"content": content.strip(), "elapsed": elapsed, "error": None,
                "input_tokens": itok, "output_tokens": otok}
    except Exception as e:
        return {"content": "", "elapsed": time.time() - start, "error": str(e),
                "input_tokens": 0, "output_tokens": 0}


def extract_json(text):
    """Best-effort parse of a JSON object/array from model output.

    Handles ```json fences and surrounding prose. Raises ValueError if nothing
    parses.
    """
    if not text:
        raise ValueError("empty model output")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    raise ValueError("no parseable JSON in model output")


def _call_json(model, prompt, timeout, num_predict, retries=1, sink=None, temperature=0.0):
    """raw_chat + extract_json with one retry that nudges toward strict JSON.

    `temperature` is forwarded to the model — keep it 0 for stable scoring stages,
    raise it for generation to sample the model's distribution. If `sink` is a list,
    append one trace dict (model / prompt / raw output / elapsed / attempts / error)
    for 'show your work' diagnostics.
    """
    last_err = None
    last_raw = ""
    last_elapsed = None
    for attempt in range(retries + 1):
        content = prompt if attempt == 0 else (
            prompt + "\n\nReturn ONLY valid JSON. No prose, no markdown fences."
        )
        r = raw_chat(model, content, timeout=timeout, num_predict=num_predict,
                     temperature=temperature)
        last_raw, last_elapsed = r["content"], r["elapsed"]
        if r["error"]:
            last_err = r["error"]
            continue
        try:
            parsed = extract_json(r["content"])
            if sink is not None:
                sink.append({"model": model, "prompt": prompt, "raw": r["content"],
                             "elapsed": r["elapsed"], "attempts": attempt + 1, "error": None})
            return parsed, None
        except ValueError as e:
            last_err = f"{e} (raw: {r['content'][:160]!r})"
    if sink is not None:
        sink.append({"model": model, "prompt": prompt, "raw": last_raw,
                     "elapsed": last_elapsed, "attempts": retries + 1, "error": last_err})
    return None, last_err


# ── prompt builders (separated so --dry-run can show them) ───────────────────


def _evidence_block(evidence, instruction):
    """Render a list of already-established facts (the 'evidence' loop) for a prompt."""
    if not evidence:
        return ""
    bullets = "\n".join(f"- {e}" for e in evidence)
    return f"\nALREADY ESTABLISHED ({instruction}):\n{bullets}\n"


def frame_prompt(problem, evidence=None):
    return (
        "You are preparing to RESPOND to a prompt. First scope it: what response does it call "
        "for, and what is the best you'd say right now?\n\n"
        f"PROMPT:\n{problem}\n"
        f"{_evidence_block(evidence, 'treat as known facts and fold into the baseline response')}"
        "\nReturn ONLY a JSON object:\n"
        '{"goal": str, "decision": str, "success_criteria": [str], "baseline_plan": str}\n'
        "- goal: the underlying objective of the prompt in one sentence.\n"
        "- decision: the kind of response/answer the prompt calls for.\n"
        "- success_criteria: 2-4 short bullet strings for a good response.\n"
        "- baseline_plan: the best response/answer you would give to the prompt RIGHT NOW, "
        "given it and any established facts above (assume the most likely interpretation of "
        "remaining ambiguity; 2-5 sentences). This baseline is what we measure value against.\n"
        "Respond ONLY with the JSON object."
    )


_LENS_DIRECTIVE = {
    "contrarian": ("These MUST CHALLENGE the baseline approach itself — question whether the framing, "
                   "tool, or strategy is even right (not just refine it)."),
    "vantage": ("These MUST be questions whose answer would DIFFER depending on which environment / "
                "server / identity / credential / token you investigate from — name the vantage axis "
                "in each (e.g. 'from prod vs staging', 'as which DB user')."),
    "premortem": ("These MUST assume the baseline plan shipped and FAILED in production. Hunt the "
                  "latent hazard whose answer, known now, would have prevented it — data loss / "
                  "corruption, security compromise, irreversible or destructive actions, silent wrong "
                  "output, runaway cost. Name the failure each question targets."),
}


def questions_prompt(problem, framing, n, avoid=None, evidence=None, family=None):
    avoid_block = ""
    if avoid:
        bullets = "\n".join(f"- {q}" for q in avoid)
        avoid_block = (
            "\nDo NOT repeat or paraphrase these already-considered questions:\n"
            f"{bullets}\n"
        )
    family_block = ""
    if family:
        directive = _LENS_DIRECTIVE.get(family.get("lens", "scoped"), "")
        family_block = (
            f"\nGenerate questions ONLY within this FAMILY of unknowns:\n"
            f"  FAMILY: {family.get('name', '')} — {family.get('scope', '')}\n"
            f"  {directive}\n"
        )
    return (
        "You are finding the key questions whose answers would most improve a RESPONSE to "
        "a prompt.\n\n"
        f"PROMPT:\n{problem}\n"
        f"{_evidence_block(evidence, 'resolved — do NOT ask about these again')}"
        f"\nGOAL: {framing.get('goal', '')}\n"
        f"RESPONSE TYPE: {framing.get('decision', '')}\n"
        f"{avoid_block}{family_block}\n"
        f"Propose {n} DISTINCT key questions whose answers are currently unknown and "
        "would change or improve the response to this prompt. Cover DIFFERENT hidden "
        "assumptions; avoid near-duplicates.\n\n"
        "Return ONLY a JSON object:\n"
        '{"questions": [{"question": str, "type": str, "why": str, "target": str}, ...]}\n'
        "- type: one of [scope, constraint, audience, data, integration, risk, "
        "success-metric, resource, assumption, other].\n"
        "- target: a SHORT label (2-5 words) naming the single hidden assumption / "
        "latent variable the question resolves. Two questions resolving the same "
        "latent MUST share the same target.\n"
        "Respond ONLY with the JSON object."
    )


def answers_prompt(problem, framing, question, m, evidence=None):
    return (
        "Project the plausible answers to a clarifying question about a prompt.\n\n"
        f"PROMPT:\n{problem}\n"
        f"{_evidence_block(evidence, 'known; if they answer the question, derivable_prob is high')}"
        f"\nGOAL: {framing.get('goal', '')}\n"
        f"QUESTION: {question}\n\n"
        f"Enumerate the {m} most plausible DISTINCT answers. For each, estimate a "
        "probability (0-1) that it is the true answer given the prompt. Also estimate "
        "derivable_prob — can you ALREADY infer the answer from the prompt + established "
        "facts (so asking adds nothing)?\n\n"
        "Return ONLY a JSON object:\n"
        '{"derivable_prob": float, "answers": [{"answer": str, "prob": float}, ...]}\n'
        "- derivable_prob: 0-1, probability the answer is ALREADY inferable from the prompt + "
        "established facts (high = you basically know it; asking buys little).\n"
        f"- Provide 2 to {m} answers; probabilities need not sum to exactly 1.\n"
        "Respond ONLY with the JSON object."
    )


def judge_prompt(problem, framing, baseline_plan, question, answers):
    enumerated = "\n".join(
        f"{i + 1}. {a.get('answer', '')}" for i, a in enumerate(answers)
    )
    return (
        "Estimate how much each possible answer would change your RESPONSE to the prompt, "
        "and the cost of answering wrong.\n\n"
        f"PROMPT:\n{problem}\n\n"
        f"GOAL: {framing.get('goal', '')}\n\n"
        "BASELINE RESPONSE (your best answer to the prompt right now):\n"
        f"{baseline_plan}\n\n"
        f"QUESTION: {question}\n\n"
        f"POSSIBLE ANSWERS:\n{enumerated}\n\n"
        "For EACH answer, in the SAME ORDER, judge two 0-1 scores:\n"
        "- delta_plan: how much your RESPONSE would CHANGE if this answer is true "
        "(0 = identical response, 1 = completely different response).\n"
        "- stakes: the cost/harm of having answered with the BASELINE response if this "
        "answer is actually true (0 = harmless, 1 = severely wrong or misleading).\n\n"
        "Return ONLY a JSON object:\n"
        '{"answers": [{"delta_plan": float, "stakes": float}, ...]}\n'
        "with exactly one entry per answer, in the given order.\n"
        "Respond ONLY with the JSON object."
    )


# ── stages ───────────────────────────────────────────────────────────────────


def frame_and_plan(problem, model, timeout=180, sink=None, evidence=None):
    """Stage 0. Returns (framing_dict, error). framing has goal/decision/
    success_criteria/baseline_plan (always a dict, even on partial failure)."""
    obj, err = _call_json(model, frame_prompt(problem, evidence), timeout, num_predict=700,
                          sink=sink)
    if not isinstance(obj, dict):
        return ({"goal": "", "decision": "", "success_criteria": [],
                 "baseline_plan": ""}, err or "framing returned non-object")
    obj.setdefault("goal", "")
    obj.setdefault("decision", "")
    obj.setdefault("success_criteria", [])
    obj.setdefault("baseline_plan", "")
    return obj, None


def _parse_question_items(obj):
    items = obj.get("questions") if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    out = []
    for q in (items or []):
        if not isinstance(q, dict):
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        out.append({"question": text,
                    "type": (q.get("type") or "other").strip(),
                    "why": (q.get("why") or "").strip(),
                    "target": (q.get("target") or "").strip(),
                    "family": (q.get("family") or "").strip(),   # families layer (else "")
                    "lens": (q.get("lens") or "").strip()})
    return out


def generate_questions(problem, framing, model, n, avoid=None, timeout=180, sink=None,
                       samples=1, temperature=0.0, evidence=None):
    """Stage 1. Draw `samples` independent generations at `temperature`, union + dedup.

    With samples>1 and temperature>0 this Monte-Carlo-samples the model's own
    distribution over "what matters" — breadth emerges from the model's uncertainty
    (the tail of the distribution), with NO human-seeded topic list. samples=1,
    temperature=0 is the deterministic (focus) path: a single greedy generation.
    Returns (deduped_records, error).
    """
    prompt = questions_prompt(problem, framing, n, avoid, evidence)
    samples = max(1, int(samples))

    def _one(_i):
        local = [] if sink is not None else None
        obj, err = _call_json(model, prompt, timeout, num_predict=900, sink=local,
                              temperature=temperature)
        return _parse_question_items(obj), (local[0] if local else None), err

    if samples == 1:
        runs = [_one(0)]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(samples, MAX_WORKERS)) as ex:
            runs = list(ex.map(_one, range(samples)))

    all_recs, errs = [], []
    for recs, cap, err in runs:
        all_recs.extend(recs)
        if sink is not None and cap:
            sink.append(cap)
        if err:
            errs.append(err)
    union = voi.dedupe(all_recs)
    return union, (None if union else (errs[0] if errs else "no questions generated"))


def consolidate_prompt(problem, candidates):
    listing = "\n".join(
        f"{i + 1}. {c.get('question', '')}  (target: {c.get('target', '')})"
        for i, c in enumerate(candidates))
    return (
        "You are de-duplicating clarifying questions for a prompt. Some of the "
        "questions below resolve the SAME underlying unknown, just worded differently "
        "(e.g. 'update latency' and 'data freshness' are the same unknown).\n\n"
        f"PROMPT:\n{problem}\n\n"
        f"CANDIDATE QUESTIONS:\n{listing}\n\n"
        "Group the questions that resolve the same underlying unknown, and return ONE "
        "canonical question per DISTINCT unknown (use the clearest phrasing). Keep "
        "genuinely distinct questions separate. Do NOT invent new questions and do NOT "
        "drop any distinct unknown.\n\n"
        "Return ONLY a JSON object:\n"
        '{"questions": [{"question": str, "type": str, "why": str, "target": str, '
        '"merged_count": int}, ...]}\n'
        "where merged_count is how many of the input questions this canonical one covers.\n"
        "Respond ONLY with the JSON object."
    )


def consolidate_questions(problem, candidates, model, timeout=150, sink=None):
    """Semantic dedup: cluster the sampled candidates by the underlying unknown and
    keep one canonical question per cluster. Topic-free — the grouping is driven by
    the questions themselves, not a seeded taxonomy. Never loses questions: on any
    failure (no JSON / empty result) it returns the input unchanged.
    """
    if len(candidates) <= 1:
        return candidates
    obj, err = _call_json(model, consolidate_prompt(problem, candidates), timeout,
                          num_predict=1500, sink=sink)
    out = _parse_question_items(obj)
    if not out:  # consolidation failed — never drop questions
        return candidates
    if isinstance(obj, dict):
        raw = obj.get("questions") or []
        for o, r in zip(out, raw):
            if isinstance(r, dict) and r.get("merged_count") is not None:
                o["merged_count"] = r.get("merged_count")
    return out


# ── families layer (a tier above questions) ──────────────────────────────────

_VANTAGE_HINT = ("system", "server", "api", "database", "db", "deploy", "auth", "token", "credential",
                 "environment", "config", "network", "integration", "service", "infrastructure",
                 "pipeline", "access", "permission", "endpoint", "cloud", "container", "repo",
                 "repository", "code")

# Hint matching is WORD-BOUNDARY-PREFIX ("deploy" → deploys/deployment, but "api" ⊄ "rapid",
# "code" ⊄ "encode"). Short hints that are prefixes of unrelated common words are matched as
# EXACT tokens instead ("prod" ⊄ "product", "drop" ⊄ "dropdown", "repo" ⊄ "report", "db" gets a
# real boundary — the old "db " substring missed "…the db" at end of text). Residual accepted:
# "auth" still prefix-matches "author" (the authentication/authorization coverage matters more).
_EXACT_HINTS = frozenset({"prod", "drop", "db", "repo"})


def _hint_pattern(hints):
    return re.compile("|".join(
        rf"\b{re.escape(h)}\b" if h in _EXACT_HINTS else rf"\b{re.escape(h)}" for h in hints))


_VANTAGE_RE = _hint_pattern(_VANTAGE_HINT)


def _vantage_relevant(framing):
    """Cheap gate: does the task involve systems/access/environments (so vantage matters)?"""
    blob = f"{framing.get('goal', '')} {framing.get('decision', '')}".lower()
    return bool(_VANTAGE_RE.search(blob))


# Failure-surface hints: the task can cause a costly/irreversible failure (so a pre-mortem matters).
# Deliberately CONSERVATIVE — the lens is auto-on, so read-only/summarize/research tasks must not trip
# it. Side-effecting verbs + high-stakes nouns only; no generic "code"/"config" (too broad), and no
# bare artifact nouns ("email"/"message"/"database") — those fire on RETRIEVAL tasks too (the #25
# tier-1 eval caught gmail-triage tripping on "email"); every genuine act-on-it task in the bank
# carries a verb hint (send/delete/write/migrate/drop).
_PREMORTEM_HINT = ("write", "send", "delete", "remove", "deploy", "release", "migrate", "drop",
                   "overwrite", "payment", "charge", "refund", "production", "prod", "credential",
                   "secret", "irreversible", "destructive", "publish", "broadcast", "mutate",
                   "transaction", "purchase", "commit", "merge")

_PREMORTEM_RE = _hint_pattern(_PREMORTEM_HINT)


def _premortem_relevant(framing):
    """Cheap gate: could acting on the baseline plan cause a costly/irreversible failure?"""
    blob = f"{framing.get('goal', '')} {framing.get('decision', '')}".lower()
    return bool(_PREMORTEM_RE.search(blob))


def families_prompt(problem, framing, n_scoped, contrarian, vantage, premortem=False):
    lenses = [f"- {n_scoped} SCOPED families: each a DISTINCT region/dimension of the unknowns "
              "(lens \"scoped\")."]
    if contrarian:
        lenses.append("- 1 CONTRARIAN family (lens \"contrarian\"): unknowns that challenge the baseline "
                      "approach itself — what if the framing/tool/strategy is wrong?")
    if vantage:
        lenses.append("- 1 VANTAGE family (lens \"vantage\"): unknowns whose answer would DIFFER by which "
                      "environment / server / identity / credential / token you investigate from.")
    if premortem:
        lenses.append("- 1 PRE-MORTEM family (lens \"premortem\"): unknowns that, if wrong, cause a "
                      "costly or irreversible FAILURE of the baseline plan in production.")
    allowed = '"scoped"|"contrarian"|"vantage"' + ('|"premortem"' if premortem else '')
    return (
        "You are organizing the key unknowns about a prompt into FAMILIES before drilling into "
        "individual questions.\n\n"
        f"PROMPT:\n{problem}\n\nGOAL: {framing.get('goal', '')}\n"
        f"RESPONSE TYPE: {framing.get('decision', '')}\n\n"
        "Propose these families (each a distinct grouping of related unknowns):\n"
        + "\n".join(lenses) + "\n\n"
        "Return ONLY a JSON object:\n"
        '{"families": [{"name": str, "scope": str, "lens": ' + allowed + "}, ...]}\n"
        "- name: 2-5 word family label. scope: one sentence on what unknowns it covers.\n"
        "Respond ONLY with the JSON object."
    )


def generate_families(problem, framing, model, n_scoped=3, contrarian=True, vantage="auto",
                      premortem="auto", timeout=180, sink=None):
    """Stage 1a. Returns (families, error). Each family: {name, scope, lens}. `vantage` and
    `premortem`: "on" | "off" | "auto" (include only when the task involves systems/access, resp.
    a failure surface)."""
    want_vantage = (vantage == "on") or (vantage == "auto" and _vantage_relevant(framing))
    want_premortem = (premortem == "on") or (premortem == "auto" and _premortem_relevant(framing))
    obj, err = _call_json(model, families_prompt(problem, framing, n_scoped, contrarian, want_vantage,
                                                 want_premortem),
                          timeout, num_predict=600, sink=sink)
    fams = []
    items = obj.get("families") if isinstance(obj, dict) else None
    for f in (items or []):
        if isinstance(f, dict) and (f.get("name") or "").strip():
            fams.append({"name": (f.get("name") or "").strip(),
                         "scope": (f.get("scope") or "").strip(),
                         "lens": (f.get("lens") or "scoped").strip()})
    return fams, (None if fams else (err or "no families generated"))


def generate_family_questions(problem, framing, families, model, n_per=3, timeout=180,
                              evidence=None, sink=None):
    """Stage 1b. For each family, generate n_per questions scoped to it (parallel), tagged with
    family + lens. Returns the families list, each carrying a 'questions' list of records."""
    if not families:
        return []

    def _one(fam):
        local = [] if sink is not None else None
        obj, err = _call_json(model, questions_prompt(problem, framing, n_per, evidence=evidence,
                                                      family=fam), timeout, num_predict=900, sink=local)
        recs = _parse_question_items(obj)
        for r in recs:
            r["family"] = fam["name"]
            r["lens"] = fam.get("lens", "scoped")
        out = dict(fam)
        out["questions"] = recs
        return out, (local[0] if local else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(families), MAX_WORKERS)) as ex:
        results = list(ex.map(_one, families))
    if sink is not None:  # collect per-family captures after the parallel map (avoid the append race)
        for _o, cap in results:
            if cap:
                sink.append(cap)
    return [out for out, _cap in results]

# NOTE: there is deliberately NO family-level "narrow/negate" stage. Families are domain EXPOSURE
# (coverage) only — every question is scored on its own merit by the per-question VOI pipeline, and
# selection (discard threshold + MMR with the family-diversity tier) does the filtering. A low-average
# family can still hold the single highest-value question (esp. contrarian/vantage), so we never drop a
# whole family; irrelevant families self-prune because their questions score low individually.


def project_answers(problem, framing, rec, model, m, timeout=120, capture=False, evidence=None):
    """Stage 2 (single question). Mutates rec with answers[] + derivable_prob."""
    sink = [] if capture else None
    obj, err = _call_json(model, answers_prompt(problem, framing, rec["question"], m, evidence),
                          timeout, num_predict=600, sink=sink)
    answers = []
    derivable = 0.0
    if isinstance(obj, dict):
        derivable = obj.get("derivable_prob", 0.0)
        for a in (obj.get("answers") or []):
            if isinstance(a, dict) and (a.get("answer") or "").strip():
                answers.append({"answer": a["answer"].strip(),
                                "prob": a.get("prob", 0.0)})
    rec["answers"] = answers
    rec["derivable_prob"] = derivable
    if err:
        rec["error"] = err
    if capture and sink:
        rec.setdefault("_trace", {})["project"] = sink[0]
    return rec


def judge_plan_change(problem, framing, baseline_plan, rec, model, timeout=150, capture=False):
    """Stage 3 (single question). Adds delta_plan + stakes to each answer in rec."""
    answers = rec.get("answers") or []
    if not answers:
        return rec
    sink = [] if capture else None
    obj, err = _call_json(
        model, judge_prompt(problem, framing, baseline_plan, rec["question"], answers),
        timeout, num_predict=500, sink=sink,
    )
    judged = obj.get("answers") if isinstance(obj, dict) else (
        obj if isinstance(obj, list) else [])
    judged = judged or []
    for i, a in enumerate(answers):
        j = judged[i] if i < len(judged) and isinstance(judged[i], dict) else {}
        a["delta_plan"] = j.get("delta_plan", 0.0)
        a["stakes"] = j.get("stakes", 0.0)
    if err:
        rec["error"] = err
    if capture and sink:
        rec.setdefault("_trace", {})["judge"] = sink[0]
    return rec


# ── stage 3, comparative variant (off by default; #24) ────────────────────────
# The absolute judge above scores each answer's delta_plan/stakes in isolation on a 0-1 scale —
# fragile WITHIN a task (the model can't reliably say 0.7 vs 0.5). The pairwise judge instead asks
# forced-choice comparisons ("which answer changes the response more?"), which models do far better,
# and aggregates them (Bradley-Terry, pairwise.py) into the SAME delta_plan/stakes [0,1] fields, so
# it's a drop-in for voi.evsi/score_record. Two virtual ANCHOR items (FLOOR = no change, CEILING =
# completely different) are present in every question's comparison set and map to 0/1, carrying an
# absolute-ish scale ACROSS questions so between-task ranking (the validated signal) is preserved.

_PAIRWISE_ANCHORS = {
    # (floor_label, ceiling_label) per dimension — the two fixed reference outcomes.
    "change": ("the response stays EXACTLY the baseline (no change at all)",
               "a COMPLETELY DIFFERENT response (opposite approach or conclusion)"),
    "stakes": ("it would NOT MATTER AT ALL which response the user got",
               "getting it wrong would be SEVERE — the response fails the user's real need"),
}
_PAIRWISE_QUESTION = {
    "change": "which one would make your RESPONSE to the prompt change MORE from the baseline?",
    "stakes": ("for which one would it MATTER MORE that you got the response right — i.e. higher "
               "cost if you assumed the baseline and that option were actually true?"),
}


def pairwise_judge_prompt(problem, framing, baseline_plan, question, items, dimension):
    """Prompt for ONE dimension's pairwise pass. `items` is the full comparison set INCLUDING the
    two anchors (index 0 = FLOOR, last = CEILING); the model judges every unordered pair."""
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
    pairs = pairwise.all_pairs(len(items))
    pair_lines = "\n".join(f"- pair [{i + 1}, {j + 1}]" for i, j in pairs)
    ask = _PAIRWISE_QUESTION.get(dimension, _PAIRWISE_QUESTION["change"])
    return (
        "Compare possible answers to a clarifying question PAIRWISE — judge each pair on its own, "
        "do NOT score them in isolation.\n\n"
        f"PROMPT:\n{problem}\n\n"
        f"GOAL: {framing.get('goal', '')}\n\n"
        "BASELINE RESPONSE (your best answer right now):\n"
        f"{baseline_plan}\n\n"
        f"QUESTION: {question}\n\n"
        f"OPTIONS (1 and {len(items)} are fixed reference points):\n{listing}\n\n"
        f"For EACH pair below, decide: {ask}\n{pair_lines}\n\n"
        "Return ONLY a JSON object:\n"
        '{"comparisons": [{"a": int, "b": int, "winner": int|"tie"}, ...]}\n'
        "- a, b: the two option NUMBERS from the pair.\n"
        "- winner: the option number that wins the comparison, or \"tie\" if roughly equal.\n"
        "Judge every listed pair. Respond ONLY with the JSON object."
    )


def _pairwise_scores(problem, framing, baseline_plan, question, answers, dimension, model,
                     timeout, sink=None):
    """Run one dimension's pairwise pass → per-real-answer score in [0,1] (anchored). Returns a list
    aligned with `answers`, or None if the comparison call yields nothing usable (caller defaults to
    0.0, mirroring the absolute judge's safe-zero)."""
    floor_lbl, ceil_lbl = _PAIRWISE_ANCHORS.get(dimension, _PAIRWISE_ANCHORS["change"])
    items = [floor_lbl] + [a.get("answer", "") for a in answers] + [ceil_lbl]
    floor_idx, ceil_idx = 0, len(items) - 1
    obj, err = _call_json(
        model, pairwise_judge_prompt(problem, framing, baseline_plan, question, items, dimension),
        timeout, num_predict=700, sink=sink)
    raw = obj.get("comparisons") if isinstance(obj, dict) else (obj if isinstance(obj, list) else None)
    comps = []
    for c in (raw or []):
        if not isinstance(c, dict):
            continue
        try:
            a, b = int(c["a"]) - 1, int(c["b"]) - 1  # 1-based in the prompt → 0-based here
        except (KeyError, TypeError, ValueError):
            continue
        w = c.get("winner")
        if isinstance(w, str) and w.strip().lower().startswith("tie"):
            outcome = 0.5
        else:
            try:
                outcome = 1.0 if int(w) - 1 == a else 0.0 if int(w) - 1 == b else 0.5
            except (TypeError, ValueError):
                outcome = 0.5
        comps.append((a, b, outcome))
    if not comps:
        return None, err
    strengths = pairwise.bradley_terry(len(items), comps)
    scores = pairwise.anchored_scores(strengths, floor_idx, ceil_idx)
    return scores[1:-1], err  # drop the two anchors → align with `answers`


def judge_plan_change_pairwise(problem, framing, baseline_plan, rec, model, timeout=150,
                               capture=False):
    """Stage 3 (comparative). Same contract as judge_plan_change: writes delta_plan + stakes onto
    each answer in rec — but elicited by pairwise comparison (BT-aggregated, anchored), not absolute
    scoring. Two model calls per question (change, stakes). Safe-zeroes on any parse failure."""
    answers = rec.get("answers") or []
    if not answers:
        return rec
    sink_c = [] if capture else None
    sink_s = [] if capture else None
    deltas, err_c = _pairwise_scores(problem, framing, baseline_plan, rec["question"], answers,
                                     "change", model, timeout, sink_c)
    stakes, err_s = _pairwise_scores(problem, framing, baseline_plan, rec["question"], answers,
                                     "stakes", model, timeout, sink_s)
    for i, a in enumerate(answers):
        a["delta_plan"] = deltas[i] if deltas and i < len(deltas) else 0.0
        a["stakes"] = stakes[i] if stakes and i < len(stakes) else 0.0
    err = err_c or err_s
    if err:
        rec["error"] = err
    if capture:
        rec.setdefault("_trace", {})["judge"] = {"change": (sink_c[0] if sink_c else None),
                                                 "stakes": (sink_s[0] if sink_s else None)}
    return rec


# ── parallel batch helpers ───────────────────────────────────────────────────


def _parallel(fn, items, max_workers=None):
    if not items:
        return []
    workers = max_workers or min(MAX_WORKERS, len(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in items}
        # preserve input order
        result_by_id = {}
        for fut in concurrent.futures.as_completed(futures):
            it = futures[fut]
            try:
                result_by_id[id(it)] = fut.result()
            except Exception as e:  # pragma: no cover - defensive
                it["error"] = str(e)
                result_by_id[id(it)] = it
    return [result_by_id[id(it)] for it in items]


def project_answers_batch(problem, framing, recs, model, m, timeout=120, capture=False,
                          evidence=None):
    return _parallel(
        lambda r: project_answers(problem, framing, r, model, m, timeout, capture, evidence),
        recs)


def judge_plan_change_batch(problem, framing, baseline_plan, recs, model, timeout=150,
                            capture=False):
    return _parallel(
        lambda r: judge_plan_change(problem, framing, baseline_plan, r, model, timeout,
                                    capture),
        recs)


def judge_plan_change_pairwise_batch(problem, framing, baseline_plan, recs, model, timeout=150,
                                     capture=False):
    return _parallel(
        lambda r: judge_plan_change_pairwise(problem, framing, baseline_plan, r, model, timeout,
                                             capture),
        recs)
