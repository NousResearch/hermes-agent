#!/usr/bin/env python3
"""pipeline.py — the model-calling stages of the information-gain skill.

Stages (each a role-specialized Ollama model, mostly via direct /api/chat raw
calls run in parallel):

    0. frame_and_plan   — restate goal/decision/success + a baseline plan (plan*_0)
    1. generate_questions — interrogate the problem into candidate questions
    2. project_answers  — plausible answers + probabilities + derivability  (parallel)
    3. judge_plan_change — per-answer Δplan and stakes vs the baseline plan  (parallel)

The pure scoring/ranking/selection math is in voi.py; this module only produces
the raw signals (answers, probabilities, Δplan, stakes) for it to score.

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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
OLLAMA_TAGS_URL = OLLAMA_URL.replace("/api/chat", "/api/tags")
MAX_WORKERS = int(os.environ.get("INFOGAIN_MAX_WORKERS", "8"))


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
        return {"content": content.strip(), "elapsed": time.time() - start, "error": None}
    except Exception as e:
        return {"content": "", "elapsed": time.time() - start, "error": str(e)}


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


def _call_json(model, prompt, timeout, num_predict, retries=1):
    """raw_chat + extract_json with one retry that nudges toward strict JSON."""
    last_err = None
    for attempt in range(retries + 1):
        content = prompt if attempt == 0 else (
            prompt + "\n\nReturn ONLY valid JSON. No prose, no markdown fences."
        )
        r = raw_chat(model, content, timeout=timeout, num_predict=num_predict)
        if r["error"]:
            last_err = r["error"]
            continue
        try:
            return extract_json(r["content"]), None
        except ValueError as e:
            last_err = f"{e} (raw: {r['content'][:160]!r})"
    return None, last_err


# ── prompt builders (separated so --dry-run can show them) ───────────────────


def frame_prompt(problem):
    return (
        "You are scoping an underspecified problem BEFORE any work begins.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        "Return ONLY a JSON object:\n"
        '{"goal": str, "decision": str, "success_criteria": [str], "baseline_plan": str}\n'
        "- goal: the underlying objective in one sentence.\n"
        "- decision: the main decision/approach that must be made.\n"
        "- success_criteria: 2-4 short bullet strings for a good outcome.\n"
        "- baseline_plan: the best recommended plan GIVEN ONLY THE PROBLEM AS STATED "
        "(assume the most likely interpretation of any ambiguity; 2-5 sentences). "
        "This baseline is what we measure information value against.\n"
        "Respond ONLY with the JSON object."
    )


def questions_prompt(problem, framing, n, avoid=None):
    avoid_block = ""
    if avoid:
        bullets = "\n".join(f"- {q}" for q in avoid)
        avoid_block = (
            "\nDo NOT repeat or paraphrase these already-considered questions:\n"
            f"{bullets}\n"
        )
    return (
        "You are interrogating an underspecified problem to find what is worth "
        "clarifying BEFORE doing the work.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"GOAL: {framing.get('goal', '')}\n"
        f"DECISION: {framing.get('decision', '')}\n"
        f"{avoid_block}\n"
        f"Propose {n} DISTINCT key questions whose answers are currently unknown and "
        "could change the recommended approach. Cover DIFFERENT hidden assumptions; "
        "avoid near-duplicates.\n\n"
        "Return ONLY a JSON object:\n"
        '{"questions": [{"question": str, "type": str, "why": str, "target": str}, ...]}\n'
        "- type: one of [scope, constraint, audience, data, integration, risk, "
        "success-metric, resource, assumption, other].\n"
        "- target: a SHORT label (2-5 words) naming the single hidden assumption / "
        "latent variable the question resolves. Two questions resolving the same "
        "latent MUST share the same target.\n"
        "Respond ONLY with the JSON object."
    )


def answers_prompt(problem, framing, question, m):
    return (
        "Project the plausible answers to a clarifying question about an "
        "underspecified problem.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"GOAL: {framing.get('goal', '')}\n"
        f"QUESTION: {question}\n\n"
        f"Enumerate the {m} most plausible DISTINCT answers. For each, estimate a "
        "probability (0-1) that it is the true answer given the problem. Also "
        "estimate whether the question is already answerable from the problem "
        "statement alone.\n\n"
        "Return ONLY a JSON object:\n"
        '{"derivable_prob": float, "answers": [{"answer": str, "prob": float}, ...]}\n'
        "- derivable_prob: 0-1, probability the question can be confidently answered "
        "from the problem statement as given (high = asking buys little).\n"
        f"- Provide 2 to {m} answers; probabilities need not sum to exactly 1.\n"
        "Respond ONLY with the JSON object."
    )


def judge_prompt(problem, framing, baseline_plan, question, answers):
    enumerated = "\n".join(
        f"{i + 1}. {a.get('answer', '')}" for i, a in enumerate(answers)
    )
    return (
        "Estimate how much each possible answer would change the recommended plan, "
        "and the cost of guessing wrong.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"GOAL: {framing.get('goal', '')}\n\n"
        "BASELINE PLAN (assuming the most likely interpretation):\n"
        f"{baseline_plan}\n\n"
        f"QUESTION: {question}\n\n"
        f"POSSIBLE ANSWERS:\n{enumerated}\n\n"
        "For EACH answer, in the SAME ORDER, judge two 0-1 scores:\n"
        "- delta_plan: how much the recommended plan would CHANGE if this answer is "
        "true (0 = identical plan, 1 = completely different approach).\n"
        "- stakes: the cost/harm of having proceeded on the BASELINE plan if this "
        "answer is actually true (0 = harmless, 1 = severe rework or failure).\n\n"
        "Return ONLY a JSON object:\n"
        '{"answers": [{"delta_plan": float, "stakes": float}, ...]}\n'
        "with exactly one entry per answer, in the given order.\n"
        "Respond ONLY with the JSON object."
    )


# ── stages ───────────────────────────────────────────────────────────────────


def frame_and_plan(problem, model, timeout=180):
    """Stage 0. Returns (framing_dict, error). framing has goal/decision/
    success_criteria/baseline_plan (always a dict, even on partial failure)."""
    obj, err = _call_json(model, frame_prompt(problem), timeout, num_predict=700)
    if not isinstance(obj, dict):
        return ({"goal": "", "decision": "", "success_criteria": [],
                 "baseline_plan": ""}, err or "framing returned non-object")
    obj.setdefault("goal", "")
    obj.setdefault("decision", "")
    obj.setdefault("success_criteria", [])
    obj.setdefault("baseline_plan", "")
    return obj, None


def generate_questions(problem, framing, model, n, avoid=None, timeout=180):
    """Stage 1. Returns (list_of_records, error). Each record: question/type/why/target."""
    obj, err = _call_json(model, questions_prompt(problem, framing, n, avoid),
                          timeout, num_predict=900)
    items = []
    if isinstance(obj, dict):
        items = obj.get("questions") or []
    elif isinstance(obj, list):
        items = obj
    out = []
    for q in items:
        if not isinstance(q, dict):
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        out.append({
            "question": text,
            "type": (q.get("type") or "other").strip(),
            "why": (q.get("why") or "").strip(),
            "target": (q.get("target") or "").strip(),
        })
    return out, (None if out else (err or "no questions generated"))


def project_answers(problem, framing, rec, model, m, timeout=120):
    """Stage 2 (single question). Mutates rec with answers[] + derivable_prob."""
    obj, err = _call_json(model, answers_prompt(problem, framing, rec["question"], m),
                          timeout, num_predict=600)
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
    return rec


def judge_plan_change(problem, framing, baseline_plan, rec, model, timeout=150):
    """Stage 3 (single question). Adds delta_plan + stakes to each answer in rec."""
    answers = rec.get("answers") or []
    if not answers:
        return rec
    obj, err = _call_json(
        model, judge_prompt(problem, framing, baseline_plan, rec["question"], answers),
        timeout, num_predict=500,
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


def project_answers_batch(problem, framing, recs, model, m, timeout=120):
    return _parallel(
        lambda r: project_answers(problem, framing, r, model, m, timeout), recs)


def judge_plan_change_batch(problem, framing, baseline_plan, recs, model, timeout=150):
    return _parallel(
        lambda r: judge_plan_change(problem, framing, baseline_plan, r, model, timeout),
        recs)
