#!/usr/bin/env python3
"""answerer.py — the grounded question-answerer + final responder (the `ask` seam).

Research ONE question with a full Hermes agent (`dispatch_single` = `hermes chat -q`,
isolated context) and distill the result to a 1-3 sentence fact-or-gap. Split out of
iterate.py so the model-dispatch coupling, the stdout-salvage workaround, and the
answer-artifact capture live in one place; iterate.py keeps the convergence loop and
re-exports these names for back-compat.

Artifact-beats-stdout (the drive.py / relentless-v2 lesson): when cfg["run_dir"] is set
AND the capability allows writes (cfg["answer_artifact_write"]), the answerer agent is
ALSO instructed to write its distilled answer to
    <run_dir>/answer-<fp(question)>.json   {"answer": "..."}
— a timeout can kill stdout after the file write landed, and dispatch_single's
is_api_error() heuristic can misclassify long legitimate answers (see _extract). The
artifact is read first; stdout parsing is the fallback. The found/NOT_FOUND judgment
stays in CODE either way (the artifact carries one answer string, never a verdict).
Under capability=read the instruction is OMITTED: the read directive says "do NOT
modify files" and a coherent prompt beats per-answer durability.

Stdlib + the ask skill's model_utils (resolved via ASK_SCRIPTS_DIR / HERMES_HOME).
"""

import hashlib
import json
import os
import re
import sys

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_ASK = os.environ.get("ASK_SCRIPTS_DIR") or os.path.join(
    _HOME, "skills", "productivity", "ask", "scripts")
sys.path.insert(0, _ASK)

try:
    from model_utils import dispatch_single, resolve_alias  # noqa: E402
    _HAVE_ASK = True
    _ASK_ERR = ""
except Exception as _e:  # the `ask` skill (model_utils) is required for live (non-mock) runs
    _HAVE_ASK = False
    _ASK_ERR = str(_e)
    dispatch_single = resolve_alias = None  # names always exist (re-export + test patching)


CANNOT_DECIDE = "CANNOT_DECIDE"
_JUDGE_HEDGE_RE = re.compile(
    r"cannot[\s_]derive|does\s+not\s+(specify|say|state|mention|indicate)|not\s+specified"
    r"|no\s+information|unclear\s+from|insufficient\s+(context|information)"
    r"|(prompt|context|spec)\s+(does\s?n[o']t|lacks)", re.IGNORECASE)


def fp(text):
    """Anti-flap fingerprint: case/whitespace/punctuation-insensitive identity hash
    (same rule as relentless-solve's harvest.fp — copied, not imported: a cross-skill
    import would invert the layering; these four lines ARE the contract)."""
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def qtext_of(question):
    """Ranked-question dict or bare string → the question text. The live loop passes the
    ranker's dict; tests and ad-hoc callers pass strings. (Interpolating the dict repr
    into the prompt was a real live bug — normalize at the boundary.)"""
    return question.get("question", "") if isinstance(question, dict) else (question or "")


def artifact_path(run_dir, qtext):
    """Absolute per-question answer artifact path (absolute because the answerer agent
    runs with its own cwd — answer_cwd — in the same container/host as this process)."""
    return os.path.abspath(os.path.join(run_dir, f"answer-{fp(qtext)}.json"))


def read_answer_artifact(path):
    """The agent-written answer string, or None (absent/malformed/empty ⇒ the caller
    falls back to stdout parsing)."""
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, ValueError):
        return None
    ans = obj.get("answer") if isinstance(obj, dict) else None
    return ans.strip() if isinstance(ans, str) and ans.strip() else None


def _artifact_instruction(path):
    return (f"\n\nWhen you have your final answer, ALSO write EXACTLY one JSON object to "
            f'{path}: {{"answer": "<your 1-3 sentence answer, or NOT_FOUND: <reason>>"}}. '
            f"Write the file even for NOT_FOUND — then reply with the same answer.")


def _strip_suggestion(text):
    """Drop a trailing `SUGGESTION:{...}` block (hermes's interactive next-step artifact)."""
    i = text.rfind("SUGGESTION:{")
    return text[:i].rstrip() if i != -1 else text


def _extract(r):
    """Best text from a dispatch_single result, as (text, error).

    Works around a false-positive in model_utils.is_api_error(): a legitimate response that is
    ABOUT errors / rate-limits / status codes can match ≥2 error keywords, so dispatch_single
    files the whole answer under `error` ("API error: <long text>") and blanks `content`. A long
    payload is a real response misclassified — recover it; a short one is a genuine API error.
    (Near-vestigial once the answer artifact lands, but kept as the stdout fallback.)
    """
    text = (r.get("content") or "").strip()
    if text:
        return _strip_suggestion(text), None
    err = (r.get("error") or "").strip()
    if err.startswith("API error: ") and len(err) > 200:  # long => real response, not an error
        return _strip_suggestion(err[len("API error: "):].strip()), None
    return "", (err or "empty response")


def grounded_answer(question, problem, evidence, cfg):
    """(found, text) — research one question with a full Hermes agent, distilled.

    Answer capture order: run_dir artifact (when enabled) → stdout via _extract. The
    NOT_FOUND parse applies identically to both sources."""
    qtext = qtext_of(question)
    facts = "\n".join(f"- {e}" for e in evidence) or "(none yet)"
    directive = cfg.get("answer_directive", "")
    head = (directive + "\n\n") if directive else ""
    prompt = (f"{head}TASK: {problem}\n\nEstablished so far:\n{facts}\n\n"
              f"Research and answer THIS question CONCISELY (1-3 sentences), using any tools you need:\n"
              f"  {qtext}\n\n"
              f"If you genuinely cannot determine it, reply EXACTLY: NOT_FOUND: <brief reason>.")
    run_dir = cfg.get("run_dir")
    use_artifact = bool(run_dir) and bool(cfg.get("answer_artifact_write", True))
    apath = artifact_path(run_dir, qtext) if use_artifact else None
    if use_artifact:
        prompt += _artifact_instruction(apath)
    r = dispatch_single(resolve_alias(cfg["answer_model"]), prompt, "", cfg["answer_toolsets"],
                        cfg["answer_max_turns"], cfg["answer_timeout"], cfg["answer_provider"],
                        cwd=cfg.get("answer_cwd"))
    text = read_answer_artifact(apath) if use_artifact else None
    if text is not None:
        text, err = _strip_suggestion(text), None
    else:
        text, err = _extract(r)
    if err:
        return False, f"research error: {err}"
    if "NOT_FOUND" in text.upper()[:48]:
        return False, text.split(":", 1)[-1].strip() if ":" in text else text
    return True, text


def triage_batch(problem, questions, evidence, cfg):
    """Route one round's questions in one no-tools model call; malformed output fails open."""
    blocks = []
    for i, question in enumerate(questions, 1):
        qtext = qtext_of(question)
        target = question.get("target", "") if isinstance(question, dict) else ""
        lines = [f"{i}. Question: {qtext}"]
        if target:
            lines.append(f"   Target: {target}")
        answers = question.get("answers", []) if isinstance(question, dict) else []
        for projected in answers[:2]:
            if not isinstance(projected, dict):
                continue
            lines.append(f"   Projected answer: {projected.get('answer', '')}")
            lines.append(f"   Delta plan: {projected.get('delta_plan', '')}")
        blocks.append("\n".join(lines))
    facts = "\n".join(f"- {item}" for item in evidence) or "(none yet)"
    prompt = (
        f"TASK: {problem}\n\nEstablished facts so far:\n{facts}\n\n"
        "Classify every numbered question using exactly one route:\n"
        "FINDABLE = an observable fact a tool-using agent could discover (in the repo, docs, "
        "the web, or by running something).\n"
        "JUDGMENT = a preference, taste, or decision with no discoverable ground truth.\n\n"
        + "\n\n".join(blocks)
        + '\n\nReply with a STRICT JSON array and nothing else: '
          '[{"i": <index>, "route": "FINDABLE"|"JUDGMENT"}]'
    )
    try:
        r = dispatch_single(resolve_alias(cfg["triage_model"]), prompt, "", "", None,
                            cfg["triage_timeout"], cfg["triage_provider"])
        text, err = _extract(r)
        if err:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return {}
    except Exception:
        return {}

    routes, seen = {}, set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        index, route = item.get("i"), item.get("route")
        if (not isinstance(index, int) or isinstance(index, bool)
                or index < 1 or index > len(questions) or index in seen
                or route not in ("FINDABLE", "JUDGMENT")):
            continue
        seen.add(index)
        routes[fp(qtext_of(questions[index - 1]))] = route
    return routes


def judgment_call(question, problem, evidence, cfg):
    """Make one conservative judgment, rejecting explicit and disguised non-decisions."""
    facts = "\n".join(f"- {item}" for item in evidence) or "(none yet)"
    prompt = (
        f"TASK: {problem}\n\nEstablished facts so far:\n{facts}\n\n"
        f"QUESTION: {qtext_of(question)}\n\n"
        "No discoverable fact exists for this question. Make a reasonable, CONSERVATIVE "
        "judgment call: prefer the reversible, standard, or least-surprising option. Reply "
        'with EXACTLY one JSON object: {"decision": "...", "rationale": "..."}. If it is '
        f"genuinely undecidable, reply exactly {CANNOT_DECIDE}: <reason> instead of JSON."
    )
    try:
        r = dispatch_single(resolve_alias(cfg["judge_model"]), prompt, "", "", None,
                            cfg["judge_timeout"], cfg["judge_provider"])
        text, err = _extract(r)
    except Exception as exc:
        return False, "", str(exc) or "judgment dispatch failed"
    if err:
        return False, "", err
    raw = text.strip()
    if raw.startswith(CANNOT_DECIDE):
        reason = raw[len(CANNOT_DECIDE):].lstrip(": ").strip()
        return False, "", reason or raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return False, "", str(exc) or raw
    if (not isinstance(parsed, dict)
            or not isinstance(parsed.get("decision"), str)
            or not isinstance(parsed.get("rationale"), str)):
        return False, "", "response must contain string decision and rationale"
    decision, rationale = parsed["decision"], parsed["rationale"]
    if _JUDGE_HEDGE_RE.search(decision):
        return False, "", rationale or decision
    return True, decision, rationale


def respond(problem, evidence, cfg):
    """Final response over the enriched context (no tools — synthesize from established facts)."""
    facts = "\n".join(f"- {e}" for e in evidence) or "(none)"
    prompt = (f"TASK: {problem}\n\nEstablished facts and known gaps:\n{facts}\n\n"
              f"Produce the best possible response to the task using what's established. "
              f"State any assumptions you make for unresolved gaps. Be direct and useful.")
    if any("(assumed:" in e or "(derived" in e for e in evidence):
        prompt += (
            " Treat derived facts as inferred, not observed: they were not directly verified. "
            "End the response with a `## Assumptions` section listing each decision, its "
            "rationale, and what to change if the assumption is wrong, followed by a "
            "`## Known gaps` section."
        )
    r = dispatch_single(resolve_alias(cfg["responder_model"]), prompt, "", cfg.get("responder_toolsets", ""),
                        None, cfg["responder_timeout"], cfg["responder_provider"],
                        cwd=cfg.get("responder_cwd"))
    text, err = _extract(r)
    return text or f"(no response: {err})"


def refine_prompt(problem, evidence, cfg):
    """Rewrite the original task using every fact, decision, and known gap established."""
    facts = "\n".join(f"- {e}" for e in evidence) or "(none)"
    prompt = (
        f"ORIGINAL PROMPT:\n{problem}\n\nEstablished facts, decisions, and known gaps:\n"
        f"{facts}\n\nRewrite the ORIGINAL PROMPT as a self-contained, improved prompt. Fold every "
        "answered fact above, whether researched or derived, into explicit constraints or "
        "context. State every assumed decision as an explicit choice made in the prompt, not "
        "merely as a footnote. Preserve the original intent and do not invent scope beyond what "
        "the evidence establishes. Always end the rewritten prompt with a `## Assumptions` "
        "section listing every assumed decision and its rationale in terms that let a human "
        "reviewer veto it."
    )
    if any("(known gap:" in e for e in evidence):
        prompt += (
            " Also end with a `## Open questions` section listing every unresolved gap as "
            '"unspecified — implementer may choose" or as a carried-forward question.'
        )
    r = dispatch_single(resolve_alias(cfg["responder_model"]), prompt, "",
                        cfg.get("responder_toolsets", ""), None, cfg["responder_timeout"],
                        cfg["responder_provider"], cwd=cfg.get("responder_cwd"))
    text, err = _extract(r)
    return text or f"(no refined prompt: {err})"
