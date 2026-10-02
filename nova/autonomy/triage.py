"""The triage rule: typed answers to atomic questions, combined into a verdict by code.

This module is **copied verbatim into the runtime** beside the policy plugin, as
``_decide.py`` is, so the rule a reviewer reads here and the rule a worker applies are the
same code. It therefore imports only the standard library and works on plain dictionaries
— the compiled question set from the policy document and the provider's normalized answers.

**The model never decides alone.** A provider answers questions; it does not say "allow".
:func:`combine` turns the answers into ``auto_ok`` only when every question passes its own
configured threshold, and into ``escalate`` — with the questions that failed and why — in
every other case, including an answer that is missing, malformed or of the wrong type. An
empty question set escalates: nothing was checked, so nothing was shown to be safe.

Normalized answers, which every provider adapter produces::

    noul:   {"type": "noul",   "p": 0.03}
    choice: {"type": "choice", "choice": "internal", "probabilities": {...}, "confidence": 0.95}
    score:  {"type": "score",  "probabilities": [0.97, 0.03, 0.0, 0.0], "confidence": 0.94}

A score's probabilities are a list in level order. A noul has no confidence: its provider
reports the probability alone, so a noul passes on ``p < block_above`` and has no minimum
confidence to meet. Choice and score carry one, and must meet their ``min_confidence``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

AUTO_OK = "auto_ok"
ESCALATE = "escalate"

NOUL, CHOICE, SCORE = "noul", "choice", "score"
QUESTION_TYPES = (NOUL, CHOICE, SCORE)


def _probability(value: Any) -> Optional[float]:
    """A number in [0, 1], or None. Booleans and NaN are not probabilities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or value < 0.0 or value > 1.0:
        return None
    return value


def _check_noul(question: Mapping[str, Any], answer: Mapping[str, Any]) -> Optional[str]:
    p = _probability(answer.get("p"))
    if p is None:
        return "no valid probability"
    limit = float(question["block_above"])
    if p >= limit:
        return f"probability {p:.2f} is at or above the limit {limit:.2f}"
    return None


def _check_choice(question: Mapping[str, Any], answer: Mapping[str, Any]) -> Optional[str]:
    options = list(question["options"])
    allowed = set(question["allowed"])
    chosen = answer.get("choice")
    probabilities = answer.get("probabilities")
    confidence = _probability(answer.get("confidence"))
    if chosen not in options or not isinstance(probabilities, Mapping) or confidence is None:
        return "no valid answer"
    values = {option: _probability(probabilities.get(option)) for option in options}
    if any(value is None for value in values.values()):
        return "no valid probabilities"
    minimum = float(question["min_confidence"])
    if chosen not in allowed:
        return f"answer {chosen!r} is not one of the allowed {sorted(allowed)}"
    if confidence < minimum:
        return f"confidence {confidence:.2f} is below the minimum {minimum:.2f}"
    inside = sum(values[option] for option in allowed)
    if inside < minimum:
        return f"probability of an allowed answer {inside:.2f} is below the minimum {minimum:.2f}"
    return None


def _check_score(question: Mapping[str, Any], answer: Mapping[str, Any]) -> Optional[str]:
    levels = list(question["levels"])
    allowed_index = levels.index(question["max_allowed_level"])
    probabilities = answer.get("probabilities")
    confidence = _probability(answer.get("confidence"))
    if not isinstance(probabilities, Sequence) or isinstance(probabilities, (str, bytes)):
        return "no valid answer"
    values = [_probability(value) for value in probabilities]
    if len(values) != len(levels) or any(value is None for value in values) or confidence is None:
        return "no valid answer"
    minimum = float(question["min_confidence"])
    likeliest = max(range(len(values)), key=lambda i: values[i])
    if likeliest > allowed_index:
        return (f"rated {levels[likeliest]!r}, above the allowed {levels[allowed_index]!r}")
    if confidence < minimum:
        return f"confidence {confidence:.2f} is below the minimum {minimum:.2f}"
    # The weighted score can sit inside the allowed band while real mass sits above it;
    # what matters is how likely it is that the level really is at most the allowed one.
    within = sum(values[: allowed_index + 1])
    if within < minimum:
        return (f"probability of {levels[allowed_index]!r} or lower {within:.2f} is below the "
                f"minimum {minimum:.2f}")
    return None


_CHECKS: Dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Optional[str]]] = {
    NOUL: _check_noul, CHOICE: _check_choice, SCORE: _check_score,
}


def combine(questions: Sequence[Mapping[str, Any]], answers: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``{"verdict", "failed": [{"id", "why"}]}`` for one call.

    ``answers`` of None means the provider gave none (timeout, outage, bad response), which
    escalates every question rather than none of them.
    """
    if not questions:
        return {"verdict": ESCALATE, "failed": [{"id": "", "why": "no questions are configured"}]}
    if answers is None:
        return {"verdict": ESCALATE, "failed": [{"id": "", "why": "provider_unavailable"}]}
    failed: List[Dict[str, str]] = []
    for question in questions:
        qid = str(question.get("id", ""))
        answer = answers.get(qid)
        kind = question.get("type")
        if kind not in _CHECKS:
            failed.append({"id": qid, "why": f"unknown question type {kind!r}"})
            continue
        if not isinstance(answer, Mapping) or answer.get("type") != kind:
            failed.append({"id": qid, "why": "no valid answer"})
            continue
        try:
            why = _CHECKS[kind](question, answer)
        except Exception as exc:  # noqa: BLE001 — a rule that cannot be applied did not pass
            why = f"could not be checked ({type(exc).__name__})"
        if why:
            failed.append({"id": qid, "why": why})
    return {"verdict": AUTO_OK if not failed else ESCALATE, "failed": failed}


def within_deadline(fn: Callable[[], Any], seconds: float) -> Optional[Any]:
    """``fn()``'s result, or None if it raised or did not finish within ``seconds``.

    A hard deadline on the whole call, not per socket read: a provider that trickles bytes
    must not hold a tool call open. The thread is abandoned, never joined past the deadline,
    and there are no retries — the hook is in the path of every escalated call.
    """
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException:  # noqa: BLE001 — any failure is "no answer"
            box["value"] = None

    worker = threading.Thread(target=run, name="triage-call", daemon=True)
    worker.start()
    worker.join(max(0.0, float(seconds)))
    if worker.is_alive():
        return None
    return box.get("value")


# -- what is sent ---------------------------------------------------------------------
#
# ``metadata_only`` (the default) sends the tool, the business action, the platform a
# message goes to, and size signals per argument — never a value. ``full_args`` sends the
# arguments themselves. In both, anything that looks like a credential is replaced before
# it is put in the state, and so is any value of this process's own secret-named
# environment variables (the agent's ``.env``), whatever it looks like.

METADATA_ONLY, FULL_ARGS = "metadata_only", "full_args"
REDACTED = "[redacted]"

_SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_-]{16,}",                       # OpenAI-style and many others
    r"sk-ant-[A-Za-z0-9_-]{16,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",                  # GitHub
    r"xox[abposr]-[A-Za-z0-9-]{10,}",               # Slack
    r"AKIA[0-9A-Z]{16}",                            # AWS access key id
    r"AIza[0-9A-Za-z_-]{30,}",                      # Google API key
    r"apikey_[A-Za-z0-9_]{20,}",
    r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b",             # Telegram bot token
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",  # JWT
    r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S+",
)]
_SECRET_KEY_NAME = re.compile(r"(?i)(pass(word)?|secret|token|api[_-]?key|credential|private[_-]?key|auth)")
_SECRET_ENV_NAME = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)")


def secret_values(environ: Mapping[str, str]) -> List[str]:
    """Values of secret-named variables, long enough that replacing them cannot erase prose."""
    return sorted({v for k, v in environ.items() if _SECRET_ENV_NAME.search(k) and len(v or "") >= 8},
                  key=len, reverse=True)


def redact(value: Any, secrets: Sequence[str] = ()) -> Any:
    """``value`` with every credential-shaped string, and every known secret, replaced."""
    if isinstance(value, str):
        for secret in secrets:
            if secret and secret in value:
                value = value.replace(secret, REDACTED)
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    if isinstance(value, Mapping):
        return {str(k): (REDACTED if _SECRET_KEY_NAME.search(str(k)) and isinstance(v, (str, int, float))
                         else redact(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    return value


def _shape(value: Any) -> Dict[str, Any]:
    """Size signals for one argument, with nothing of its content."""
    if isinstance(value, str):
        return {"type": "text", "chars": len(value), "lines": value.count("\n") + 1 if value else 0,
                "links": len(re.findall(r"https?://", value)),
                "email_addresses": len(re.findall(r"[^\s@]+@[^\s@]+\.[^\s@]+", value)),
                "digits": sum(ch.isdigit() for ch in value),
                "attachments": value.count("MEDIA:")}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, Mapping):
        return {"type": "object", "keys": len(value)}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "items": len(value)}
    return {"type": "other"}


def _platform(args: Mapping[str, Any]) -> str:
    """The channel a message goes to, from a ``platform:recipient`` target. Never the recipient."""
    target = args.get("target")
    if isinstance(target, str) and ":" in target:
        head = target.split(":", 1)[0].strip().lower()
        if re.match(r"^[a-z][a-z0-9_-]{0,31}$", head):
            return head
    return ""


def build_state(tool: str, action: str, args: Any, data: str, secrets: Sequence[str] = ()) -> Dict[str, Any]:
    """What the provider is shown about one call. Unknown ``data`` is treated as metadata only."""
    args = args if isinstance(args, Mapping) else {}
    state: Dict[str, Any] = {"tool": tool, "business_action": action, "data": METADATA_ONLY}
    platform = _platform(args)
    if platform:
        state["platform"] = platform
    if data == FULL_ARGS:
        state["data"] = FULL_ARGS
        state["arguments"] = redact(dict(args), secrets)
    else:
        state["arguments"] = {str(k): _shape(v) for k, v in sorted(args.items(), key=lambda kv: str(kv[0]))}
    return state


def digest(value: Any) -> str:
    """A stable fingerprint of ``value``: what the audit log keeps instead of content."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_answers(questions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The fake provider's answers: the safest possible answer to each question."""
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        if q.get("type") == NOUL:
            out[q["id"]] = {"type": NOUL, "p": 0.0}
        elif q.get("type") == CHOICE:
            pick = list(q["allowed"])[0]
            out[q["id"]] = {"type": CHOICE, "choice": pick, "confidence": 1.0,
                            "probabilities": {o: (1.0 if o == pick else 0.0) for o in q["options"]}}
        elif q.get("type") == SCORE:
            out[q["id"]] = {"type": SCORE, "confidence": 1.0,
                            "probabilities": [1.0] + [0.0] * (len(q["levels"]) - 1)}
    return out


def explain(result: Mapping[str, Any], *, mode: str, proceed: bool) -> str:
    """One line for the person deciding: what triage made of this call, and why."""
    failed = list(result.get("failed") or ())
    if mode == "shadow":
        head = ("Triage (shadow) would have let this through: every check passed"
                if result.get("verdict") == AUTO_OK and not failed
                else "Triage (shadow) would have asked a person anyway")
    elif proceed:
        head = "Triage let this through: every check passed"
    else:
        head = "Triage asked a person"
    if failed:
        reasons = "; ".join(f"{f.get('id') or 'triage'}: {f.get('why')}" for f in failed[:4])
        more = f" (+{len(failed) - 4} more)" if len(failed) > 4 else ""
        return f"{head} — {reasons}{more}."
    return head + "."
