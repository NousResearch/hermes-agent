"""LLM judge core + JSON schema validation for its structured output."""

import json
import time
import urllib.request

from . import crash_filter
from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Structured output contract (AC: judge returns pass/fail verdict, confidence
# score, and explicit reasoning for failures).
# ---------------------------------------------------------------------------
VERDICTS = ("pass", "fail", "skip", "ambiguous")


def _validate(payload: dict) -> tuple[bool, str]:
    """Return (ok, error). Validates the judge's structured output."""
    if not isinstance(payload, dict):
        return False, "output is not a JSON object"
    v = payload.get("verdict")
    if v not in VERDICTS:
        return False, f"verdict '{v}' not in {VERDICTS}"
    if not isinstance(payload.get("crash_related"), bool):
        return False, "crash_related must be boolean"
    conf = payload.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return False, f"confidence {conf!r} not in [0,1]"
    r = payload.get("reasoning")
    if not isinstance(r, list) or not all(isinstance(x, str) for x in r):
        return False, "reasoning must be a list of strings"
    # For failures, require explicit reasoning.
    if v == "fail" and not r:
        return False, "fail verdict requires explicit reasoning"
    return True, ""


class JudgeError(Exception):
    pass


class LLMJudge:
    """Calls an OpenAI-compatible chat-completions endpoint, rotating credentials
    and retrying on transient failures (rate limit, empty content)."""

    def __init__(self, base_url, api_key, model, timeout=90, max_retries=3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _call(self, system, user) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
        }).encode()
        last_err = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content
                last_err = JudgeError("LLM returned empty content (retryable)")
            except Exception as e:  # noqa: BLE001 - retry transient
                last_err = e
            if attempt < self.max_retries:
                time.sleep(2.0 * (attempt + 1))
        raise JudgeError(f"LLM call failed after {self.max_retries + 1} tries: {last_err}")

    @staticmethod
    def _parse(content: str) -> dict:
        if not content or not content.strip():
            raise JudgeError("judge returned empty content")
        content = content.strip()
        # Strip code fences if the model wrapped JSON in ```json ... ```
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Tolerate trailing text after the JSON object.
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                return json.loads(content[start:end])
            except Exception:
                raise JudgeError("judge returned non-JSON output")


def judge_run(expected_behaviour: str, log_text: str, judge: LLMJudge) -> dict:
    """Full pipeline: crash-filter first, then LLM judge only if semantic.

    Returns the structured verdict dict with a top-level 'crash_related' flag
    always set (True for crash-routed runs, False for semantic-judged runs).
    """
    started = time.time()
    filt = crash_filter.classify(log_text)

    if filt["crash_related"]:
        return {
            "verdict": "skip",
            "crash_related": True,
            "confidence": 1.0,
            "reasoning": [f"crash_related: {filt['reason']}"],
            "summary": "Routed to crash suite (not a semantic regression).",
            "crash_markers": filt["markers"],
            "elapsed_s": round(time.time() - started, 3),
        }

    user = USER_PROMPT_TEMPLATE.format(expected=expected_behaviour, logs=log_text)
    content = judge._call(SYSTEM_PROMPT, user)
    payload = judge._parse(content)
    ok, err = _validate(payload)
    if not ok:
        raise JudgeError(f"judge output failed validation: {err}. raw={content[:200]}")
    payload["elapsed_s"] = round(time.time() - started, 3)
    return payload
