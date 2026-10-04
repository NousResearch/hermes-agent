"""The TypeSafe (Jev) provider: its wire format, and a client that fails closed.

The request and response shapes follow https://docs.typesafe.ai/api.md exactly::

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <key>
    {"state": ..., "model": "jev-latest", "questions": {id: {"type", "instructions", "criteria"}}}

    -> {"model": "jev-1.13.0", "answers": {id: {...}}, "usage": {"input_tokens", "output_tokens"}}

All questions go in one request (the documented parallel-questions pattern). A noul's
criteria are optional and not sent; a choice's are its options; a score's are its levels in
order, and its answer's probabilities come back keyed by level index.

**This file is copied into the runtime** beside the policy plugin, so everything above the
``TypesafeProvider`` class is standard library only — ``urllib``, ``json``, ``time``. The
class itself is the control plane's typed wrapper and imports NOVA lazily, inside a method
the worker never calls.

Every failure raises ``TriageProviderError``: a missing key, a non-200 status (401, 422,
429, 529, anything), a body that is not the documented shape, an answer missing for any
question, a probability outside [0, 1], a missing model version. There are no retries here
— the caller is in the path of a tool call and escalates to a person instead.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

NAME = "typesafe"
#: The question types this provider answers. Others (the recipient) are facts NOVA looks up
#: itself, and are neither sent nor expected back.
ANSWERS = ("noul", "choice", "score")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
#: The SDK's own variable name, so a key already set for TypeSafe's tools is found here too.
API_KEY_ENV = "TYPESAFE_API_KEY"
#: Bodies larger than this are not read: no documented answer set comes near it.
_MAX_RESPONSE_BYTES = 1 << 20


class TriageProviderError(Exception):
    """No usable answer. ``reason`` carries no request content and is safe to record."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _wire_question(question: Mapping[str, Any]) -> Dict[str, Any]:
    kind = question.get("type")
    if kind == "noul":
        return {"type": "noul", "instructions": question["instructions"]}
    if kind == "choice":
        return {"type": "choice", "instructions": question["instructions"],
                "criteria": {option: None for option in question["options"]}}
    if kind == "score":
        return {"type": "score", "instructions": question["instructions"],
                "criteria": list(question["levels"])}
    raise TriageProviderError(f"unknown question type {kind!r}")


def build_request(state: Any, questions: Sequence[Mapping[str, Any]], *, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """The documented request body for one call carrying every question."""
    return {
        "state": state,
        "model": model,
        "questions": {str(q["id"]): _wire_question(q) for q in questions if q.get("type") in ANSWERS},
    }


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TriageProviderError("an answer carried a non-numeric probability")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise TriageProviderError("an answer carried a probability outside [0, 1]")
    return value


def _normalize(question: Mapping[str, Any], answer: Any) -> Dict[str, Any]:
    kind = question["type"]
    if not isinstance(answer, Mapping) or answer.get("type") != kind:
        raise TriageProviderError(f"answer {question['id']!r} is missing or of the wrong type")
    if kind == "noul":
        return {"type": "noul", "p": _probability(answer.get("noul"))}
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise TriageProviderError(f"answer {question['id']!r} has no probabilities")
    confidence = _probability(answer.get("confidence"))
    if kind == "choice":
        options = list(question["options"])
        if set(probabilities) != set(options) or answer.get("choice") not in options:
            raise TriageProviderError(f"answer {question['id']!r} names options that were not asked")
        return {"type": "choice", "choice": answer["choice"], "confidence": confidence,
                "probabilities": {o: _probability(probabilities[o]) for o in options}}
    levels = list(question["levels"])
    expected = {str(index) for index in range(len(levels))}
    if set(map(str, probabilities)) != expected:
        raise TriageProviderError(f"answer {question['id']!r} does not cover the levels asked")
    by_index = {str(k): v for k, v in probabilities.items()}
    return {"type": "score", "confidence": confidence,
            "probabilities": [_probability(by_index[str(i)]) for i in range(len(levels))]}


def parse_response(body: Any, questions: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, int]]:
    """``(normalized answers, model version, usage)``, or raise on anything undocumented."""
    if not isinstance(body, Mapping):
        raise TriageProviderError("the response is not a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise TriageProviderError("the response does not say which model answered")
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TriageProviderError("the response has no answers")
    normalized = {str(q["id"]): _normalize(q, answers.get(str(q["id"])))
                  for q in questions if q.get("type") in ANSWERS}
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return normalized, model.strip(), {
        key: int(usage[key]) for key in ("input_tokens", "output_tokens")
        if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
    }


def request_answers(
    api_key: str,
    state: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    timeout: float = 2.0,
    endpoint: str = ENDPOINT,
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, int], int]:
    """One call: ``(normalized answers, model version, usage, latency_ms)``. Raises on failure.

    ``timeout`` bounds each socket operation; the caller adds a hard deadline over the whole
    call (``triage.within_deadline``).
    """
    if not api_key or not api_key.strip():
        raise TriageProviderError(f"no API key ({API_KEY_ENV} is not set)")
    if not any(q.get("type") in ANSWERS for q in questions):
        raise TriageProviderError("no questions to ask")
    payload = json.dumps(build_request(state, questions, model=model), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — fixed https endpoint
            status = getattr(response, "status", 200)
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise TriageProviderError(f"the provider answered HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise TriageProviderError(f"the provider could not be reached ({type(exc).__name__})") from None
    latency_ms = int((time.monotonic() - started) * 1000)
    if status != 200:
        raise TriageProviderError(f"the provider answered HTTP {status}")
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise TriageProviderError("the response was too large")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise TriageProviderError("the response is not JSON") from None
    normalized, version, usage = parse_response(body, questions)
    return normalized, version, usage, latency_ms


class TypesafeProvider:
    """The control plane's typed client. The worker uses :func:`request_answers` directly."""

    name = NAME

    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 2.0,
                 endpoint: str = ENDPOINT, model: str = DEFAULT_MODEL):
        import os

        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self.timeout = float(timeout)
        self.endpoint = endpoint
        self.model = model

    def ask(self, state: Any, questions: Sequence[Any]):
        from nova.autonomy.providers.base import ProviderError, answers_from
        from nova.autonomy.triage import within_deadline

        compiled = [q.to_dict() if hasattr(q, "to_dict") else dict(q) for q in questions]
        failure: Dict[str, str] = {}

        def call():
            try:
                return request_answers(self._api_key, state, compiled, timeout=self.timeout,
                                       endpoint=self.endpoint, model=self.model)
            except TriageProviderError as exc:
                failure["reason"] = exc.reason
                return None

        result = within_deadline(call, self.timeout)
        if result is None:
            raise ProviderError(failure.get("reason") or f"no answer within {self.timeout:g}s")
        normalized, version, usage, latency_ms = result
        return answers_from(normalized, provider=self.name, model_version=version,
                            latency_ms=latency_ms, usage=usage)
