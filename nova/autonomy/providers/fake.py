"""A deterministic provider for tests and demos. Never calls anything.

Answers are scripted per question id. Anything not scripted gets the *safest* answer (a
noul of 0.0, the first allowed choice, the lowest level, full confidence) so a demo with
the fake provider shows triage passing, and a test scripts only the answer it is about.

It records every request it was given, so a test can assert on exactly what would have
left the environment — the point of ``data: metadata_only``.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

from nova.autonomy.providers.base import Answers, ProviderError, answers_from
from nova.autonomy.triage import safe_answers


class FakeProvider:
    name = "fake"

    def __init__(self, answers: Optional[Mapping[str, Mapping[str, Any]]] = None, *,
                 model_version: str = "fake-1", fail: str = "", delay_seconds: float = 0.0):
        #: Scripted normalized answers by question id (see ``triage`` for the shape).
        self.scripted = dict(answers or {})
        self.model_version = model_version
        #: When set, every call raises ``ProviderError(fail)``.
        self.fail = fail
        self.delay_seconds = float(delay_seconds)
        self.requests: list[dict[str, Any]] = []

    def ask(self, state: Any, questions: Sequence[Any]) -> Answers:
        compiled = [q.to_dict() if hasattr(q, "to_dict") else dict(q) for q in questions]
        self.requests.append({"state": state, "questions": compiled})
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail:
            raise ProviderError(self.fail)
        safe = safe_answers(compiled)
        normalized = {q["id"]: dict(self.scripted.get(q["id"]) or safe[q["id"]]) for q in compiled if q["id"] in safe}
        return answers_from(normalized, provider=self.name, model_version=self.model_version,
                            latency_ms=int(self.delay_seconds * 1000))
