"""Track response.model changes across LLM calls so the context
compressor can rebudget to the actual backend's context length.

Many OpenAI-compatible endpoints expose router-style model ids such as
``openrouter/auto`` or ``:free``-suffixed names where the concrete
backend selected per request varies (Llama 3.3, Qwen, DeepSeek, …).
Each backend has its own context window. Without observing the live
``response.model`` value, Hermes keeps its compressor calibrated to
whatever the operator configured at startup — which is usually wrong
for whichever backend the router actually picked.

This module is a tiny state machine: it remembers the last observed
model id and reports back when it changes. It does *not* perform the
``context_length`` lookup (``agent.model_metadata`` already owns that)
and it does *not* mutate the compressor. The caller decides what to do
with the change signal.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class AdaptiveContextTracker:
    """Stateful observer that fires when ``response.model`` changes.

    Typical use::

        tracker = AdaptiveContextTracker()
        new_model = tracker.observe(getattr(response, "model", None))
        if new_model is not None:
            ctx = get_model_context_length(new_model, ...)
            compressor.update_model(model=new_model, context_length=ctx, ...)
    """

    def __init__(self) -> None:
        self._last_seen: str | None = None
        self._last_changed_at: float = 0.0
        self._change_count: int = 0

    def observe(self, response_model: str | None) -> str | None:
        """Record the model id reported by an LLM response.

        Returns the new id when it differs from the previously seen
        value, so callers know to rebudget. Returns ``None`` on the
        first observation, when the value is unchanged, or when the
        input is missing/non-string (defensive: response objects in
        some adapters may not carry ``model``).
        """
        if not response_model or not isinstance(response_model, str):
            return None
        if self._last_seen is None:
            # First observation: adopt silently. The agent was already
            # configured with some model id at startup; this is the
            # baseline against which subsequent changes are detected.
            self._last_seen = response_model
            return None
        if response_model == self._last_seen:
            return None
        previous = self._last_seen
        self._last_seen = response_model
        self._last_changed_at = time.monotonic()
        self._change_count += 1
        logger.info(
            "adaptive-context: backend changed %r -> %r (change #%d)",
            previous, response_model, self._change_count,
        )
        return response_model

    @property
    def last_seen(self) -> str | None:
        return self._last_seen

    @property
    def change_count(self) -> int:
        return self._change_count

    def summary(self) -> dict:
        """Snapshot of tracker state for UX surfaces (e.g. /usage).

        Returns a plain dict so callers don't have to reach into
        private attributes. ``seconds_since_last_change`` is ``None``
        until the first transition is observed.
        """
        seconds_since: float | None = None
        if self._change_count > 0 and self._last_changed_at > 0:
            seconds_since = max(0.0, time.monotonic() - self._last_changed_at)
        return {
            "last_seen": self._last_seen,
            "change_count": self._change_count,
            "seconds_since_last_change": seconds_since,
        }
