"""Track response.model changes across LLM calls so the context
compressor can rebudget to the actual backend's context length.

Many OpenAI-compatible endpoints expose router-style model ids such as
``openrouter/auto`` or ``:free``-suffixed names where the concrete
backend selected per request varies (Llama 3.3, Qwen, DeepSeek, etc.).
Each backend has its own context window. Without observing the live
``response.model`` value, Hermes keeps its compressor calibrated to
whatever the operator configured at startup, which is usually wrong
for whichever backend the router actually picked.

``AdaptiveContextTracker`` is a tiny state machine: it remembers the last
observed model id and reports back when it changes. It does *not* perform
the ``context_length`` lookup (``agent.model_metadata`` already owns that)
and it does *not* mutate the compressor. The module-level helpers wire it
into the agent: ``init_adaptive_context`` (init, gated on
``compression.adaptive_context_window``), ``rebudget_on_backend_change``
(after each successful provider response) and ``usage_line`` (/usage).
"""
from __future__ import annotations

import logging
import time
from typing import Any

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


def init_adaptive_context(agent: Any, agent_cfg: dict) -> None:
    """Attach ``agent._adaptive_context`` when ``compression.adaptive_context_window`` is on.

    Off by default: the tracker only helps router-style model ids (``openrouter/auto``,
    ``:free`` variants, fallback chains) whose concrete backend, and so whose context
    window, varies per request.
    """
    from utils import is_truthy_value

    compression_cfg = agent_cfg.get("compression") or {}
    enabled = is_truthy_value(compression_cfg.get("adaptive_context_window"), default=False)
    agent._adaptive_context = AdaptiveContextTracker() if enabled else None


def rebudget_on_backend_change(agent: Any, response: Any) -> None:
    """Re-budget the compressor to the new backend's window when ``response.model`` changed."""
    tracker = getattr(agent, "_adaptive_context", None)
    if tracker is None or response is None:
        return
    new_model = tracker.observe(getattr(response, "model", None))
    compressor = getattr(agent, "context_compressor", None)
    if not new_model or compressor is None:
        return
    api_key = getattr(agent, "api_key", "")
    # The lookup may hit the network (endpoint /models, live catalogs); a failure here
    # must never interrupt the turn, the compressor just keeps its current budget.
    try:
        from agent.model_metadata import get_model_context_length

        new_ctx = get_model_context_length(
            new_model, base_url=agent.base_url, api_key=api_key, provider=agent.provider,
        )
    except Exception:
        logger.debug("adaptive-context: context length lookup for %r failed", new_model, exc_info=True)
        return
    if new_ctx:
        compressor.update_model(
            model=new_model, context_length=new_ctx, base_url=agent.base_url,
            api_key=api_key, provider=agent.provider, api_mode=agent.api_mode,
        )


def usage_line(tracker: AdaptiveContextTracker) -> str:
    """The localized "Adaptive ctx:" /usage line (gateway markdown; the CLI strips backticks)."""
    from agent.i18n import t

    summary = tracker.summary()
    if not summary["last_seen"]:
        return t("gateway.usage.label_adaptive_context_pending")
    if summary["change_count"] == 0:
        return t("gateway.usage.label_adaptive_context_baseline", model=summary["last_seen"])
    return t(
        "gateway.usage.label_adaptive_context_changes",
        model=summary["last_seen"], count=summary["change_count"],
    )
