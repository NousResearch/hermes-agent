"""Summary-hook dispatch and cancellation rollback for context compression."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.auxiliary_client import AuxiliaryExplicitCancellation

if TYPE_CHECKING:
    from agent.context_compressor import _HandoffScan


def _accepts_keyword_argument(callable_obj: Any, name: str) -> bool:
    """Return whether an inspectable callable accepts ``name`` as a keyword."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class SummaryDispatchMixin:
    def _summarize_window(
        self, messages: List[Dict[str, Any]], turns_to_summarize: List[Dict[str, Any]], scan: "_HandoffScan",
        focus_topic: Optional[str], memory_context: str, bypass_cooldown: bool,
    ) -> Optional[str]:
        """Run the summary LLM; a cancellation rolls back the handoff scan's self-heal mutation first."""
        # Focus-topic derivation scans user turns; only pay when a summary is generated.
        summary_kwargs: Dict[str, Any] = {
            "focus_topic": focus_topic or self._derive_auto_focus_topic(messages),
            "memory_context": memory_context,
        }
        if _accepts_keyword_argument(self._generate_summary, "bypass_cooldown"):
            summary_kwargs["bypass_cooldown"] = bypass_cooldown
        try:
            return self._generate_summary(turns_to_summarize, **summary_kwargs)
        except AuxiliaryExplicitCancellation:
            # Cancellation is a true no-op: restore the scan's mutation before the exception escapes.
            self._previous_summary = scan.previous_summary_before
            self._summary_has_user_turn = scan.has_user_turn_before
            raise
