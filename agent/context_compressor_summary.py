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
    def _adopt_main_api_key(self, api_key: Any, *, provider: str = "") -> bool:
        """Adopt a live same-provider credential onto the compressor snapshot.

        Returns True when the snapshot was replaced. The fallen-back pin must not omit
        an OAuth-shaped key that came from the live client or published runtime (#138469).
        """
        live_provider = str(provider or "").strip().lower()
        self_provider = str(getattr(self, "provider", "") or "").strip().lower()
        if not live_provider or live_provider != self_provider:
            return False
        live_key = api_key
        if live_key is None or live_key == "":
            return False
        if isinstance(live_key, str):
            live_key = live_key.strip()
            if not live_key:
                return False
        self.api_key = live_key
        self._live_main_credential_adopted = True
        return True

    def _adopt_live_main_api_key(self) -> bool:
        """Replace a stale compressor snapshot with this turn's published main credential.

        After fallback the pin copies ``self.api_key``. That snapshot can already have diverged
        from the working client when the main loop never 401'd (#138469, gap after #121756).
        Read only the context-local runtime so a compat mirror from another session cannot leak in.
        """
        from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT
        live = _RUNTIME_MAIN_CONTEXT.get() or {}
        if isinstance(live, dict) and self._adopt_main_api_key(
            live.get("api_key"), provider=str(live.get("provider") or ""),
        ):
            return True
        return bool(getattr(self, "_live_main_credential_adopted", False))

    @staticmethod
    def _oauth_shaped_credential(value: Any) -> bool:
        """True for rotate-and-revoke OAuth shapes (Anthropic oat/JWT/cc-), not Console static keys."""
        if not isinstance(value, str) or not value:
            return False
        from agent.anthropic_credentials import _is_oauth_token
        return _is_oauth_token(value)

    def _apply_summary_route(self, call_kwargs: dict, pinned: Optional[dict[str, Any]] = None) -> None:
        """Pin the summary route onto ``call_kwargs`` for both compression entry points.

        After a fallback, an omitted route is NOT "use the main model": ``call_llm`` re-resolves
        ``auxiliary.compression`` from config — the model that just failed — and a second fallback
        is refused, so every later attempt aborts (#123362). Name the main runtime explicitly.
        A stall-fallback ``pinned`` route replaces the whole route: merged over the main runtime, the
        main api_key/base_url would ride into the fallback entry's call (to its host, or instead of it).
        """
        if pinned:
            # Clear first: a keyless pin (local server) resolves its own credential, never the main one.
            for key in ("provider", "model", "base_url", "api_key", "api_mode"):
                call_kwargs.pop(key, None)
            call_kwargs.update(pinned)
            return
        if self.summary_model:
            call_kwargs["model"] = self.summary_model
            return
        if not getattr(self, "_summary_model_fallen_back", False):
            return
        adopted = self._adopt_live_main_api_key()
        pin_key = self.api_key
        # No live session key: do not stamp a revocable snapshot as explicit api_key. call_llm would
        # skip pool/live resolution and 401 for the rest of the latch (#138469). Static keys stay pinned.
        if pin_key and not adopted and self._oauth_shaped_credential(pin_key):
            pin_key = None
        for key, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("base_url", self.base_url),
            ("api_key", pin_key),
            ("api_mode", getattr(self, "api_mode", "")),
        ):
            if value:
                call_kwargs[key] = value
            elif key == "api_key":
                call_kwargs.pop("api_key", None)
        runtime = call_kwargs.get("main_runtime")
        if not isinstance(runtime, dict) and pin_key is None:
            # Seed an empty runtime so a later setdefault cannot re-inject the omitted snapshot.
            runtime = {}
            call_kwargs["main_runtime"] = runtime
        if isinstance(runtime, dict):
            if pin_key:
                runtime["api_key"] = pin_key
            else:
                runtime.pop("api_key", None)

    def _summarize_window(
        self, messages: list[dict[str, Any]], turns_to_summarize: list[dict[str, Any]], scan: _HandoffScan,
        focus_topic: Optional[str], memory_context: str, bypass_cooldown: bool,
    ) -> Optional[str]:
        """Run the summary LLM; a cancellation rolls back the handoff scan's self-heal mutation first.
        A deterministic pin (repeated stall, #112420) skips the LLM: ``None`` lets Phase 3 insert the static
        fallback summary, or abort under ``abort_on_summary_failure`` exactly like a failed summary call."""
        from agent.context_compressor import take_deterministic_summary_pin
        if take_deterministic_summary_pin():
            # A detached stale attempt must not stamp error state or mutate the shared telemetry dict
            # the fallback owns; unwind as a cancellation before any write lands.
            from agent.conversation_compression import _raise_if_stale_attempt

            _raise_if_stale_attempt(self)
            # Surfaces through the fallback summary's reason line and the host's one-shot user warning.
            self._last_summary_error = (
                "summary model stalled on every route; deterministic fallback summary inserted"
            )
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["failure_class"] = "stall_deterministic_fallback"
            return None
        # Focus-topic derivation scans user turns; only pay when a summary is generated.
        summary_kwargs: dict[str, Any] = {
            "focus_topic": focus_topic or self._derive_auto_focus_topic(messages),
            "memory_context": memory_context,
        }
        if _accepts_keyword_argument(self._generate_summary, "bypass_cooldown"):
            summary_kwargs["bypass_cooldown"] = bypass_cooldown
        try:
            return self._generate_summary(turns_to_summarize, **summary_kwargs)
        except AuxiliaryExplicitCancellation:
            # Cancellation is a true no-op: restore the scan's mutation before the exception escapes.
            # Guard by THIS attempt's ownership: a detached stale attempt (a fallback already claimed
            # and is working the compressor) must not revert summary state the newer attempt advanced.
            # The caller's own generation rides a ContextVar because shared compressor attributes can
            # only name the current owner, never the caller's attempt. Working-attempt comparison (not
            # the entry generation) so a no-op claim does not suppress the owning attempt's rollback.
            from agent.conversation_compression import _caller_attempt_is_current

            if _caller_attempt_is_current(self):
                self._previous_summary = scan.previous_summary_before
                self._summary_has_user_turn = scan.has_user_turn_before
            raise
