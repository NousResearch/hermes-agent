"""Aux-route attribution for compression summary calls (#72636).

The compressor facade keeps main-model identity on ``self.provider`` /
``self.summary_model`` / ``self.base_url``; the auxiliary identity actually used on the
wire is tracked here so abort diagnostics point at the endpoint that failed. The
user-facing sink that reads this state lives in ``conversation_compression_diagnostics``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)


class AuxRouteAttributionMixin:
    """Track which auxiliary identity a compression summary call used, or would have used."""

    def _init_aux_route_attribution(self) -> None:
        # The auxiliary identity actually used on the wire for the most
        # recent summary call — written by call_llm's route_callback,
        # distinct from the main-model identity on self.provider /
        # self.summary_model / self.base_url. Read by the compression-abort
        # diagnostic (#72636).
        self._last_aux_call_provider: str = ""
        self._last_aux_call_model: str = ""
        self._last_aux_call_base_url: str = ""
        # Effective pre-dispatch identity: task config plus explicit call overrides
        # (stall pin or main fallback). Used only if no physical request was sent.
        self._last_aux_config_provider: str = ""
        self._last_aux_config_model: str = ""
        self._last_aux_config_base_url: str = ""
        # Per-attempt failure classification ("auth" | "quota" | "network" | "other" |
        # None). Unlike _last_summary_auth_failure / _last_summary_network_failure,
        # which are intentionally sticky across compress() calls to preserve
        # the cooldown guard (see compress()), this field is reset at the top
        # of every summary attempt so the abort diagnostic reflects the
        # CURRENT attempt's failure mode, not a stale prior one (#72636).
        self._last_attempt_failure_class: Optional[str] = None

    def _prepare_aux_route_attribution(self, call_kwargs: Dict[str, Any]) -> None:
        """Reset the attempt and capture its effective route before client construction.

        Resolve the same arguments dispatch consumes, AFTER stall pins and main-runtime
        fallback are applied. A missing credential must name that selected route, not
        the primary compression config it replaced. Physical callbacks supersede this
        snapshot once a request reaches the provider.
        """
        self._last_attempt_failure_class = None
        self._last_aux_call_provider = ""
        self._last_aux_call_model = ""
        self._last_aux_call_base_url = ""
        self._last_aux_config_provider = ""
        self._last_aux_config_model = ""
        self._last_aux_config_base_url = ""
        _aux_provider = ""
        _aux_model = call_kwargs.get("model") or ""
        _resolved_base = None
        try:
            from agent.auxiliary_client import _resolve_task_provider_model

            _resolved_provider, _resolved_model, _resolved_base, _, _ = (
                _resolve_task_provider_model(
                    call_kwargs.get("task"),
                    **{key: call_kwargs.get(key) for key in ("provider", "model", "base_url", "api_key")},
                )
            )
            _aux_provider = _resolved_provider or ""
            _aux_model = _resolved_model or _aux_model or self.model or ""
        except Exception:
            # Best-effort pre-resolution only: the diagnostic falls back to the
            # static identity fields when config resolution itself fails.
            logger.debug("compression aux identity pre-resolution failed", exc_info=True)
        # Auto remains unresolved until client selection. Explicit routes can be
        # named even when missing credentials prevent constructing their client.
        if _aux_provider not in ("", "auto", None):
            _cfg_base = str(_resolved_base or "")
            if _cfg_base:
                try:
                    from agent.auxiliary_client import _extract_url_query_params
                    _cfg_base, _ = _extract_url_query_params(_cfg_base)
                except (ImportError, ValueError):
                    _cfg_base = _cfg_base.split("?", 1)[0]
            self._last_aux_config_provider = str(_aux_provider)
            self._last_aux_config_model = str(_aux_model or "")
            self._last_aux_config_base_url = _cfg_base

    def _record_aux_route(self, provider, model, base_url) -> None:
        """route_callback target: invoked before every call_llm physical request.

        The final callback is therefore the route that terminated the call, after
        auto-detection / fallback. Strip any query string defensively — some proxies
        carry credentials as ?key=... and this value is surfaced in user-facing
        diagnostics on Telegram/Discord/Slack/CLI (#72636).
        """
        _safe_base = ""
        if base_url:
            try:
                from agent.auxiliary_client import _extract_url_query_params
                _safe_base, _ = _extract_url_query_params(str(base_url))
            except (ImportError, ValueError):
                _safe_base = str(base_url).split("?", 1)[0]
        self._last_aux_call_provider = provider or ""
        self._last_aux_call_model = model or ""
        self._last_aux_call_base_url = _safe_base or ""

    def _aux_route_label(self, route: Dict[str, str]) -> str:
        """Route label for summary diagnostics.

        Keep the same wire → effective config → static precedence as failure identity.
        Without a wire snapshot, a configured auxiliary model must not be replaced
        by the main model merely because ``summary_model`` is empty (#72636).
        """
        provider = (route.get("provider") or self._last_aux_call_provider
                    or self._last_aux_config_provider or self.provider or "auto")
        model = (route.get("model") or self._last_aux_call_model
                 or self._last_aux_config_model or self.summary_model or self.model)
        return f"(provider={provider} model={model})"

    def _summary_failure_identity(self) -> Tuple[str, str, str]:
        """(provider, model, base_url) identifying the failed aux summary call.

        ``summary_model`` is permanently empty in production, so the static fields alone
        would log the MAIN identity ("(main)" / the main provider) for an auxiliary
        failure. Name the route the aux lane actually used — the wire identity first,
        the config-layer identity when the call died pre-dispatch — with the static
        fields as the final fallback for an unset auxiliary.compression
        (#72636 review / #113582).
        """
        return (
            self._last_aux_call_provider or self._last_aux_config_provider or self.provider or "auto",
            self._last_aux_call_model or self._last_aux_config_model or self.summary_model or "(main)",
            self._last_aux_call_base_url or self._last_aux_config_base_url or self.base_url or "default",
        )

    def _record_summary_access_failure(self, error: Exception) -> None:
        """Keep the legacy preservation flag while distinguishing billing from credentials."""
        from agent.context_compressor import _exc_status_code, _SUMMARY_PERMANENT_QUOTA_MARKERS

        self._last_summary_auth_failure = True
        quota = _exc_status_code(error) == 402 or any(
            marker in str(error).lower() for marker in _SUMMARY_PERMANENT_QUOTA_MARKERS
        )
        self._last_attempt_failure_class = "quota" if quota else "auth"

    def _classify_attempt_failure(self, failure_class: str) -> None:
        """Record this attempt's failure class unless a more specific one is already set.

        The generic branch (timeout, JSON decode, 5xx, ... — anything that is not auth
        and not a network stream close) is classified "other" so the abort diagnostic
        does not inherit a stale "auth" verdict from a prior attempt (#72636).
        """
        if self._last_attempt_failure_class is None:
            self._last_attempt_failure_class = failure_class

    def _record_aux_compression_call(
        self, *, prompt_messages: List[Dict[str, Any]], max_tokens: int | None, duration_ms: int,
        aux_provider: str | None = None, aux_model: str | None = None,
        effective_aux_context: int | None = None, phase_timings: Dict[str, Any] | None = None,
    ) -> None:
        from agent.context_compressor import _safe_int

        telemetry = getattr(self, "_active_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            return
        telemetry["aux_prompt_tokens"] = estimate_messages_tokens_rough(prompt_messages)
        telemetry["aux_output_reservation"] = _safe_int(max_tokens)
        if aux_provider:
            telemetry["aux_provider"] = aux_provider
        if aux_model:
            telemetry["aux_model"] = aux_model
        if effective_aux_context is not None:
            telemetry["effective_aux_context"] = _safe_int(effective_aux_context)
        if telemetry["effective_aux_context"] is not None and telemetry["aux_prompt_tokens"] is not None:
            telemetry["fit_margin"] = (telemetry["effective_aux_context"] - telemetry["aux_prompt_tokens"]
                                       - (telemetry["aux_output_reservation"] or 0))
        telemetry["aux_call_duration_ms"] = (telemetry.get("aux_call_duration_ms") or 0) + max(0, int(duration_ms))
        for key in ("queue_wait_ms", "prompt_build_ms", "time_to_first_progress_ms", "summary_generation_ms", "commit_ms"):
            if not isinstance(phase_timings, dict) or key not in phase_timings:
                continue
            value = _safe_int(phase_timings[key])
            # Wait and generation phases accumulate across retries; the rest are point readings.
            accumulate = key in {"queue_wait_ms", "summary_generation_ms"} and value is not None
            telemetry[key] = (telemetry.get(key) or 0) + value if accumulate else value
