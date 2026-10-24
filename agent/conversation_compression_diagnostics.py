"""User-facing diagnostics for context compression outcomes.

Moved out of ``agent/conversation_compression.py`` (facade): the terminal compaction-done
status edge and the compression auth/abort attribution hint. The auth hint reads the
wire/config identity state tracked by ``agent/context_compressor_attribution``.
"""

from __future__ import annotations

from typing import Any


def _emit_compaction_done(agent: Any) -> None:
    """Emit the structured terminal edge for a started compaction."""
    from agent.conversation_compression import COMPACTION_DONE_STATUS, _swallow

    status_callback = getattr(agent, "status_callback", None)
    if not status_callback:
        return
    with _swallow('status_callback error in compaction completion', exc_info=True):
        status_callback("compacted", COMPACTION_DONE_STATUS)


def _emit_compression_auth_hint(agent: Any) -> None:
    """Render the final attempt's route/class after compression aborts.

    The current class is independent of sticky conversation-preservation flags.
    Wording must pass the gateway noise filter so messaging users see it too.
    """
    from agent.context_compressor_attribution import select_summary_route

    _ctx_comp = getattr(agent, "context_compressor", None)
    if _ctx_comp is None:
        return
    _cls = getattr(_ctx_comp, "_last_attempt_failure_class", None)
    if _cls not in ("auth", "quota", "network", "other"):
        return

    route = select_summary_route(_ctx_comp)
    _aux_provider, _aux_model, _aux_base = route.provider, route.model, route.base_url
    _pre_dispatch = route.configured and not route.dispatched
    if _pre_dispatch:
        _note = " (no request was dispatched)"
    elif not route.configured:
        _note = " (auxiliary.compression is not configured — using main model)"
    else:
        _note = ""

    # Per-class guidance. Wording avoids the gateway noise-filter patterns
    # (notably "auxiliary ... failed" and "compression summary failed") so
    # the message reaches Telegram/Discord/Slack, not just local/CLI.
    if _cls == "auth":
        if _pre_dispatch:
            _guidance = "credential missing — set this provider's API key"
        else:
            _guidance = (
                "auth/permission error — check the credential and "
                "auxiliary.compression in config.yaml"
            )
    elif _cls == "quota":
        _guidance = "billing/quota exhausted — check this provider's balance and usage limits"
    elif _cls == "network":
        _guidance = "network/connection error — this is usually transient"
    else:
        _guidance = "error — see agent.log for detail"

    _headline = (
        "⚠ Compression auxiliary provider could not start its request "
        if _pre_dispatch
        else {
            "network": "⚠ Compression auxiliary endpoint could not be reached ",
            "auth": "⚠ Compression auxiliary endpoint rejected the request ",
            "quota": "⚠ Compression auxiliary endpoint rejected the request ",
            "other": "⚠ Compression auxiliary endpoint returned an unusable response or request error ",
        }[_cls]
    )
    # Sanitize at the SINK, regardless of which producer filled _aux_base.
    # The route_callback and config-layer captures strip the query at their
    # sources, but the no-route/no-explicit-aux fallback above reads
    # compressor.base_url (the MAIN model's URL) raw — and some proxies carry
    # credentials as ?key=... (#72636 review, defect 2). A user-facing
    # diagnostic must never be one forgetful producer away from leaking one.
    # Fragments are not sent to the provider and may themselves contain ?key=...
    _aux_base = _aux_base.split("#", 1)[0]
    try:
        from agent.auxiliary_client import _extract_url_query_params
        _aux_base, _ = _extract_url_query_params(_aux_base)
    except (ImportError, ValueError):
        _aux_base = str(_aux_base).split("?", 1)[0]
    agent._emit_warning(
        f"{_headline}"
        f"({_guidance}). "
        f"🔌 Provider: {_aux_provider}  Model: {_aux_model}  "
        f"🌐 Endpoint: {_aux_base}{_note}"
    )
