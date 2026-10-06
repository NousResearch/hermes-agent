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
    """Surface the compression auxiliary's own provider/model/endpoint when
    compression aborted on the auxiliary summary call (#72636).

    The identity ACTUALLY used on the wire is recorded by
    ``call_llm``'s ``route_callback`` — invoked before every physical wire
    attempt after the real client is built (auto-detection, fallback chains,
    and ``client.base_url`` applied), so the final snapshot reflects the route
    the failed request really took, not the config-layer pre-resolution from
    ``_resolve_task_provider_model`` (which ``call_llm`` may override).
    The callback writes ``_last_aux_call_provider`` /
    ``_last_aux_call_model`` / ``_last_aux_call_base_url`` on the
    compressor, with the base_url query-stripped to avoid leaking
    credentials that some proxies carry as ``?key=...``. This is *not*
    the same as ``compressor.provider`` / ``compressor.summary_model`` /
    ``compressor.base_url`` — those carry the *main* model's identity.

    The surrounding "Compression aborted" warning says *that* something
    failed; this companion block says *which* endpoint to fix.

    Called only from the compression-abort branch of
    :func:`_candidate_rejected`. The helper runs after the current
    compression attempt has completed, avoiding the pre-request and
    unrelated-main-error ordering problems that an earlier call site
    in the main-model API-error path introduced.

    The gate is the per-attempt ``_last_attempt_failure_class`` (reset at
    the top of every ``_generate_summary`` attempt), NOT the sticky
    ``_last_summary_auth_failure`` — that flag intentionally persists
    across compress() calls to protect the cooldown guard, so a 401
    followed by a forced retry that fails with a 500 would otherwise be
    mis-attributed as an auth failure (#72636).

    When the route_callback never fired (e.g. ``call_llm`` raised "no
    provider configured" before a client existed), the identity fields
    stay empty and the diagnostic falls back to the main-model identity
    with an explicit note, rather than inventing a phantom endpoint.

    Wording is chosen to pass the gateway noise filter
    (``_TELEGRAM_NOISY_STATUS_RE``) — it must NOT match
    ``auxiliary\\s+.+\\s+failed`` / ``compression\\s+summary\\s+failed``
    patterns, otherwise messaging platforms (Telegram/Discord/Slack)
    silently swallow the diagnostic and the user never sees it.
    """
    _ctx_comp = getattr(agent, "context_compressor", None)
    if _ctx_comp is None:
        return
    _cls = getattr(_ctx_comp, "_last_attempt_failure_class", None)
    if _cls not in ("auth", "network", "other"):
        return

    _aux_provider = (getattr(_ctx_comp, "_last_aux_call_provider", "") or "").strip()
    _aux_model = (getattr(_ctx_comp, "_last_aux_call_model", "") or "").strip()
    _aux_base = (getattr(_ctx_comp, "_last_aux_call_base_url", "") or "").strip()

    _cfg_provider = (getattr(_ctx_comp, "_last_aux_config_provider", "") or "").strip()
    _cfg_model = (getattr(_ctx_comp, "_last_aux_config_model", "") or "").strip()
    _cfg_base = (getattr(_ctx_comp, "_last_aux_config_base_url", "") or "").strip()

    # _last_aux_call_* is written by call_llm's route_callback after the real
    # client is built (auto-detection / fallback applied). When the callback
    # never fired, no physical request was dispatched. Two distinct shapes
    # must not be conflated (#72636):
    #
    # 1. auxiliary.compression IS configured with an explicit provider, but
    #    the call died before dispatch — e.g. that provider's API key is
    #    missing ("Provider 'X' is set in config.yaml but no API key was
    #    found"). Report the CONFIGURED identity as a pre-dispatch failure;
    #    never substitute the healthy main endpoint here.
    # 2. auxiliary.compression is unset, so the summary call would have used
    #    the main model. Fall back to the main-model identity and say so,
    #    rather than inventing a phantom endpoint.
    _pre_dispatch = False
    if not _aux_provider and not _aux_model and not _aux_base:
        if _cfg_provider or _cfg_model or _cfg_base:
            _aux_provider = _cfg_provider or "unknown"
            _aux_model = _cfg_model or "unknown"
            _aux_base = _cfg_base or "unknown"
            _pre_dispatch = True
            _note = " (no request was dispatched)"
        else:
            _aux_provider = (getattr(_ctx_comp, "provider", "") or "auto").strip() or "auto"
            _aux_model = (getattr(_ctx_comp, "model", "") or "unknown").strip() or "unknown"
            _aux_base = (getattr(_ctx_comp, "base_url", "") or "unknown").strip() or "unknown"
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
    elif _cls == "network":
        _guidance = "network/connection error — this is usually transient"
    else:
        _guidance = "error — see agent.log for detail"

    _headline = (
        "⚠ Compression auxiliary provider could not start its request "
        if _pre_dispatch
        else "⚠ Compression auxiliary endpoint could not be reached "
    )
    # Sanitize at the SINK, regardless of which producer filled _aux_base.
    # The route_callback and config-layer captures strip the query at their
    # sources, but the no-route/no-explicit-aux fallback above reads
    # compressor.base_url (the MAIN model's URL) raw — and some proxies carry
    # credentials as ?key=... (#72636 review, defect 2). A user-facing
    # diagnostic must never be one forgetful producer away from leaking one.
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
