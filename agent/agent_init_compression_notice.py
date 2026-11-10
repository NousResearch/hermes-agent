"""Init-time compression banner and the once-per-profile Codex gpt-5.x autoraise notice."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Optional

from hermes_constants import get_hermes_home


def _build_codex_gpt5_autoraise_notice(
    autoraise: dict[str, Any], context_length: Optional[int] = None
) -> str:
    """One-time notice when Codex gpt-5.x raises compaction (``autoraise``: model/from/to).

    ``context_length`` is the live-resolved window (Codex's catalog shifts server-side) so the
    banner reports what this session got. Printed for CLI and replayed via status_callback for
    gateway users, so it must be self-contained and include the exact opt-back-out command.
    """
    model = str(autoraise.get("model") or "gpt-5.4/5.5").strip().lower().rsplit("/", 1)[-1]
    if isinstance(context_length, int) and context_length > 0:
        cap = f"{round(context_length / 1000)}K"
    else:
        # Static fallback: codex-spark is natively 128K; gpt-5.4/5.5/5.6 are capped at 272K.
        cap = "128K" if model.startswith("gpt-5.3-codex-spark") else "272K"
    from_pct = round(autoraise["from"] * 100)
    to_pct = round(autoraise["to"] * 100)
    return (
        f"ℹ Codex {model} caps context at {cap}, so auto-compaction was raised "
        f"to {to_pct}% (from {from_pct}%) to use more of the window before "
        f"summarizing.\n"
        f"  Opt back out: hermes config set compression.codex_gpt55_autoraise false"
    )


def _codex_gpt55_autoraise_notice_marker():
    """Per-profile marker path (``$HERMES_HOME`` is profile-scoped; not a config key)."""
    return get_hermes_home() / ".codex_gpt55_autoraise_notice"


def _codex_gpt55_autoraise_notice_state(autoraise: dict[str, Any]) -> str:
    """Notice identity keyed on what it displays (model + from→to percentages).

    An unchanged threshold stays silent across restarts; a changed global threshold or a
    different autoraised Codex model re-notifies once.
    """
    model = str(autoraise.get("model") or "").strip().lower().rsplit("/", 1)[-1]
    from_pct = round(float(autoraise["from"]) * 100)
    to_pct = round(float(autoraise["to"]) * 100)
    return f"{model}:{from_pct}:{to_pct}"


def _codex_gpt55_autoraise_notice_seen(autoraise: dict[str, Any]) -> bool:
    """True if this exact notice was already shown for this profile (unreadable = unseen)."""
    try:
        current = _codex_gpt55_autoraise_notice_state(autoraise)
        return _codex_gpt55_autoraise_notice_marker().read_text(
            encoding="utf-8"
        ).strip() == current
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _record_codex_gpt55_autoraise_notice(autoraise: dict[str, Any]) -> None:
    """Persist that the notice was shown. Best-effort: a failure only re-shows it later."""
    with suppress(OSError, KeyError, TypeError, ValueError):
        marker = _codex_gpt55_autoraise_notice_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_codex_gpt55_autoraise_notice_state(autoraise), encoding="utf-8")


def _emit_compression_summary(agent, cs):
    # Codex autoraise notice: once per profile/config state (persisted marker; the gateway
    # rebuilds the agent per message). The display gate hides the banner, not the autoraise.
    _autoraise = agent._compression_threshold_autoraised or {}
    _autoraise_notice = None
    if (
        # A change in the raised threshold (or the autoraised model) updates the marker state and
        # re-notifies once. The config display gate (compression.codex_gpt55_autoraise_notice) still
        # suppresses the banner entirely without disabling the threshold autoraise. See #54432.
        bool(_autoraise)
        and cs.enabled
        and cs.autoraise_notice_enabled
        and not _codex_gpt55_autoraise_notice_seen(_autoraise)
    ):
        _autoraise_notice = _build_codex_gpt5_autoraise_notice(
            _autoraise, context_length=getattr(agent.context_compressor, "context_length", None)
        )

    if not agent.quiet_mode:
        _cc = agent.context_compressor
        if cs.enabled:
            # The active engine's own threshold — a plugin's differs from cs.threshold.
            _pct = getattr(_cc, "threshold_percent", cs.threshold)
            _cap = getattr(_cc, "threshold_tokens_cap", None)
            # Name the cap only when it is what set the trigger; on small windows the ratio already sits below it.
            _eff_cap = getattr(_cc, "_effective_threshold_cap", lambda _ctx: None)(_cc.context_length)
            _cap_binds = _eff_cap is not None and _cc.threshold_tokens == _eff_cap
            _cap_note = f" (capped at {_cap:,} tokens)" if _cap_binds else ""
            print(f"📊 Context limit: {_cc.context_length:,} tokens (compress at {int(_pct*100)}% = {_cc.threshold_tokens:,}{_cap_note})")
        else:
            print(f"📊 Context limit: {_cc.context_length:,} tokens (auto-compression disabled)")
        # Gateway users get the same text via _compression_warning on turn 1.
        if _autoraise_notice:
            agent._safe_print(_autoraise_notice, diagnostic=True)

    # status_callback isn't wired yet: stash for replay on the first turn; mark shown so
    # repeated inits stay silent.
    agent._compression_warning = _autoraise_notice
    if _autoraise_notice:
        _record_codex_gpt55_autoraise_notice(_autoraise)
    # Feasibility check deferred to the first turn near threshold (eager costs ~400ms cold).
    agent._compression_feasibility_checked = False
