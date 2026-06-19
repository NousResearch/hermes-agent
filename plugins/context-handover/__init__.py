"""context-handover plugin.

Monitors context usage each turn via the ``post_llm_call`` hook. When usage
exceeds *threshold* (default 90 %), it:

1. Writes a handover document to ``notes_dir`` capturing the current task,
   last response, and recent conversation.
2. Resets the session via ``session_store.reset_session(session_key)`` so the
   next turn starts with a clean context window.
3. Reseeds the fresh session with a ``handover: <path>`` message so the agent
   reads the doc and continues seamlessly (requires ``auto_continue=true``).
4. Disables the agent's built-in preflight compaction while active, since a
   handover is strictly better than compaction at high-% usage.

Optionally updates the Discord bot presence to ``"{pct}% · {task}"`` each turn
when ``presence=true`` (default).

Configuration (hermes config.yaml ``plugins.context-handover.*`` or plugin.yaml
``extra.*`` defaults):

  threshold:     0.90   — fraction of context_length to trigger handover
  notes_dir:     ~/notes — directory for handover documents
  auto_continue: true   — reset+reseed after writing doc (false = doc only)
  presence:      true   — update Discord presence each turn

Scope note: the reset+reseed path requires a live GatewayRunner (via
``gateway.run._gateway_runner_ref``). In CLI-only mode, the session reset will
be skipped gracefully (warning logged). A follow-up should wire CLI /new
semantics for non-gateway sessions.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from . import handover as _handover

logger = logging.getLogger(__name__)

# Module-level config, set at register() time.
_config: dict[str, Any] = {}
_config_lock = threading.Lock()


def _get_cfg(key: str, default: Any) -> Any:
    with _config_lock:
        return _config.get(key, default)


def _load_config(ctx_or_none: Any = None) -> dict[str, Any]:
    """Load plugin config from hermes config.yaml, falling back to defaults."""
    defaults = {
        "threshold": 0.90,
        "notes_dir": "~/notes",
        "auto_continue": True,
        "presence": True,
    }
    try:
        from hermes_cli.config import cfg_get, load_config  # type: ignore
        cfg = load_config()
        section = cfg_get(cfg, "plugins", "context-handover", default={}) or {}
        merged = {**defaults, **section}
    except Exception:
        merged = dict(defaults)
    return merged


def _on_post_llm_call(
    agent: Any = None,
    session_id: str = "",
    task_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: Optional[list] = None,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    """post_llm_call hook: check context %, update presence, trigger handover."""
    if agent is None:
        return

    pct = _handover.compute_context_pct(agent)
    if pct is None:
        return

    threshold = _get_cfg("threshold", 0.90) * 100  # convert to %
    notes_dir = Path(_get_cfg("notes_dir", "~/notes"))
    auto_continue = _get_cfg("auto_continue", True)
    presence_enabled = _get_cfg("presence", True)

    task = _handover._extract_task(conversation_history or [], user_message)

    # -- Discord presence update (best-effort each turn) ---
    if presence_enabled:
        try:
            from gateway.run import _gateway_runner_ref  # type: ignore
            runner = _gateway_runner_ref()
            if runner is not None:
                _handover.update_discord_presence(runner, pct, task)
        except Exception as exc:
            logger.debug("[context-handover] presence update skipped: %s", exc)

    # -- Handover threshold check ---
    if pct < threshold:
        return

    logger.info(
        "[context-handover] Context at %.1f%% >= %.1f%% threshold — triggering handover",
        pct,
        threshold,
    )

    # Write the handover doc.
    try:
        doc_path = _handover.write_handover_doc(
            notes_dir=notes_dir,
            task=task,
            assistant_response=assistant_response,
            conversation_history=conversation_history or [],
            pct=pct,
            model=model,
        )
    except Exception as exc:
        logger.error("[context-handover] Failed to write handover doc: %s", exc)
        return

    # Attempt session reset + reseed via the gateway runner.
    session_key = getattr(agent, "_gateway_session_key", None)
    if not session_key:
        logger.warning(
            "[context-handover] No _gateway_session_key on agent — "
            "wrote handover doc but cannot reset session. "
            "This plugin requires a gateway session (not CLI-only). "
            "Doc: %s",
            doc_path,
        )
        return

    try:
        from gateway.run import _gateway_runner_ref  # type: ignore
        runner = _gateway_runner_ref()
        if runner is None:
            logger.warning(
                "[context-handover] GatewayRunner not available — "
                "wrote handover doc but cannot reset session. Doc: %s",
                doc_path,
            )
            return

        _handover.trigger_handover(
            agent=agent,
            runner=runner,
            doc_path=doc_path,
            session_key=session_key,
            auto_continue=auto_continue,
        )
    except Exception as exc:
        logger.error("[context-handover] Handover trigger failed: %s", exc)


def register(ctx: Any) -> None:
    global _config
    with _config_lock:
        _config = _load_config(ctx)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    logger.info(
        "[context-handover] Registered. threshold=%.0f%% notes_dir=%s "
        "auto_continue=%s presence=%s",
        _get_cfg("threshold", 0.90) * 100,
        _get_cfg("notes_dir", "~/notes"),
        _get_cfg("auto_continue", True),
        _get_cfg("presence", True),
    )
