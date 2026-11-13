"""Checkpoint-enablement resolution for TUI-created agents (issue #79625).

Desktop sessions (``hermes serve`` → TUI gateway) never set ``HERMES_TUI_CHECKPOINTS``
(that env var is only set by the ``hermes --tui --checkpoints`` CLI path), so agent
creation previously gated filesystem checkpointing on the env var alone and silently
ignored ``checkpoints.enabled`` in config.yaml. These helpers resolve enablement the
way the messaging gateway does (``hermes_agent/gateway/run.py`` ``_checkpoint_agent_kwargs``):
the env var (explicit CLI opt-in) wins when set, otherwise the ``checkpoints``
config section applies with ``DEFAULT_CONFIG`` defaults.
"""

import os

from utils import is_truthy_value


def _load_checkpoints_enabled(cfg: dict | None = None) -> bool:
    """Whether filesystem checkpoints are enabled for TUI-created agents.

    Keeps the legacy ``checkpoints: true`` bool form working and falls back to the
    ``DEFAULT_CONFIG`` default when the section is absent or malformed.
    """
    cp_cfg = (cfg or {}).get("checkpoints", {})
    if isinstance(cp_cfg, bool):
        cp_cfg = {"enabled": cp_cfg}
    elif not isinstance(cp_cfg, dict):
        cp_cfg = {}
    from hermes_agent.hermes_cli.config import DEFAULT_CONFIG
    return bool(cp_cfg.get("enabled", DEFAULT_CONFIG["checkpoints"]["enabled"]))


def resolve_checkpoints_enabled(cfg: dict | None = None) -> bool:
    """Resolve checkpoint enablement with the env-var override first.

    ``HERMES_TUI_CHECKPOINTS`` (set by ``hermes --tui --checkpoints``) is the
    explicit opt-in and wins when present; otherwise fall back to the
    ``checkpoints`` config section (issue #79625 — the desktop backend never
    sets the env var, so config was being silently ignored).
    """
    env_val = os.environ.get("HERMES_TUI_CHECKPOINTS")
    if env_val is not None:
        return is_truthy_value(env_val)
    return _load_checkpoints_enabled(cfg)


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    """``/rollback`` number or hash → the checkpoint's hash (``hermes_agent/tools/checkpoint_refs.py``, every surface's rule)."""
    from hermes_agent.tools.checkpoint_refs import resolve_checkpoint_ref
    return resolve_checkpoint_ref(mgr.list_checkpoints(cwd), ref)



def undo_rollback_hint_for(session: dict, target_message: dict, kept_history: list) -> dict | None:
    """``/undo``'s changed-files line for a sidecar session (``hermes_agent/tools/checkpoint_undo_hint.py``): the same
    manager, cwd and container refusal its ``rollback.*`` methods use. None: no line.

    The store is the SESSION's profile's (``_resolve_checkpoint_base`` reads the bound HERMES_HOME), so the
    whole read runs under that profile's runtime scope: an RPC thread otherwise holds the launch profile's."""
    mgr = getattr(session.get("agent"), "_checkpoint_mgr", None)
    if mgr is None or not mgr.enabled:
        return None
    from hermes_agent.tools.checkpoint_undo_hint import undo_hint_payload, undo_span_start
    from hermes_agent.tui_gateway import server  # method bodies are bound onto server.py's globals (method_ctx.py)

    def hint(mgr, cwd):  # under the same profile scope rollback.restore runs in (_with_checkpoints)
        if server._container_checkpoint_refusal(session, mgr, cwd):
            return None
        since = undo_span_start((target_message or {}).get("timestamp"), kept_history)
        return undo_hint_payload(mgr, cwd, since, task_id=None)
    return server._with_checkpoints(session, hint)


def undo_hint_fields(session: dict, rewound, params: dict) -> dict:
    """``session.undo``'s ``undo_hint`` field for a rewind ``(kept, live_view, count)``; ``{}`` for none or /retry."""
    if not rewound or not rewound[2] or params.get("intent") == "retry":
        return {}
    hint = undo_rollback_hint_for(session, rewound[1], rewound[0])
    return {"undo_hint": hint} if hint else {}
