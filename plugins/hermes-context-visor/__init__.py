"""Hermes Context Visor plugin — safe fixed /visor slash launcher.

Registers an in-session slash command that launches the graphical
``hermes-context-visor`` browser cockpit with a fixed argv list.

Safety:
  * No shell=True
  * No user-input interpolation into a shell
  * No broker / Docker / Hindsight / LCM mutation
  * On failure, returns platform paste instructions
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _current_profile_session_id() -> str:
    """Best-effort current session id from active profile state.db (read-only)."""
    try:
        import os
        import sqlite3

        profile = (
            os.environ.get("HERMES_VISOR_PROFILE")
            or os.environ.get("HERMES_PROFILE")
            or "default"
        )
        # Prefer profile home when HERMES_HOME already points at a profile dir.
        hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
        candidates = [
            hermes_home / "state.db",
            hermes_home / "profiles" / profile / "state.db",
            Path.home() / ".hermes" / "profiles" / profile / "state.db",
        ]
        state_db = next((p for p in candidates if p.exists()), None)
        if state_db is None:
            return ""
        conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return str(row[0] or "") if row else ""
        finally:
            conn.close()
    except Exception:
        return ""


def _resolve_loaded_lcm_engine():
    """Prefer the active session-bound LCM clone over the unbound singleton.

    ``get_registered_lcm_engine()`` returns the process-wide plugin singleton,
    which often has empty ``current_session_id`` / ``current_conversation_id``
    outside an active turn. Writing that unbound engine into the cockpit
    snapshot blanks Live Status / Fresh Tail after ``/visor``.
    """
    try:
        from hermes_cli.plugins import get_plugin_manager
    except Exception:
        return None
    try:
        manager = get_plugin_manager()
        loaded = manager._plugins.get("hermes-lcm")
        module = getattr(loaded, "module", None) if loaded else None
        if module is None:
            return None

        # Prefer the session-bound active clone when the LCM registry exposes it.
        resolve_active = getattr(module, "resolve_active_lcm_engine", None)
        if not callable(resolve_active):
            engine_mod = getattr(module, "engine", None)
            resolve_active = getattr(engine_mod, "resolve_active_lcm_engine", None)
        if not callable(resolve_active):
            try:
                from hermes_lcm.engine_registry import (  # type: ignore
                    resolve_active_lcm_engine as resolve_active,
                )
            except Exception:
                resolve_active = None
        if callable(resolve_active):
            try:
                session_id = _current_profile_session_id()
                active = (
                    resolve_active(session_id=session_id, conversation_id=session_id)
                    if session_id
                    else None
                )
                if active is not None and (
                    getattr(active, "current_session_id", None)
                    or getattr(active, "current_conversation_id", None)
                ):
                    return active
            except Exception:
                logger.debug(
                    "hermes-context-visor: active LCM resolve failed",
                    exc_info=True,
                )

        getter = getattr(module, "get_registered_lcm_engine", None)
        return getter() if callable(getter) else None
    except Exception:
        logger.debug("hermes-context-visor: failed to resolve loaded LCM plugin", exc_info=True)
        return None


def _write_live_lcm_snapshot_if_available() -> None:
    _ensure_cockpit_importable()
    try:
        from context_cockpit.live_lcm import write_live_lcm_snapshot_for_engine
    except Exception:
        return
    try:
        engine = _resolve_loaded_lcm_engine()
        if engine is None:
            return
        hermes_home = Path(str(engine.get_runtime_identity().get("hermes_home") or "")).expanduser()
        if not hermes_home:
            return
        # Never force: unbound singleton must not clobber a bound per-turn snapshot.
        write_live_lcm_snapshot_for_engine(engine, hermes_home, force=False)
    except Exception:
        logger.debug("hermes-context-visor: live LCM snapshot refresh unavailable", exc_info=True)

# Resolve the cockpit package shipped inside this plugin directory.
def _ensure_cockpit_importable() -> None:
    plugin_dir = Path(__file__).resolve().parent
    if (plugin_dir / "context_cockpit").is_dir() and str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))


def handle_visor_command(raw_args: str) -> str:
    """Slash handler: ``fn(raw_args: str) -> str``.

    Ignores arbitrary args except a tiny allowlist (``force``, ``help``).
    Never passes user text into a shell.
    """
    _ensure_cockpit_importable()
    try:
        from context_cockpit.launcher import (
            launch_context_visor,
            platform_fallback_instructions,
        )
    except Exception as exc:
        try:
            from context_cockpit.launcher import platform_fallback_instructions as _fb
            extra = _fb()
        except Exception:
            extra = (
            "Install/sync: profiles/personal-ops/scripts/context_cockpit + "
            "hermes-context-visor launcher."
            )
        return f"Context Cockpit launcher unavailable ({exc}).\n{extra}"

    args = (raw_args or "").strip().lower()
    if args in {"help", "-h", "--help"}:
        return (
            "Usage: /visor\n"
            "Opens the read-only Hermes Context Cockpit in your browser.\n"
            "Options: /visor force  (open even if already running)\n"
            "Does not execute compress, broker, or any mutating command."
        )

    force = args in {"force", "--force", "-f"}
    # Reject anything else — no free-form command execution.
    if args and not force:
        return (
            "Usage: /visor [force]\n"
            "Refusing unknown arguments (no shell passthrough)."
        )

    _write_live_lcm_snapshot_if_available()
    result = launch_context_visor(force=force)
    if result.ok:
        # Always return an unmistakable operator-facing line — Desktop renders
        # this as a system message. Silent "already running" felt like a no-op.
        header = f"Context Cockpit: {result.method or 'ok'}"
        body = (result.message or "").strip()
        tip = (
            f"Browser URL: {result.url or 'localhost cockpit'} "
            "Status without UI: hermes-context-visor --json"
        )
        return f"{header}\n{body}\n{tip}" if body else f"{header}\n{tip}"
    return result.message


def register(ctx) -> None:
    """Plugin entry — register /visor only (no tools, no hooks, no writes)."""
    register_command = getattr(ctx, "register_command", None)
    if not callable(register_command):
        logger.info("hermes-context-visor: register_command unavailable on this host")
        return
    register_command(
        "visor",
        handle_visor_command,
        description="Open the read-only graphical Context Cockpit (context / LCM / cost / model)",
    )
    logger.info("hermes-context-visor: registered /visor")
