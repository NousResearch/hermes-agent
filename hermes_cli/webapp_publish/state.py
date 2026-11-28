"""Record of what the Desktop setup flow installed, so switching modes cleans up.

The flow writes ``webapp/desktop-setup.json`` under the gateway home listing the
artifacts it created (Caddy site blocks verbatim, the avahi conf backup, the
published mDNS name). Reconfiguring into the OTHER mode removes exactly those
artifacts — never blocks the user authored or edited themselves (removal is an
exact-text match) — and offers to restore the machine's previous mDNS identity.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STATE_REL = Path("webapp") / "desktop-setup.json"


def load_state(home: Path) -> dict:
    try:
        return json.loads((home / STATE_REL).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def save_state(home: Path, state: dict) -> None:
    path = home / STATE_REL
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def forget_state(home: Path) -> None:
    try:
        (home / STATE_REL).unlink()
    except OSError:
        pass


def cleanup_previous_mode(home: Path, new_mode: str, *,
                          confirm: Optional[Callable[[str, bool], bool]] = None) -> list[str]:
    """Remove the previous mode's recorded artifacts when switching modes.
    Returns human-readable action lines. ``confirm(question, default)`` gates
    the mDNS identity restore; None skips that restore. Never raises for
    permission problems — it reports what it could not do."""
    from . import caddy, mdns

    state = load_state(home)
    old_mode = state.get("mode")
    if not old_mode or old_mode == new_mode:
        return []
    actions: list[str] = []

    blocks = [b for b in state.get("caddy_blocks", []) if isinstance(b, str) and b.strip()]
    caddyfile = caddy.DEFAULT_CADDYFILE
    if blocks and caddyfile.exists():
        try:
            text = caddyfile.read_text(encoding="utf-8-sig")
            remaining = text
            removed = []
            for block in blocks:
                if block in remaining:
                    remaining = remaining.replace(block, "")
                    removed.append(block.splitlines()[0].strip())
            if removed:
                backup = caddyfile.with_name(caddyfile.name + ".bak-mode-switch")
                shutil.copy2(caddyfile, backup)
                caddyfile.write_text(remaining.rstrip("\n") + "\n", encoding="utf-8")
                if caddy.validate(caddyfile):
                    caddy.reload_caddy()
                    actions.append("Removed previous mode's Caddy site block(s): " + ", ".join(removed))
                else:
                    caddy.restore_backup(caddyfile, backup)
                    actions.append("Caddy cleanup skipped (validation failed; config restored).")
        except OSError:
            actions.append(f"Could not clean {caddyfile} (permissions) — remove the old site block manually.")

    if old_mode == "local" and state.get("avahi_backup") and confirm is not None:
        backup = Path(state["avahi_backup"])
        name = state.get("mdns_name") or "the previous mDNS name"
        if backup.exists() and confirm(
                f"  Restore this machine's previous mDNS identity (undo {name})?", True):
            try:
                shutil.copy2(backup, mdns.DEFAULT_AVAHI_CONF)
                mdns.restart_avahi()
                actions.append("Restored the previous avahi configuration (mDNS identity).")
            except OSError:
                actions.append(f"Could not restore {mdns.DEFAULT_AVAHI_CONF} (permissions).")

    forget_state(home)
    return actions