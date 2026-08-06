"""/memory slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from gateway.platforms.base import MessageEvent
from hermes_cli.config import atomic_config_write

class MemoryCommandsMixin:
    """/memory handlers."""

    async def _handle_memory_command(self, event: MessageEvent) -> str:
        """Handle /memory — review pending memory writes + toggle the approval gate.

        Memory entries are small enough to review inline in a chat bubble, so
        the full pending/approve/reject/approval flow works on every platform.
        Gate changes persist to config.yaml and evict the cached agent so the
        new setting takes effect on the next message.
        """
        from gateway.run import _hermes_home
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        from tools.memory_tool import load_on_disk_store

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []
        session_key = self._session_key_for_source(event.source)
        config_path = _hermes_home / "config.yaml"

        def _set_approval(enabled: bool):
            # Write-back round-trip: raw read is correct (merged defaults must
            # not be persisted back to the user's file).
            from hermes_cli.config import read_user_config_raw
            user_config = read_user_config_raw(config_path)
            user_config.setdefault("memory", {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            # New setting must take effect next message → drop cached agent.
            self._evict_cached_agent(session_key)

        # Apply approved writes against a fresh on-disk store (the gateway has
        # no long-lived agent; the store persists to the same MEMORY/USER.md).
        # load_on_disk_store() honors the user's configured char limits.
        store = load_on_disk_store()

        out = handle_pending_subcommand(
            wa.MEMORY, args, memory_store=store, set_mode_fn=_set_approval,
        )
        if out is None:
            out = ("Unknown /memory subcommand. Use: pending, approve <id>, "
                   "reject <id>, approval <on|off>.")
        return out
