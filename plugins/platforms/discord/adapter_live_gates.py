"""Discord user/role allowlists that follow config.yaml without a reconnect.

``connect()`` copies the gate into ``_allowed_user_ids`` / ``_allowed_role_ids`` inside the owning
profile's scope (#72348). Every intake path (messages, DMs, slash commands, voice, component clicks)
calls :meth:`DiscordLiveGatesMixin._refresh_live_gates` first, which re-derives that copy when the
profile's config.yaml changed since. What stays connect-time: the Server Members intent, and the
resolution of username entries to IDs (a username added later matches only after a reconnect).
"""
from __future__ import annotations

from typing import Any, Callable

from gateway.platforms._shared import bridge_owns_env, mark_live_gate_baseline, refresh_live_gate_extra

# The ``extra`` keys the gate reads (``DiscordAdapter._gate_raw``): refreshed together so env and
# extra never disagree about which config they came from.
_LIVE_GATE_EXTRA_KEYS = (
    "allow_from", "allowed_users", "allowed_roles", "allowed_channels", "ignored_channels",
    "no_thread_channels", "free_response_channels", "allow_all_users", "allow_bots",
)


class DiscordLiveGatesMixin:
    _gate_raw: Callable[..., Any]
    _gate_csv_set: Callable[[Any], set]
    _snapshot_gate_env: Callable[[], None]

    def _mark_live_gate_baseline(self) -> None:
        mark_live_gate_baseline(self)

    def _derive_connect_gates(self) -> None:
        """connect()'s gate, inside the owning profile's scope. A fresh connection resolves usernames
        from the configured names again (an account can rename, another can take its name), so the
        username->ID map starts empty; only a repeated on_ready on one connection keeps it."""
        from plugins.platforms.discord.adapter import _multiplex_active
        self._snapshot_gate_env()
        # Single profile: a prior connection's resolution rewrote the bridged env to IDs; resolve the
        # configured names (YAML ``allow_from``) again instead, or a later save drops their mapping.
        configured = (getattr(self.config, "extra", None) or {}).get("allow_from")
        if configured and not _multiplex_active() and bridge_owns_env("DISCORD_ALLOWED_USERS"):
            self._gate_env_snapshot["DISCORD_ALLOWED_USERS"] = configured
        self._username_to_id = {}
        self._allowed_user_ids = self._get_allowed_users()
        self._allowed_role_ids = self._get_allowed_roles()  # ANY listed role grants access
        self._mark_live_gate_baseline()

    def _get_allowed_users(self) -> set:
        """This adapter's DISCORD_ALLOWED_USERS entries (per-profile, cleaned); a username this adapter
        resolved at connect reads as its ID, so a refresh keeps it working and its removal revokes it."""
        from plugins.platforms.discord.adapter import _clean_discord_id
        raw = self._gate_raw("allow_from", "DISCORD_ALLOWED_USERS")
        if raw is None:
            extra = getattr(getattr(self, "config", None), "extra", None)
            if isinstance(extra, dict):
                raw = extra.get("allowed_users")
        resolved = getattr(self, "_username_to_id", None) or {}
        entries = {_clean_discord_id(str(entry)) for entry in self._gate_csv_set(raw)}
        return {resolved.get(entry.lower(), entry) for entry in entries if entry}

    def _get_allowed_roles(self) -> set:
        """This adapter's DISCORD_ALLOWED_ROLES role IDs (per-profile)."""
        raw = self._gate_raw("allowed_roles", "DISCORD_ALLOWED_ROLES")
        return {
            int(str(entry).strip()) for entry in self._gate_csv_set(raw)
            if str(entry).strip().isdigit()
        }

    def _refresh_live_gates(self) -> None:
        """Re-derive the gate when the owning profile's config.yaml changed (a stat otherwise)."""
        refresh_live_gate_extra(self, _LIVE_GATE_EXTRA_KEYS, on_reload=self._rederive_live_gates)

    def _rederive_live_gates(self) -> None:
        """Runs inside the owning profile's scope, so the env snapshot is the owner's."""
        self._snapshot_gate_env()
        self._allowed_user_ids = self._get_allowed_users()
        self._allowed_role_ids = self._get_allowed_roles()
        resolved = set((getattr(self, "_username_to_id", None) or {}).values())
        self._username_resolved_ids = resolved & self._allowed_user_ids

    def _current_gate_sets(self) -> tuple[set, set]:
        """``(user_ids, role_ids)`` as of now, for component views built before an edit."""
        self._refresh_live_gates()
        return self._allowed_user_ids, self._allowed_role_ids
