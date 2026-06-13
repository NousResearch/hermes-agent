"""Tests for `/reload-skills` resyncing the Discord ``/skill`` autocomplete.

Before this change, ``_register_skill_group`` captured the skill catalog
in closure variables (``entries`` and ``skill_lookup``) so that the one
``tree.add_command`` call at startup owned the only live copy of the
skill list. The closure is never re-entered after startup, so
``/reload-skills`` (which rescans the on-disk skill dir and refreshes
the in-process registry) had no way to propagate its results into the
autocomplete — new skills stayed invisible in the dropdown and deleted
skills returned an "Unknown skill" error when the stale autocomplete
entry was clicked.

The fix promotes those two variables to instance attributes
(``_skill_entries`` / ``_skill_lookup``) and exposes a
``refresh_skill_group()`` method that rescans and mutates them in
place. The gateway ``_handle_reload_skills_command`` iterates its
connected adapters and calls the method on any that expose it.

No ``tree.sync()`` is required because Discord fetches autocomplete
options dynamically on every keystroke — we only need to rebind the
data the live callbacks already read from.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _make_adapter():
    """Construct a DiscordAdapter without going through __init__ / token checks."""
    from gateway.platforms.discord import DiscordAdapter
    from gateway.platforms.base import Platform
    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    # ``platform`` is set by BasePlatformAdapter.__init__, which we skip
    # above; the inherited ``.name`` property dereferences it for log
    # formatting, so set it explicitly.
    adapter.platform = Platform.DISCORD
    return adapter


class TestRefreshSkillGroup:
    def test_refresh_repopulates_entries_after_catalog_change(
        self, monkeypatch
    ) -> None:
        """The initial catalog is replaced wholesale on refresh.

        Mirrors the observable /reload-skills case: a user adds a new
        skill to ~/.hermes/skills/, runs /reload-skills, and expects
        the autocomplete to surface it on the very next keystroke.
        """
        adapter = _make_adapter()

        # Start-of-process state: /register built the catalog from the
        # original collector output.
        adapter._skill_entries = [
            ("old-skill", "Pre-existing skill", "/old-skill"),
        ]
        adapter._skill_lookup = {"old-skill": ("Pre-existing skill", "/old-skill")}
        adapter._skill_group_reserved_names = set()
        adapter._skill_group_hidden_count = 0

        # User adds new-skill to disk and removes old-skill.
        def fake_collector(*, reserved_names):
            return (
                {"creative": [("new-skill", "Fresh skill", "/new-skill")]},  # categories
                [],  # uncategorized
                0,   # hidden
            )

        monkeypatch.setattr(
            "hermes_cli.commands.discord_skill_commands_by_category",
            fake_collector,
        )

        new_count, hidden = adapter.refresh_skill_group()

        assert new_count == 1
        assert hidden == 0
        # Old skill is gone, new skill is present.
        names = [n for n, _d, _k in adapter._skill_entries]
        assert names == ["new-skill"]
        assert "old-skill" not in adapter._skill_lookup
        assert adapter._skill_lookup["new-skill"] == ("Fresh skill", "/new-skill")

    def test_refresh_sorts_entries_alphabetically(self, monkeypatch) -> None:
        """Autocomplete order must be stable and predictable across refreshes."""
        adapter = _make_adapter()
        adapter._skill_entries = []
        adapter._skill_lookup = {}
        adapter._skill_group_reserved_names = set()
        adapter._skill_group_hidden_count = 0

        def fake_collector(*, reserved_names):
            # Intentionally unsorted — the fix must resort.
            return (
                {"zzz": [("zebra", "", "/zebra")]},
                [("alpha", "", "/alpha")],
                0,
            )

        monkeypatch.setattr(
            "hermes_cli.commands.discord_skill_commands_by_category",
            fake_collector,
        )

        adapter.refresh_skill_group()

        names = [n for n, _d, _k in adapter._skill_entries]
        assert names == sorted(names) == ["alpha", "zebra"]

    def test_refresh_handles_collector_exception_gracefully(
        self, monkeypatch
    ) -> None:
        """A broken collector must not take down /reload-skills."""
        adapter = _make_adapter()
        adapter._skill_entries = [("keep", "kept", "/keep")]
        adapter._skill_lookup = {"keep": ("kept", "/keep")}
        adapter._skill_group_reserved_names = set()
        adapter._skill_group_hidden_count = 0

        def boom(*, reserved_names):
            raise RuntimeError("simulated collector failure")

        monkeypatch.setattr(
            "hermes_cli.commands.discord_skill_commands_by_category",
            boom,
        )

        new_count, hidden = adapter.refresh_skill_group()
        # Returns previously-cached count, no crash, existing entries
        # preserved so the live autocomplete keeps working.
        assert new_count == 1
        assert hidden == 0
        assert adapter._skill_entries == [("keep", "kept", "/keep")]


class TestRegisterSkillGroupUsesInstanceState:
    """The closure-based ``entries`` / ``skill_lookup`` must be gone.

    If the callbacks in ``_register_skill_group`` still close over
    local variables instead of reading from ``self``, the refresh
    method is useless — autocomplete will keep serving the stale list.

    The full slash-command registration path pulls in ``discord.app_commands``
    decorators (``@describe`` / ``@autocomplete`` / ``Command``), which
    are unstubbed in the hermetic test env. We assert the data-shaped
    side-effects instead: after ``_register_skill_group`` returns
    (successfully or not), ``_skill_entries`` and ``_skill_lookup`` must
    be populated from the collector output, because
    ``_refresh_skill_catalog_state`` runs before any decorator evaluation.
    """

    def test_refresh_catalog_state_populates_instance_attrs(
        self, monkeypatch
    ) -> None:
        adapter = _make_adapter()
        adapter._skill_group_reserved_names = set()

        def fake_collector(*, reserved_names):
            return (
                {"creative": [("ascii-art", "Make ASCII", "/ascii-art")]},
                [],
                0,
            )
        monkeypatch.setattr(
            "hermes_cli.commands.discord_skill_commands_by_category",
            fake_collector,
        )

        adapter._refresh_skill_catalog_state()

        # Instance-level state populated — the autocomplete + handler
        # callbacks both read from these, so `refresh_skill_group`
        # mutating them in place is enough to pick up new skills.
        assert adapter._skill_entries == [
            ("ascii-art", "Make ASCII", "/ascii-art"),
        ]
        assert adapter._skill_lookup == {
            "ascii-art": ("Make ASCII", "/ascii-art"),
        }
        assert adapter._skill_group_hidden_count == 0


class TestHandleReloadSkillsCallsRefreshSkillGroup:
    """Gateway-side integration: /reload-skills must call refresh on adapters."""

    def test_orchestrator_calls_refresh_skill_group_on_every_adapter(self):
        """Sync + async refresh_skill_group implementations both get awaited/called.

        The orchestrator iterates ``self.adapters`` and calls
        ``refresh_skill_group`` if it exists. Adapters that don't
        implement it (today: everything except Discord) are silently
        skipped without raising.
        """
        import asyncio
        from unittest.mock import patch, MagicMock

        # Import without constructing a real runner — test the method
        # directly against an ``object.__new__`` instance.
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)

        sync_refresh = MagicMock(return_value=(5, 0))
        async_called = {"flag": False}

        class AsyncAdapter:
            name = "async-platform"
            async def refresh_skill_group(self):
                async_called["flag"] = True
                return (3, 0)

        class SyncAdapter:
            name = "sync-platform"
            refresh_skill_group = sync_refresh

        class NoOpAdapter:
            name = "other"
            # No refresh_skill_group — must not crash.

        runner.adapters = {
            "discord": AsyncAdapter(),
            "slack": SyncAdapter(),
            "telegram": NoOpAdapter(),
        }

        # Mock reload_skills itself so no disk scan runs.
        fake_result = {"added": [], "removed": [], "total": 7}
        with patch(
            "agent.skill_commands.reload_skills", return_value=fake_result
        ):
            event = MagicMock()
            event.source = MagicMock()
            # _session_key_for_source may be called — make it safe.
            runner._session_key_for_source = lambda src: None
            runner._pending_skills_reload_notes = {}

            result = asyncio.get_event_loop().run_until_complete(
                runner._handle_reload_skills_command(event)
            )

        assert "Skills Reloaded" in result
        assert sync_refresh.called, "sync adapter refresh must be invoked"
        assert async_called["flag"], "async adapter refresh must be awaited"
