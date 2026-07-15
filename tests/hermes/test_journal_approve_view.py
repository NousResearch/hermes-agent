"""Adapter-level tests for JournalApproveView's per-todo project selection.

Covers the JOURNAL_APPROVE_PROJECTS routing feature added on top of issue #9
(journal dev-todo approve-to-backlog button):

  - Single configured project -> no dropdown is shown; view.project is fixed.
  - Multiple configured projects (<=25) -> a discord.ui.Select is shown with
    one option per project, defaulting to the first entry.
  - >25 configured projects -> truncated defensively to Discord's cap.
  - Selecting a dropdown option updates view.project (authorized clicker).
  - An unauthorized clicker's selection is rejected and does not mutate
    view.project.
  - The button's Approve flow uses whatever project is currently selected.

discord.py is installed in this environment (see plugins/platforms/discord/
adapter.py's ``DISCORD_AVAILABLE`` gate), so these tests import the adapter
module directly and exercise the real ``discord.ui.View``/``discord.ui.Select``
classes rather than a stub, following the pattern used by
tests/gateway/test_discord_send.py and tests/test_discord_bedtime_overnight_sprint.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("discord", reason="discord.py not installed")

import plugins.platforms.discord.adapter as adapter_mod

discord = adapter_mod.discord
if not hasattr(adapter_mod, "JournalApproveView"):
    pytest.skip("JournalApproveView unavailable (discord guard not taken)", allow_module_level=True)
JournalApproveView = adapter_mod.JournalApproveView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interaction(user_id: str = "111", values: list | None = None) -> MagicMock:
    """Minimal discord.Interaction-shaped mock for component callbacks."""
    user = MagicMock()
    user.id = int(user_id)
    user.roles = []

    interaction = MagicMock()
    interaction.user = user
    interaction.data = {"values": values} if values is not None else {}
    interaction.response = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _approve_button(view):
    """Return the view's [Approve] discord.ui.Button item.

    discord.py's ``@discord.ui.button`` decorator replaces the plain method
    with an ``_ItemCallback`` descriptor bound to the item instance at
    ``View.__init__`` time — calling ``view.approve(interaction, button)``
    directly raises ``TypeError: 'Button' object is not callable`` because
    the instance attribute is the Button item, not the function. The
    supported way to invoke it is via the item's own ``.callback``, which
    already has ``self``/``button`` bound and takes just ``interaction``.
    """
    return _buttons(view)[0]


def _make_view(projects, allowed_user_ids=None, allowed_role_ids=None) -> "JournalApproveView":
    return JournalApproveView(
        todo_id="todo-1",
        title="Fix auth bug",
        body="Details about the bug",
        projects=projects,
        allowed_user_ids=allowed_user_ids or {"111"},
        allowed_role_ids=allowed_role_ids or set(),
    )


def _selects(view) -> list:
    return [item for item in view.children if isinstance(item, discord.ui.Select)]


def _buttons(view) -> list:
    return [item for item in view.children if isinstance(item, discord.ui.Button)]


# ---------------------------------------------------------------------------
# Single project -> no dropdown, project fixed
# ---------------------------------------------------------------------------

class TestSingleProject:
    def test_no_select_added(self):
        view = _make_view(["owner/repo"])
        assert _selects(view) == []

    def test_only_the_approve_button_is_present(self):
        view = _make_view(["owner/repo"])
        assert len(view.children) == 1
        assert isinstance(view.children[0], discord.ui.Button)

    def test_project_is_fixed_to_the_only_entry(self):
        view = _make_view(["owner/repo"])
        assert view.project == "owner/repo"

    def test_empty_projects_list_leaves_project_blank_and_no_select(self):
        """Defensive: an empty projects list (shouldn't normally reach the
        view, since callers skip when get_approve_projects() is empty) must
        not crash and must not add a dropdown."""
        view = _make_view([])
        assert view.project == ""
        assert _selects(view) == []


# ---------------------------------------------------------------------------
# Multiple projects -> dropdown with <=25 options, default first
# ---------------------------------------------------------------------------

class TestMultipleProjects:
    def test_select_is_added_for_multiple_projects(self):
        view = _make_view(["owner/repo1", "owner/repo2", "owner/repo3"])
        selects = _selects(view)
        assert len(selects) == 1

    def test_select_has_one_option_per_project(self):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects)
        select = _selects(view)[0]
        assert [opt.value for opt in select.options] == projects
        assert [opt.label for opt in select.options] == projects

    def test_first_project_is_selected_by_default(self):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects)
        select = _selects(view)[0]
        defaults = [opt.default for opt in select.options]
        assert defaults == [True, False, False]
        assert view.project == "owner/repo1"

    def test_approve_button_still_present_alongside_select(self):
        view = _make_view(["owner/repo1", "owner/repo2"])
        assert len(_buttons(view)) == 1
        assert len(_selects(view)) == 1

    def test_more_than_25_projects_is_truncated_to_25(self):
        projects = [f"owner/repo{i}" for i in range(30)]
        view = _make_view(projects)
        assert len(view.projects) == 25
        select = _selects(view)[0]
        assert len(select.options) == 25
        assert view.projects == projects[:25]

    def test_exactly_25_projects_keeps_select_and_all_options(self):
        projects = [f"owner/repo{i}" for i in range(25)]
        view = _make_view(projects)
        select = _selects(view)[0]
        assert len(select.options) == 25


# ---------------------------------------------------------------------------
# Selection updates view.project (authorized) / rejected (unauthorized)
# ---------------------------------------------------------------------------

class TestProjectSelectionCallback:
    @pytest.mark.asyncio
    async def test_authorized_selection_updates_project(self):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects, allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="111", values=["owner/repo2"])

        await view._on_project_selected(interaction)

        assert view.project == "owner/repo2"
        interaction.response.edit_message.assert_awaited_once()
        interaction.response.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_selection_updates_option_defaults(self):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects, allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="111", values=["owner/repo3"])

        await view._on_project_selected(interaction)

        select = _selects(view)[0]
        defaults = {opt.value: opt.default for opt in select.options}
        assert defaults == {
            "owner/repo1": False,
            "owner/repo2": False,
            "owner/repo3": True,
        }

    @pytest.mark.asyncio
    async def test_unauthorized_selection_is_rejected(self):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects, allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="999", values=["owner/repo2"])

        await view._on_project_selected(interaction)

        # project unchanged (still the default first entry)
        assert view.project == "owner/repo1"
        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        interaction.response.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_selection_message_mentions_authorisation(self):
        projects = ["owner/repo1", "owner/repo2"]
        view = _make_view(projects, allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="999", values=["owner/repo2"])

        await view._on_project_selected(interaction)

        args, _ = interaction.response.send_message.call_args
        message = args[0] if args else ""
        assert "not authorised" in message.lower()

    @pytest.mark.asyncio
    async def test_selection_after_resolved_is_rejected(self):
        """Once the todo has been approved, further dropdown interaction
        (e.g. a stale client re-sending a select event) must not reopen it."""
        projects = ["owner/repo1", "owner/repo2"]
        view = _make_view(projects, allowed_user_ids={"111"})
        view.resolved = True
        interaction = _make_interaction(user_id="111", values=["owner/repo2"])

        await view._on_project_selected(interaction)

        assert view.project == "owner/repo1"
        interaction.response.send_message.assert_awaited_once()
        args, _ = interaction.response.send_message.call_args
        message = args[0] if args else ""
        assert "already" in message.lower()

    @pytest.mark.asyncio
    async def test_missing_values_falls_back_to_current_project(self):
        """Defensive: if interaction.data carries no 'values' key (shouldn't
        happen for a real Select interaction), the current selection is kept
        rather than raising or clearing project."""
        projects = ["owner/repo1", "owner/repo2"]
        view = _make_view(projects, allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="111", values=None)

        await view._on_project_selected(interaction)

        assert view.project == "owner/repo1"
        interaction.response.edit_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Approve() uses whatever project is currently selected
# ---------------------------------------------------------------------------

class TestApproveUsesSelectedProject:
    @pytest.mark.asyncio
    async def test_approve_posts_with_the_selected_project(self, monkeypatch):
        projects = ["owner/repo1", "owner/repo2", "owner/repo3"]
        view = _make_view(projects, allowed_user_ids={"111"})

        select_interaction = _make_interaction(user_id="111", values=["owner/repo3"])
        await view._on_project_selected(select_interaction)
        assert view.project == "owner/repo3"

        mock_handle = MagicMock(
            return_value={"success": True, "duplicate": False, "ticket_number": 5}
        )
        monkeypatch.setattr(
            "services.hermes.journal_approve.handle_journal_approve", mock_handle
        )

        approve_interaction = _make_interaction(user_id="111")
        await _approve_button(view).callback(approve_interaction)

        mock_handle.assert_called_once()
        _, kwargs = mock_handle.call_args
        assert kwargs["project"] == "owner/repo3"

    @pytest.mark.asyncio
    async def test_approve_with_default_single_project(self, monkeypatch):
        view = _make_view(["owner/only-repo"], allowed_user_ids={"111"})

        mock_handle = MagicMock(
            return_value={"success": True, "duplicate": False, "ticket_number": 9}
        )
        monkeypatch.setattr(
            "services.hermes.journal_approve.handle_journal_approve", mock_handle
        )

        interaction = _make_interaction(user_id="111")
        await _approve_button(view).callback(interaction)

        _, kwargs = mock_handle.call_args
        assert kwargs["project"] == "owner/only-repo"

    @pytest.mark.asyncio
    async def test_unauthorized_approve_click_does_not_call_handler(self, monkeypatch):
        view = _make_view(["owner/repo1", "owner/repo2"], allowed_user_ids={"111"})

        mock_handle = MagicMock()
        monkeypatch.setattr(
            "services.hermes.journal_approve.handle_journal_approve", mock_handle
        )

        interaction = _make_interaction(user_id="999")
        await _approve_button(view).callback(interaction)

        mock_handle.assert_not_called()
        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
