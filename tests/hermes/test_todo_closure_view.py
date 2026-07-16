"""Adapter-level tests for TodoClosureView and the away-mode bedtime gate.

TodoClosureView (plugins/platforms/discord/adapter.py) is the select-then-
button control posted by the daily todo-closure-view dispatcher, modeled
directly on JournalApproveView's select-then-button pattern — same auth gate
(_component_check_auth), same "select records values, buttons act on them"
shape. These tests mirror tests/hermes/test_journal_approve_view.py's
mocking approach (real discord.py, MagicMock/AsyncMock Interaction stubs)
rather than reinventing a stub Interaction class.

The second half of this file covers the away-mode kill switch wired into
DiscordAdapter._bedtime_scheduler_loop: when away_mode.is_away() is true,
the loop must skip straight past the backlog fetch (no fetch_backlog_count
call) for that night.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord", reason="discord.py not installed")

import plugins.life_ops.discord_adapter as adapter_mod
from gateway.config import Platform
from plugins.life_ops import away_mode
from plugins.life_ops import todo_store as ts

discord = adapter_mod.discord
if not hasattr(adapter_mod, "TodoClosureView"):
    pytest.skip("TodoClosureView unavailable (discord guard not taken)", allow_module_level=True)
TodoClosureView = adapter_mod.TodoClosureView


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


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


def _todos(*pairs, priority="medium"):
    """[{"key": k, "text": t, "priority": priority}, ...] for view construction."""
    return [{"key": k, "text": t, "priority": priority} for k, t in pairs]


def _make_view(open_todos, allowed_user_ids=None, allowed_role_ids=None) -> "TodoClosureView":
    return TodoClosureView(
        open_todos=open_todos,
        allowed_user_ids=allowed_user_ids or {"111"},
        allowed_role_ids=allowed_role_ids or set(),
    )


def _buttons(view) -> dict:
    """{label: discord.ui.Button} for the view's three action buttons."""
    return {item.label: item for item in view.children if isinstance(item, discord.ui.Button)}


def _seed_open(key, text="fake task text", for_date="2026-07-14", priority="medium"):
    ts.upsert_from_contract(
        [{"key": key, "text": text, "priority": priority, "source_dates": [for_date]}],
        for_date,
    )


# ---------------------------------------------------------------------------
# Auth gate — select
# ---------------------------------------------------------------------------


class TestSelectAuthGate:
    @pytest.mark.asyncio
    async def test_authorized_select_records_values(self):
        view = _make_view(_todos(("k1", "Task one"), ("k2", "Task two")), allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="111", values=["k1"])

        await view._on_select(interaction)

        assert view.selected == {"k1"}
        interaction.response.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unauthorized_select_is_rejected(self):
        view = _make_view(_todos(("k1", "Task one")), allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="999", values=["k1"])

        await view._on_select(interaction)

        assert view.selected == set()
        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        args, _ = interaction.response.send_message.call_args
        assert "not authorised" in args[0].lower()


# ---------------------------------------------------------------------------
# Auth gate — buttons
# ---------------------------------------------------------------------------


class TestButtonAuthGate:
    @pytest.mark.asyncio
    async def test_unauthorized_mark_done_does_not_call_close_todo(self, monkeypatch):
        view = _make_view(_todos(("k1", "Task one")), allowed_user_ids={"111"})
        view.selected = {"k1"}

        mock_close = MagicMock(return_value={"ok": True, "status": "done"})
        monkeypatch.setattr("plugins.life_ops.todo_store.close_todo", mock_close)

        interaction = _make_interaction(user_id="999")
        await _buttons(view)["Mark Done"].callback(interaction)

        mock_close.assert_not_called()
        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_unauthorized_dismiss_does_not_call_close_todo(self, monkeypatch):
        view = _make_view(_todos(("k1", "Task one")), allowed_user_ids={"111"})
        view.selected = {"k1"}

        mock_close = MagicMock(return_value={"ok": True, "status": "dismissed"})
        monkeypatch.setattr("plugins.life_ops.todo_store.close_todo", mock_close)

        interaction = _make_interaction(user_id="999")
        await _buttons(view)["Dismiss"].callback(interaction)

        mock_close.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_snooze_does_not_call_close_todo(self, monkeypatch):
        view = _make_view(_todos(("k1", "Task one")), allowed_user_ids={"111"})
        view.selected = {"k1"}

        mock_close = MagicMock(return_value={"ok": True, "status": "snoozed"})
        monkeypatch.setattr("plugins.life_ops.todo_store.close_todo", mock_close)

        interaction = _make_interaction(user_id="999")
        await _buttons(view)["Snooze 1 week"].callback(interaction)

        mock_close.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_button_click_with_nothing_selected_prompts_to_select(self):
        view = _make_view(_todos(("k1", "Task one")), allowed_user_ids={"111"})
        interaction = _make_interaction(user_id="111")

        await _buttons(view)["Mark Done"].callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, _ = interaction.response.send_message.call_args
        assert "select at least one" in args[0].lower()


# ---------------------------------------------------------------------------
# Select + button flow against a real (tmp HERMES_HOME) todo_store
# ---------------------------------------------------------------------------


class TestApplyActionsAgainstRealStore:
    @pytest.mark.asyncio
    async def test_mark_done_closes_each_selected_key_and_removes_from_options(self, tmp_path):
        _seed_open("k1", text="First task")
        _seed_open("k2", text="Second task")
        _seed_open("k3", text="Third task")
        open_todos = ts.get_open_todos()
        view = _make_view(open_todos, allowed_user_ids={"111"})

        select_interaction = _make_interaction(user_id="111", values=["k1", "k2"])
        await view._on_select(select_interaction)
        assert view.selected == {"k1", "k2"}

        apply_interaction = _make_interaction(user_id="111")
        await _buttons(view)["Mark Done"].callback(apply_interaction)

        conn = ts.connect(tmp_path / "todos.db")
        try:
            rows = {
                r["key"]: r["status"]
                for r in conn.execute("SELECT key, status FROM todos").fetchall()
            }
        finally:
            conn.close()
        assert rows["k1"] == "done"
        assert rows["k2"] == "done"
        assert rows["k3"] == "open"  # untouched — was never selected

        remaining_values = {opt.value for opt in view._select.options}
        assert "k1" not in remaining_values
        assert "k2" not in remaining_values
        assert "k3" in remaining_values
        # Selection is cleared after a successful apply.
        assert view.selected == set()

    @pytest.mark.asyncio
    async def test_dismiss_closes_selected_keys_as_dismissed(self):
        _seed_open("k1", text="First task")
        view = _make_view(ts.get_open_todos(), allowed_user_ids={"111"})

        await view._on_select(_make_interaction(user_id="111", values=["k1"]))
        await _buttons(view)["Dismiss"].callback(_make_interaction(user_id="111"))

        result = ts.close_todo("k1", "dismiss", "test")  # idempotent re-close check
        assert result["status"] == "dismissed"

    @pytest.mark.asyncio
    async def test_snooze_closes_selected_keys_as_snoozed_one_week_out(self):
        _seed_open("k1", text="First task")
        view = _make_view(ts.get_open_todos(), allowed_user_ids={"111"})

        await view._on_select(_make_interaction(user_id="111", values=["k1"]))
        await _buttons(view)["Snooze 1 week"].callback(_make_interaction(user_id="111"))

        open_keys = {d["key"] for d in ts.get_open_keys()}
        assert "k1" not in open_keys  # snoozed, so no longer open

    @pytest.mark.asyncio
    async def test_all_options_removed_disables_the_select(self):
        _seed_open("only-key", text="Only task")
        view = _make_view(ts.get_open_todos(), allowed_user_ids={"111"})

        await view._on_select(_make_interaction(user_id="111", values=["only-key"]))
        await _buttons(view)["Mark Done"].callback(_make_interaction(user_id="111"))

        assert view._select.disabled is True
        assert [opt.value for opt in view._select.options] == ["__none__"]


# ---------------------------------------------------------------------------
# Bedtime scheduler — away-mode kill switch
# ---------------------------------------------------------------------------


def _bare_adapter() -> "adapter_mod.LifeOpsDiscordAdapter":
    """A DiscordAdapter instance with __init__ bypassed.

    _bedtime_scheduler_loop only touches self.name (a property derived from
    self.platform) and self._client before/at the point this test cuts the
    loop short, so a full PlatformConfig-backed construction (heavy: voice
    state, thread trackers, etc.) is unnecessary — same "construct the bare
    minimum" approach as instantiating a dataclass-like test double.
    """
    adapter = object.__new__(adapter_mod.LifeOpsDiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._client = None
    return adapter


class TestBedtimeAwayModeGate:
    @pytest.mark.asyncio
    async def test_away_mode_active_skips_backlog_fetch(self, tmp_path):
        away_mode.set_away(until="2099-01-01")
        adapter = _bare_adapter()

        sleep_calls = {"n": 0}

        async def _fake_sleep(_delay):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise asyncio.CancelledError()
            return None

        with patch("asyncio.sleep", side_effect=_fake_sleep), \
             patch("plugins.life_ops.bedtime.fetch_backlog_count") as mock_fetch:
            await adapter._bedtime_scheduler_loop({"hour": 3, "minute": 0})

        mock_fetch.assert_not_called()
        assert sleep_calls["n"] == 2  # one skipped night, then cancelled on the second sleep

    @pytest.mark.asyncio
    async def test_away_mode_inactive_proceeds_to_backlog_fetch(self, tmp_path):
        # away_mode never set -> is_away() is False -> loop proceeds past the
        # gate on the first iteration and (since self._client is None) exits
        # via the "DISCORD_HOME_CHANNEL not set" continue after fetching.
        adapter = _bare_adapter()

        sleep_calls = {"n": 0}

        async def _fake_sleep(_delay):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise asyncio.CancelledError()
            return None

        with patch("asyncio.sleep", side_effect=_fake_sleep), \
             patch("plugins.life_ops.bedtime.fetch_backlog_count", return_value=3) as mock_fetch:
            await adapter._bedtime_scheduler_loop({"hour": 3, "minute": 0})

        mock_fetch.assert_called_once()
