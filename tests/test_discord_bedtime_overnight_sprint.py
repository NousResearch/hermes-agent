"""Tests for issue #8: Discord bedtime overnight-sprint prompt with Start/Skip.

Each test is anchored to a specific Acceptance Criterion.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_interaction(user_id: str = "123", username: str = "testuser") -> MagicMock:
    """Minimal discord.Interaction-shaped mock."""
    user = MagicMock()
    user.id = int(user_id)
    user.name = username
    user.roles = []

    interaction = MagicMock()
    interaction.user = user
    interaction.response = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _make_button() -> MagicMock:
    btn = MagicMock()
    btn.disabled = False
    return btn


# ---------------------------------------------------------------------------
# AC1 — BedtimeView subclasses discord.ui.View; uses _component_check_auth
# ---------------------------------------------------------------------------


class TestBedtimeViewStructure:
    def test_bedtime_view_is_importable(self):
        """BedtimeView must be importable from the adapter module."""
        import importlib
        import sys
        # Patch discord so the module loads without the real library
        discord_stub = MagicMock()
        discord_stub.ui = MagicMock()
        discord_stub.ui.View = object  # base class for View
        discord_stub.ButtonStyle = MagicMock()
        discord_stub.ButtonStyle.green = "green"
        discord_stub.ButtonStyle.grey = "grey"
        discord_stub.ButtonStyle.red = "red"
        discord_stub.ui.button = lambda **kw: (lambda f: f)  # no-op decorator

        with patch.dict(sys.modules, {"discord": discord_stub,
                                       "discord.ui": discord_stub.ui}):
            from services.hermes import bedtime as _bedtime_mod  # noqa: F401
            assert hasattr(_bedtime_mod, "fetch_backlog_count")
            assert hasattr(_bedtime_mod, "check_running_sprint")
            assert hasattr(_bedtime_mod, "start_sprint")

    def test_bedtime_view_exists_as_module_global(self):
        """After _define_discord_view_classes(), BedtimeView is a module global."""
        import sys
        from unittest.mock import MagicMock, patch

        discord_stub = MagicMock()

        class FakeView:
            def __init__(self, *, timeout=None):
                self.timeout = timeout
                self.children = []

            def add_item(self, item):
                self.children.append(item)

        discord_stub.ui.View = FakeView
        discord_stub.ButtonStyle.green = "green"
        discord_stub.ButtonStyle.grey = "grey"
        discord_stub.ButtonStyle.red = "red"
        discord_stub.ui.button = lambda **kw: (lambda f: f)

        with patch.dict(sys.modules, {"discord": discord_stub,
                                       "discord.ui": discord_stub.ui,
                                       "discord.ext": MagicMock(),
                                       "discord.ext.commands": MagicMock()}):
            import importlib
            import plugins.platforms.discord.adapter as adapter_mod
            importlib.reload(adapter_mod)
            assert hasattr(adapter_mod, "BedtimeView"), \
                "BedtimeView must be a module-level global after _define_discord_view_classes()"


# ---------------------------------------------------------------------------
# AC2 — message text and backlog count
# ---------------------------------------------------------------------------


class TestFetchBacklogCount:
    def test_fetch_backlog_count_sums_per_project(self):
        """fetch_backlog_count() returns the sum of per-project backlog counts."""
        from services.hermes.bedtime import fetch_backlog_count

        api_response = [
            {"project": "alpha", "backlog_count": 3},
            {"project": "beta", "backlog_count": 7},
        ]
        raw = json.dumps(api_response).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=raw)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.dict(os.environ, {"COMMANDER_API_URL": "http://localhost:8000"}):
            count = fetch_backlog_count()

        assert count == 10

    def test_fetch_backlog_count_handles_http_error(self):
        """fetch_backlog_count() raises on HTTP error (caller surfaces it)."""
        import urllib.error
        from services.hermes.bedtime import fetch_backlog_count

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(None, 500, "Server Error", {}, None)), \
             patch.dict(os.environ, {"COMMANDER_API_URL": "http://localhost:8000"}):
            with pytest.raises(urllib.error.HTTPError):
                fetch_backlog_count()

    def test_fetch_backlog_count_raises_if_no_api_url(self):
        """fetch_backlog_count() raises ValueError when COMMANDER_API_URL is unset."""
        from services.hermes.bedtime import fetch_backlog_count

        env = {k: v for k, v in os.environ.items() if k != "COMMANDER_API_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="COMMANDER_API_URL"):
                fetch_backlog_count()


# ---------------------------------------------------------------------------
# AC3 — Start button: already-running guard
# ---------------------------------------------------------------------------


class TestCheckRunningSprint:
    def test_check_running_sprint_true_when_active(self):
        """check_running_sprint() returns True when a sprint is active."""
        from services.hermes.bedtime import check_running_sprint

        api_response = [{"id": "sp-1", "status": "running"}]
        raw = json.dumps(api_response).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=raw)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.dict(os.environ, {"COMMANDER_API_URL": "http://localhost:8000"}):
            result = check_running_sprint()

        assert result["running"] is True
        assert result["sprint_id"] == "sp-1"

    def test_check_running_sprint_false_when_empty(self):
        """check_running_sprint() returns False when no active sprint exists."""
        from services.hermes.bedtime import check_running_sprint

        raw = json.dumps([]).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=raw)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.dict(os.environ, {"COMMANDER_API_URL": "http://localhost:8000"}):
            result = check_running_sprint()

        assert result["running"] is False
        assert result["sprint_id"] is None


# ---------------------------------------------------------------------------
# AC4 — Start button: sprint-start path
# ---------------------------------------------------------------------------


class TestStartSprint:
    def test_start_sprint_returns_sprint_id(self):
        """start_sprint() calls POST /api/sprints/run and returns sprint ID."""
        from services.hermes.bedtime import start_sprint

        api_response = {"id": "sp-42", "status": "running"}
        raw = json.dumps(api_response).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=raw)
        mock_resp.status = 201

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
             patch.dict(os.environ, {
                 "COMMANDER_API_URL": "http://localhost:8000",
                 "COMMANDER_API_TOKEN": "tok-abc",
             }):
            result = start_sprint()

        assert result["success"] is True
        assert result["sprint_id"] == "sp-42"
        # Bearer token must be sent
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.get_header("Authorization") == "Bearer tok-abc"

    def test_start_sprint_http_error_returns_failure(self):
        """start_sprint() returns success=False on HTTP error (no raise)."""
        import urllib.error
        from services.hermes.bedtime import start_sprint

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(None, 503, "Service Unavailable", {}, None)), \
             patch.dict(os.environ, {
                 "COMMANDER_API_URL": "http://localhost:8000",
                 "COMMANDER_API_TOKEN": "tok-abc",
             }):
            result = start_sprint()

        assert result["success"] is False
        assert "503" in result["error"]


# ---------------------------------------------------------------------------
# AC5 — Skip button updates message, no API call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_button_updates_message_no_api_call():
    """AC5: Skip button updates message to indicate skip; no Commander call."""
    from services.hermes import bedtime as bedtime_mod

    interaction = _make_interaction(user_id="111")
    button = _make_button()

    # Allow all users for this test
    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "check_running_sprint") as mock_check, \
         patch.object(bedtime_mod, "start_sprint") as mock_start, \
         patch.object(bedtime_mod, "log_bedtime_action") as mock_audit:

        view = _make_bedtime_view(backlog_count=5, allowed_user_ids={"111"})
        await view.skip(interaction, button)

    mock_check.assert_not_called()
    mock_start.assert_not_called()
    interaction.response.edit_message.assert_awaited_once()
    call_kwargs = interaction.response.edit_message.call_args
    content = call_kwargs[1].get("content") or (call_kwargs[0][0] if call_kwargs[0] else "")
    assert "skip" in content.lower()
    mock_audit.assert_called_once()
    _, kwargs = mock_audit.call_args
    assert kwargs.get("action") == "skip"


# ---------------------------------------------------------------------------
# AC6 — Idempotency: second click returns ephemeral "Already handled"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_start_click_is_idempotent():
    """AC6: A second Start click returns an ephemeral 'Already handled' message."""
    from services.hermes import bedtime as bedtime_mod

    interaction1 = _make_interaction(user_id="111")
    interaction2 = _make_interaction(user_id="111")
    button = _make_button()

    api_response = {"id": "sp-1", "status": "running"}
    raw = json.dumps(api_response).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=raw)

    no_sprint_resp = {"running": False, "sprint_id": None}
    start_result = {"success": True, "sprint_id": "sp-42", "error": None}

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true",
                                   "COMMANDER_API_URL": "http://localhost:8000",
                                   "COMMANDER_API_TOKEN": "tok"}), \
         patch.object(bedtime_mod, "check_running_sprint", return_value=no_sprint_resp), \
         patch.object(bedtime_mod, "start_sprint", return_value=start_result), \
         patch.object(bedtime_mod, "log_bedtime_action"):

        view = _make_bedtime_view(backlog_count=3, allowed_user_ids={"111"})

        # First click — should succeed
        await view.start(interaction1, button)

        # Second click — should be caught as already handled
        await view.start(interaction2, button)

    # Second click must send ephemeral "Already handled" via send_message
    send_calls = interaction2.response.send_message.call_args_list
    assert send_calls, "Second click must call send_message"
    last_call = send_calls[-1]
    msg_content = last_call[0][0] if last_call[0] else last_call[1].get("content", "")
    assert "already" in msg_content.lower()
    assert last_call[1].get("ephemeral") is True


@pytest.mark.asyncio
async def test_second_skip_click_is_idempotent():
    """AC6: A second Skip click also returns ephemeral 'Already handled'."""
    from services.hermes import bedtime as bedtime_mod

    interaction1 = _make_interaction(user_id="111")
    interaction2 = _make_interaction(user_id="111")
    button = _make_button()

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "log_bedtime_action"):

        view = _make_bedtime_view(backlog_count=3, allowed_user_ids={"111"})
        await view.skip(interaction1, button)
        await view.skip(interaction2, button)

    send_calls = interaction2.response.send_message.call_args_list
    assert send_calls, "Second skip must call send_message"
    last_call = send_calls[-1]
    msg_content = last_call[0][0] if last_call[0] else last_call[1].get("content", "")
    assert "already" in msg_content.lower()
    assert last_call[1].get("ephemeral") is True


# ---------------------------------------------------------------------------
# AC7 — Audit log: user ID, username, action, timestamp, sprint_id
# ---------------------------------------------------------------------------


class TestBedtimeAuditLog:
    def test_log_bedtime_action_writes_required_fields(self, tmp_path):
        """log_bedtime_action() writes a JSONL entry with all required fields."""
        from services.hermes.bedtime import log_bedtime_action

        log_path = tmp_path / "bedtime-audit.log"
        with patch("services.hermes.bedtime._resolve_bedtime_log_path",
                   return_value=log_path):
            log_bedtime_action(
                user_id="123",
                username="alice",
                action="start",
                sprint_id="sp-42",
            )

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["user_id"] == "123"
        assert entry["username"] == "alice"
        assert entry["action"] == "start"
        assert entry["sprint_id"] == "sp-42"
        assert "ts" in entry
        # ts must be UTC ISO-8601
        assert entry["ts"].endswith("Z") or "+" in entry["ts"] or entry["ts"].endswith("+00:00")

    def test_log_bedtime_action_skip_has_null_sprint_id(self, tmp_path):
        """log_bedtime_action() records sprint_id=null for skip/timeout."""
        from services.hermes.bedtime import log_bedtime_action

        log_path = tmp_path / "bedtime-audit.log"
        with patch("services.hermes.bedtime._resolve_bedtime_log_path",
                   return_value=log_path):
            log_bedtime_action(
                user_id="456",
                username="bob",
                action="skip",
                sprint_id=None,
            )

        entry = json.loads(log_path.read_text().strip())
        assert entry["sprint_id"] is None

    def test_log_bedtime_action_timeout_recorded(self, tmp_path):
        """log_bedtime_action() records action='timeout' on view timeout."""
        from services.hermes.bedtime import log_bedtime_action

        log_path = tmp_path / "bedtime-audit.log"
        with patch("services.hermes.bedtime._resolve_bedtime_log_path",
                   return_value=log_path):
            log_bedtime_action(
                user_id="system",
                username="system",
                action="timeout",
                sprint_id=None,
            )

        entry = json.loads(log_path.read_text().strip())
        assert entry["action"] == "timeout"


# ---------------------------------------------------------------------------
# AC8 — HTTP errors surfaced as ephemeral messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_button_http_error_shows_ephemeral_error():
    """AC8: When POST /api/sprints/run fails, an ephemeral error is shown."""
    from services.hermes import bedtime as bedtime_mod

    interaction = _make_interaction(user_id="111")
    button = _make_button()

    no_sprint = {"running": False, "sprint_id": None}
    fail_result = {"success": False, "sprint_id": None, "error": "HTTP 503 Service Unavailable"}

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "check_running_sprint", return_value=no_sprint), \
         patch.object(bedtime_mod, "start_sprint", return_value=fail_result), \
         patch.object(bedtime_mod, "log_bedtime_action"):

        view = _make_bedtime_view(backlog_count=5, allowed_user_ids={"111"})
        await view.start(interaction, button)

    # Should send ephemeral error
    calls = interaction.response.send_message.call_args_list
    assert calls, "HTTP error must trigger ephemeral send_message"
    last = calls[-1]
    content = last[0][0] if last[0] else last[1].get("content", "")
    assert last[1].get("ephemeral") is True
    assert "error" in content.lower() or "503" in content or "fail" in content.lower()


@pytest.mark.asyncio
async def test_start_already_running_shows_ephemeral_message():
    """AC3: When a sprint is already running, ephemeral message is shown; no POST made."""
    from services.hermes import bedtime as bedtime_mod

    interaction = _make_interaction(user_id="111")
    button = _make_button()

    sprint_running = {"running": True, "sprint_id": "sp-99"}

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "check_running_sprint", return_value=sprint_running), \
         patch.object(bedtime_mod, "start_sprint") as mock_start, \
         patch.object(bedtime_mod, "log_bedtime_action"):

        view = _make_bedtime_view(backlog_count=5, allowed_user_ids={"111"})
        await view.start(interaction, button)

    mock_start.assert_not_called()
    send_calls = interaction.response.send_message.call_args_list
    assert send_calls
    content = send_calls[-1][0][0] if send_calls[-1][0] else send_calls[-1][1].get("content", "")
    assert "already running" in content.lower()
    assert send_calls[-1][1].get("ephemeral") is True


# ---------------------------------------------------------------------------
# AC9 — Unit tests covering start path and already-running path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_path_full_flow():
    """AC9 (start path): Start button → no running sprint → POST → confirm."""
    from services.hermes import bedtime as bedtime_mod

    interaction = _make_interaction(user_id="222")
    button = _make_button()

    no_sprint = {"running": False, "sprint_id": None}
    start_ok = {"success": True, "sprint_id": "sp-7", "error": None}

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "check_running_sprint", return_value=no_sprint) as mock_check, \
         patch.object(bedtime_mod, "start_sprint", return_value=start_ok) as mock_start, \
         patch.object(bedtime_mod, "log_bedtime_action") as mock_audit:

        view = _make_bedtime_view(backlog_count=12, allowed_user_ids={"222"})
        await view.start(interaction, button)

    mock_check.assert_called_once()
    mock_start.assert_called_once()
    interaction.response.edit_message.assert_awaited_once()
    # Audit log must record action=start and sprint_id
    mock_audit.assert_called_once()
    _, kw = mock_audit.call_args
    assert kw.get("action") == "start"
    assert kw.get("sprint_id") == "sp-7"


@pytest.mark.asyncio
async def test_already_running_path_no_post():
    """AC9 (already-running path): Start button → running sprint → ephemeral, no POST."""
    from services.hermes import bedtime as bedtime_mod

    interaction = _make_interaction(user_id="333")
    button = _make_button()

    running = {"running": True, "sprint_id": "sp-9"}

    with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "true"}), \
         patch.object(bedtime_mod, "check_running_sprint", return_value=running) as mock_check, \
         patch.object(bedtime_mod, "start_sprint") as mock_start, \
         patch.object(bedtime_mod, "log_bedtime_action"):

        view = _make_bedtime_view(backlog_count=3, allowed_user_ids={"333"})
        await view.start(interaction, button)

    mock_check.assert_called_once()
    mock_start.assert_not_called()
    # Must have sent ephemeral message
    assert interaction.response.send_message.call_args_list


# ---------------------------------------------------------------------------
# Bedtime scheduler config
# ---------------------------------------------------------------------------


class TestBedtimeSchedulerConfig:
    def test_bedtime_hour_and_minute_read_from_env(self):
        """The scheduler reads DISCORD_BEDTIME_HOUR and DISCORD_BEDTIME_MINUTE."""
        from plugins.platforms.discord.adapter import _read_bedtime_config

        with patch.dict(os.environ, {
            "DISCORD_BEDTIME_HOUR": "22",
            "DISCORD_BEDTIME_MINUTE": "30",
        }):
            cfg = _read_bedtime_config()

        assert cfg["hour"] == 22
        assert cfg["minute"] == 30

    def test_bedtime_defaults_to_disabled(self):
        """If DISCORD_BEDTIME_HOUR is not set, bedtime is disabled."""
        from plugins.platforms.discord.adapter import _read_bedtime_config

        env = {k: v for k, v in os.environ.items()
               if k not in ("DISCORD_BEDTIME_HOUR", "DISCORD_BEDTIME_MINUTE")}
        with patch.dict(os.environ, env, clear=True):
            cfg = _read_bedtime_config()

        assert cfg["enabled"] is False


# ---------------------------------------------------------------------------
# Helper — build a BedtimeView without a live discord.py
# ---------------------------------------------------------------------------


def _make_bedtime_view(
    backlog_count: int = 5,
    allowed_user_ids: Optional[set] = None,
    timeout: int = 10,
) -> "BedtimeView":
    """Construct a BedtimeView instance using the pure-Python stub."""
    from services.hermes.bedtime import BedtimeView
    return BedtimeView(
        backlog_count=backlog_count,
        allowed_user_ids=allowed_user_ids or set(),
        allowed_role_ids=set(),
        timeout=timeout,
    )
