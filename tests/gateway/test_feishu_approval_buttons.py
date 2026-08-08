"""Tests for Feishu interactive card approval buttons."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Feishu mock so FeishuAdapter can be imported without lark-oapi
# ---------------------------------------------------------------------------
def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so the import succeeds."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
    """Create a FeishuAdapter with mocked internals."""
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_abc",
) -> SimpleNamespace:
    """Create a mock Feishu card action callback data object."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(
                tag="button",
                value=action_value,
            ),
        ),
    )


def _close_submitted_coro(coro, _loop):
    """Close scheduled coroutines in sync-handler tests to avoid unawaited warnings."""
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)


class _RunCoroFuture:
    """A fake Future that runs the coroutine synchronously on creation.

    Used where a test must verify that the coroutine passed to
    ``_submit_on_loop`` actually drove a downstream call (e.g. that
    ``tools.slash_confirm.resolve`` ran with the chosen value).
    """

    def __init__(self, coro, _loop):
        self._coro = coro
        self._done_callbacks = []

    def add_done_callback(self, fn):
        self._done_callbacks.append(fn)
        return self

    def done(self):
        return True

    def result(self, timeout=None):
        return None


def _run_submitted_coro(coro, _loop, **_kwargs):
    """Schedule a coroutine synchronously (run it to completion inline).

    Mirrors what ``_submit_on_loop`` does via
    ``agent.async_utils.safe_schedule_threadsafe``, but executes the coroutine
    immediately so downstream async calls (e.g. ``tools.slash_confirm.resolve``)
    actually run and can be asserted.  ``safe_schedule_threadsafe`` also accepts
    ``logger``/``log_message``/``log_level`` kwargs, which are ignored here.
    """
    try:
        import asyncio

        async def _runner():
            await coro

        try:
            asyncio.get_running_loop().run_until_complete(_runner())
        except RuntimeError:
            asyncio.run(_runner())
    except Exception:
        pass
    return _RunCoroFuture(coro, _loop)


# ===========================================================================
# send_exec_approval — interactive card with buttons
# ===========================================================================

class TestFeishuExecApproval:
    """Test send_exec_approval sends an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="rm -rf /important",
                session_key="agent:main:feishu:group:oc_12345",
                description="dangerous deletion",
            )

        assert result.success is True
        assert result.message_id == "msg_001"

        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"

        # Verify card payload contains the command and buttons
        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert "rm -rf /important" in card["elements"][0]["content"]
        assert "dangerous deletion" in card["elements"][0]["content"]

        # Check buttons
        actions = card["elements"][1]["actions"]
        assert len(actions) == 4
        action_names = [a["value"]["hermes_action"] for a in actions]
        assert action_names == [
            "approve_once", "approve_session", "approve_always", "deny"
        ]

    @pytest.mark.asyncio
    async def test_stores_approval_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="echo test",
                session_key="my-session-key",
            )

        assert len(adapter._approval_state) == 1
        approval_id = list(adapter._approval_state.keys())[0]
        state = adapter._approval_state[approval_id]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_002"
        assert state["chat_id"] == "oc_12345"


# ===========================================================================
# send_update_prompt — interactive card with buttons
# ===========================================================================

class TestFeishuUpdatePrompt:
    """Test send_update_prompt sends an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_up_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_update_prompt(
                chat_id="oc_12345",
                prompt="Restore stashed changes after update?",
                default="y",
                session_key="agent:main:feishu:group:oc_12345",
                metadata={"thread_id": "th_1"},
            )

        assert result.success is True
        assert result.message_id == "msg_up_001"

        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["metadata"] == {"thread_id": "th_1"}

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert "Restore stashed changes after update?" in card["elements"][0]["content"]
        assert "Default: `y`" in card["elements"][0]["content"]
        actions = card["elements"][1]["actions"]
        assert [a["value"]["hermes_update_prompt_action"] for a in actions] == ["y", "n"]


# ===========================================================================
# _resolve_approval — approval state pop + gateway resolution
# ===========================================================================

class TestResolveApproval:
    """Test _resolve_approval pops state and calls resolve_gateway_approval."""

    @pytest.mark.asyncio
    async def test_resolves_once(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(1, "once", "Norbert", open_id="ou_user1", chat_id="oc_12345")

        mock_resolve.assert_called_once_with("agent:main:feishu:group:oc_12345", "once")
        assert 1 not in adapter._approval_state


    @pytest.mark.asyncio
    async def test_unauthorized_click_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._admins = {"ou_admin"}
        adapter._approval_state[5] = {
            "session_key": "sess-5",
            "message_id": "msg_005",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._resolve_approval(5, "once", "Mallory", open_id="ou_intruder", chat_id="oc_12345")

        mock_resolve.assert_not_called()
        assert 5 in adapter._approval_state


# ===========================================================================
# _handle_card_action_event — non-approval card actions
# ===========================================================================

class TestNonApprovalCardAction:
    """Non-approval card actions should still route as synthetic commands."""

    @pytest.mark.asyncio
    async def test_routes_as_synthetic_command(self):
        adapter = _make_adapter()

        data = _make_card_action_data(
            action_value={"custom_action": "something_else"},
            token="tok_normal",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Dave", "user_id_alt": None},
            ),
            patch.object(adapter, "get_chat_info", new_callable=AsyncMock, return_value={"name": "Test Chat"}),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as mock_handle,
        ):
            await adapter._handle_card_action_event(data)

        mock_handle.assert_called_once()
        event = mock_handle.call_args[0][0]
        assert "/card button" in event.text


# ===========================================================================
# _on_card_action_trigger — inline card response for approval actions
# ===========================================================================

class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture(autouse=False)
def _patch_callback_card_types(monkeypatch):
    """Provide real-ish P2CardActionTriggerResponse / CallBackCard for tests."""
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


class TestCardActionCallbackResponse:
    """Test that _on_card_action_trigger returns updated card inline."""

    def test_drops_action_when_loop_not_ready(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = None
        data = _make_card_action_data({"hermes_action": "approve_once", "approval_id": 1})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_returns_card_for_approve_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._approval_state[1] = {
            "session_key": "sess-1",
            "message_id": "msg-1",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 1},
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert response.card.type == "raw"
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "Approved once" in card["header"]["title"]["content"]
        assert "Bob" in card["elements"][0]["content"]


    def test_ignores_expired_cached_name(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_expired"}
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg-4",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 4},
            open_id="ou_expired",
        )
        adapter._sender_name_cache["ou_expired"] = ("Old Name", 1)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        card = response.card.data
        assert "Old Name" not in card["elements"][0]["content"]
        assert "ou_expired" in card["elements"][0]["content"]

    def test_rejects_approval_click_from_unauthorized_user(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._approval_state[5] = {
            "session_key": "sess-5",
            "message_id": "msg-5",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 5},
            open_id="ou_attacker",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()


    def test_update_prompt_unauthorized_operator_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_006",
            "chat_id": "oc_12345",
        }
        adapter._allowed_group_users = {"ou_allowed"}
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 1},
            open_id="ou_intruder",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()


    def test_update_prompt_chat_mismatch_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._update_prompt_state[8] = {
            "session_key": "sess-up-8",
            "message_id": "msg_up_008",
            "chat_id": "oc_expected",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 8},
            chat_id="oc_mismatch",
            open_id="ou_bob",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 8 in adapter._update_prompt_state
        mock_submit.assert_not_called()


class TestResolveUpdatePrompt:
    """Test update prompt resolution persists the response file."""

    @pytest.mark.asyncio
    async def test_writes_response_file(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_003",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(1, "y", "Alice")

        assert (tmp_path / ".hermes" / ".update_response").read_text() == "y"
        assert 1 not in adapter._update_prompt_state


# ===========================================================================
# Clarify — send_clarify interactive card + multi_select text fallback
# ===========================================================================

class TestFeishuClarifyCard:
    """Test send_clarify renders an interactive card with choice buttons."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card_with_buttons(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_clar_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Which server to deploy?",
                choices=["staging", "prod"],
                clarify_id="clar_001",
                session_key="agent:main:feishu:group:oc_12345",
            )

        assert result.success is True
        assert result.message_id == "msg_clar_001"

        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "blue"
        assert "Which server to deploy?" in card["elements"][0]["content"]
        # Buttons for each choice + "Other"
        actions = card["elements"][-1]["actions"]
        assert len(actions) == 3
        action_values = [a["value"] for a in actions]
        assert action_values[0] == {
            "hermes_clarify_action": "choice",
            "clarify_id": "clar_001",
            "choice_idx": 0,
        }
        assert action_values[-1] == {
            "hermes_clarify_action": "other",
            "clarify_id": "clar_001",
        }

    @pytest.mark.asyncio
    async def test_multi_select_falls_back_to_text_renderer(self):
        """multi_select must NOT render scalar choice buttons (JSON-array
        contract can't be expressed by a single button click). It renders a
        text prompt instead, which the gateway text-intercept resolves."""
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_clar_ms_001"),
        )
        entries = {
            "clar_ms_1": SimpleNamespace(multi_select=True),
        }
        with (
            patch.object(
                adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_send,
            patch(
                "plugins.platforms.feishu.adapter._clarify_entries", entries,
            ) if False else patch("tools.clarify_gateway._entries", entries),
        ):
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Pick all environments",
                choices=["staging", "prod", "dev"],
                clarify_id="clar_ms_1",
                session_key="sess-ms-1",
            )

        assert result.success is True
        card = json.loads(mock_send.call_args[1]["payload"])
        # No action buttons — only markdown elements (question, list, prompt)
        assert not any(e.get("tag") == "action" for e in card["elements"])
        # The multi-select hint is present
        assert any("多选" in e.get("content", "") for e in card["elements"])

    @pytest.mark.asyncio
    async def test_stores_clarify_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_clar_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_clarify(
                chat_id="oc_12345",
                question="Confirm?",
                choices=["yes", "no"],
                clarify_id="clar_002",
                session_key="sess-2",
            )

        state = adapter._clarify_state["clar_002"]
        assert state["session_key"] == "sess-2"
        assert state["message_id"] == "msg_clar_002"
        assert state["chat_id"] == "oc_12345"


class TestClarifyCardAction:
    """Test _handle_clarify_card_action resolves a choice via gateway."""

    def test_choice_button_resolves(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._clarify_state["clar_1"] = {
            "session_key": "sess-1",
            "message_id": "msg-1",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {
                "hermes_clarify_action": "choice",
                "clarify_id": "clar_1",
                "choice_idx": 1,
            },
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        entries = {
            "clar_1": SimpleNamespace(choices=["staging", "prod"]),
        }
        with (
            patch("tools.clarify_gateway._entries", entries),
            patch(
                "agent.async_utils.safe_schedule_threadsafe",
                side_effect=_run_submitted_coro,
            ),
        ):
            response = adapter._handle_clarify_card_action(
                event=data.event, action_value=data.event.action.value, loop=MagicMock(),
            )

        assert response is not None
        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "prod" in card["elements"][0]["content"]
        assert "clar_1" not in adapter._clarify_state


# ===========================================================================
# Slash-confirm — card + whitelist validation (fail closed)
# ===========================================================================

class TestFeishuSlashConfirmCard:
    """Test send_slash_confirm renders a three-button confirmation card."""

    @pytest.mark.asyncio
    async def test_sends_three_button_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Reload MCP?",
                message="This invalidates the provider prompt cache.",
                session_key="sess-sc-1",
                confirm_id="sc_001",
            )

        assert result.success is True
        assert result.message_id == "msg_sc_001"

        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        actions = card["elements"][-1]["actions"]
        assert [a["value"]["hermes_slash_confirm_action"] for a in actions] == [
            "once", "always", "cancel",
        ]
        assert all(a["value"]["confirm_id"] == "sc_001" for a in actions)

    @pytest.mark.asyncio
    async def test_stores_slash_confirm_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm?",
                message="Do it?",
                session_key="sess-sc-2",
                confirm_id="sc_002",
            )

        state = adapter._slash_confirm_state["sc_002"]
        assert state["session_key"] == "sess-sc-2"
        assert state["chat_id"] == "oc_12345"


class TestSlashConfirmCardAction:
    """Test _handle_slash_confirm_card_action resolves valid choices and
    fails closed on malformed callback data."""

    def _make_ready_adapter(self, confirm_id="sc_1"):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._slash_confirm_state[confirm_id] = {
            "session_key": "sess-sc-1",
            "message_id": "msg-sc-1",
            "chat_id": "oc_12345",
        }
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)
        return adapter

    def test_valid_once_resolves(self, _patch_callback_card_types):
        adapter = self._make_ready_adapter()
        data = _make_card_action_data(
            {"hermes_slash_confirm_action": "once", "confirm_id": "sc_1"},
            open_id="ou_bob",
        )

        with (
            patch(
                "tools.slash_confirm.resolve",
                new_callable=AsyncMock, return_value="done",
            ) as mock_resolve,
            patch(
                "agent.async_utils.safe_schedule_threadsafe",
                side_effect=_run_submitted_coro,
            ),
        ):
            response = adapter._handle_slash_confirm_card_action(
                event=data.event, action_value=data.event.action.value, loop=MagicMock(),
            )

        assert response is not None
        assert response.card is not None
        assert response.card.data["header"]["title"]["content"] == "✅ 已批准（仅本次）"
        assert "sc_1" not in adapter._slash_confirm_state
        mock_resolve.assert_called_once_with("sess-sc-1", "sc_1", "once")

    def test_malformed_choice_fails_closed(self, _patch_callback_card_types):
        """A callback with an unexpected choice value must NEVER reach the
        destructive handler. It is rejected, state is NOT popped, and no
        resolve runs."""
        adapter = self._make_ready_adapter()
        data = _make_card_action_data(
            {"hermes_slash_confirm_action": "rm_rf_everything", "confirm_id": "sc_1"},
            open_id="ou_bob",
        )

        with (
            patch(
                "tools.slash_confirm.resolve",
                new_callable=AsyncMock, return_value="done",
            ) as mock_resolve,
            patch("asyncio.run_coroutine_threadsafe") as mock_submit,
        ):
            response = adapter._handle_slash_confirm_card_action(
                event=data.event, action_value=data.event.action.value, loop=MagicMock(),
            )

        assert response is not None
        assert response.card is not None
        assert response.card.data["header"]["title"]["content"] == "❌ 无效操作"
        # State preserved so a legitimate retry isn't blocked, and the
        # destructive handler was never invoked.
        assert "sc_1" in adapter._slash_confirm_state
        mock_resolve.assert_not_called()
        mock_submit.assert_not_called()

    def test_cancel_is_valid(self, _patch_callback_card_types):
        adapter = self._make_ready_adapter()
        data = _make_card_action_data(
            {"hermes_slash_confirm_action": "cancel", "confirm_id": "sc_1"},
            open_id="ou_bob",
        )

        with (
            patch(
                "tools.slash_confirm.resolve",
                new_callable=AsyncMock, return_value="cancelled",
            ) as mock_resolve,
            patch(
                "agent.async_utils.safe_schedule_threadsafe",
                side_effect=_run_submitted_coro,
            ),
        ):
            response = adapter._handle_slash_confirm_card_action(
                event=data.event, action_value=data.event.action.value, loop=MagicMock(),
            )

        assert response is not None
        assert response.card.data["header"]["title"]["content"] == "❌ 已取消"
        assert response.card.data["header"]["template"] == "grey"
        mock_resolve.assert_called_once_with("sess-sc-1", "sc_1", "cancel")

    def test_unauthorized_click_does_not_resolve(self, _patch_callback_card_types):
        adapter = self._make_ready_adapter()
        adapter._allowed_group_users = {"ou_other"}
        data = _make_card_action_data(
            {"hermes_slash_confirm_action": "once", "confirm_id": "sc_1"},
            open_id="ou_intruder",
        )

        with (
            patch(
                "tools.slash_confirm.resolve",
                new_callable=AsyncMock, return_value="done",
            ) as mock_resolve,
            patch("asyncio.run_coroutine_threadsafe") as mock_submit,
        ):
            response = adapter._handle_slash_confirm_card_action(
                event=data.event, action_value=data.event.action.value, loop=MagicMock(),
            )

        assert response is not None
        assert response.card is None
        assert "sc_1" in adapter._slash_confirm_state
        mock_resolve.assert_not_called()
        mock_submit.assert_not_called()


