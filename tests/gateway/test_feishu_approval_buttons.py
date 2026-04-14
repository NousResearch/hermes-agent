"""Tests for Feishu interactive card approval buttons."""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    if "lark_oapi" not in sys.modules:
        mod = MagicMock()
        # Must cover every submodule path imported by feishu.py's top-level
        # try/except block, otherwise the whole block falls to ImportError
        # and sets P2CardActionTriggerResponse etc. to None.
        for name in (
            "lark_oapi",
            "lark_oapi.api",
            "lark_oapi.api.application",
            "lark_oapi.api.application.v6",
            "lark_oapi.api.im",
            "lark_oapi.api.im.v1",
            "lark_oapi.core",
            "lark_oapi.core.const",
            "lark_oapi.event",
            "lark_oapi.event.callback",
            "lark_oapi.event.callback.model",
            "lark_oapi.event.callback.model.p2_card_action_trigger",
            "lark_oapi.event.dispatcher_handler",
            "lark_oapi.ws",
        ):
            sys.modules.setdefault(name, mod)
    if "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from gateway.platforms.feishu import FeishuAdapter


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
    form_value: dict = None,
    action_name: str = None,
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
                form_value=form_value,
                name=action_name,
            ),
        ),
    )


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

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_exec_approval(
            chat_id="oc_12345", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_003"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            long_cmd = "x" * 5000
            await adapter.send_exec_approval(
                chat_id="oc_12345", command=long_cmd, session_key="s"
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        content = card["elements"][0]["content"]
        assert "..." in content
        assert len(content) < 5000

    @pytest.mark.asyncio
    async def test_multiple_approvals_get_unique_ids(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_x"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_1", command="cmd1", session_key="s1"
            )
            await adapter.send_exec_approval(
                chat_id="oc_2", command="cmd2", session_key="s2"
            )

        assert len(adapter._approval_state) == 2
        ids = list(adapter._approval_state.keys())
        assert ids[0] != ids[1]


# ===========================================================================
# _handle_card_action_event — approval button clicks
# ===========================================================================

class TestFeishuApprovalCallback:
    """Test the approval intercept in _handle_card_action_event."""

    @pytest.mark.asyncio
    async def test_resolves_approval_on_click(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_once", "approval_id": 1},
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_user1", "user_name": "Norbert", "user_id_alt": None},
            ),
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            result = await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("agent:main:feishu:group:oc_12345", "once")
        # Inline card response returned (no separate _update_approval_card call)
        assert result is not None
        assert getattr(result, "card", None) is not None

        # State should be cleaned up
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_deny_button(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = {
            "session_key": "some-session",
            "message_id": "msg_002",
            "chat_id": "oc_12345",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "deny", "approval_id": 2},
            token="tok_deny",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_alice", "user_name": "Alice", "user_id_alt": None},
            ),
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            result = await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("some-session", "deny")
        assert result is not None
        assert getattr(result, "card", None) is not None

    @pytest.mark.asyncio
    async def test_session_approval(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "msg_003",
            "chat_id": "oc_99",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_session", "approval_id": 3},
            token="tok_ses",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Bob", "user_id_alt": None},
            ),
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            result = await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("sess-3", "session")
        assert result is not None
        assert getattr(result, "card", None) is not None

    @pytest.mark.asyncio
    async def test_always_approval(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg_004",
            "chat_id": "oc_55",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_always", "approval_id": 4},
            token="tok_alw",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Carol", "user_id_alt": None},
            ),
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("sess-4", "always")

    @pytest.mark.asyncio
    async def test_already_resolved_drops_silently(self):
        adapter = _make_adapter()
        # No state for approval_id 99 — already resolved

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_once", "approval_id": 99},
            token="tok_gone",
        )

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_card_action_event(data)

        # Should NOT resolve — already handled
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_approval_actions_route_normally(self):
        """Non-approval card actions should still become synthetic commands."""
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
            patch("tools.approval.resolve_gateway_approval") as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        # Should NOT resolve any approval
        mock_resolve.assert_not_called()
        # Should have routed as synthetic command
        mock_handle.assert_called_once()
        event = mock_handle.call_args[0][0]
        assert "/card button" in event.text


# ===========================================================================
# _update_approval_card — card replacement after resolution
# ===========================================================================

class TestFeishuClarifyButtons:
    @pytest.mark.asyncio
    async def test_send_clarify_prompt_sends_interactive_card(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_clarify_1"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify_prompt(
                chat_id="oc_12345",
                question="你想测试哪一种？",
                choices=["多选按钮", "开放输入"],
                prompt_id="prompt-1",
            )
        assert result.success is True
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "blue"
        actions = []
        for element in card["elements"]:
            if element.get("tag") == "action":
                actions.extend(element["actions"])
        assert any(a["value"].get("hermes_clarify") for a in actions)
        assert any(a["value"].get("choice_text") == "多选按钮" for a in actions)

    @pytest.mark.asyncio
    async def test_clarify_button_resolves_waiter_and_updates_card(self):
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-42"] = {
            "event": evt,
            "response": None,
            "question": "继续做什么？",
            "message_id": "msg_clarify_42",
        }
        data = _make_card_action_data(
            action_value={
                "hermes_clarify": True,
                "prompt_id": "prompt-42",
                "choice_index": 1,
                "choice_text": "检查配置",
            },
            token="tok_clarify",
        )
        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Alice", "user_id_alt": None},
            ),
        ):
            result = await adapter._handle_card_action_event(data)
        assert evt.is_set()
        assert adapter._clarify_state["prompt-42"]["response"] == "检查配置"
        # Verify inline card response is returned
        assert result is not None
        assert getattr(result, "card", None) is not None
        assert result.card.data["header"]["template"] == "green"


# ===========================================================================
# Card V2 form building (ask-form)
# ===========================================================================

class TestFeishuAskFormCard:

    def test_build_ask_form_card_basic_structure(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "env",
                "question": "Which environment?",
                "options": [
                    {"label": "staging", "description": "Test env"},
                    {"label": "production"},
                ],
            }
        ]
        card = adapter._build_ask_form_card(questions, "prompt-123")
        assert card["schema"] == "2.0"
        assert "body" in card
        assert "elements" in card["body"]
        assert card["header"]["template"] == "blue"

    def test_build_ask_form_card_has_form_container(self):
        adapter = _make_adapter()
        questions = [{"header": "q1", "question": "What?"}]
        card = adapter._build_ask_form_card(questions, "prompt-1")
        forms = [e for e in card["body"]["elements"] if e.get("tag") == "form"]
        assert len(forms) == 1
        assert forms[0]["name"] == "hermes_ask_form"

    def test_build_ask_form_card_single_select_option(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "deploy_env",
                "question": "Deploy where?",
                "options": [
                    {"label": "staging", "recommended": True},
                    {"label": "production"},
                ],
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        selects = [e for e in form["elements"] if e.get("tag") == "select_static"]
        assert len(selects) == 1
        assert selects[0]["name"] == "deploy_env"
        assert selects[0]["initial_option"] == "staging"
        assert len(selects[0]["options"]) == 2

    def test_build_ask_form_card_multi_select_option(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "tags",
                "question": "Select tags",
                "multiSelect": True,
                "options": [
                    {"label": "bug"},
                    {"label": "feature"},
                ],
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        multi = [e for e in form["elements"] if e.get("tag") == "multi_select_static"]
        assert len(multi) == 1
        assert multi[0]["name"] == "tags"

    def test_build_ask_form_card_freeform_input(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "notes",
                "question": "Any notes?",
                "allowFreeformInput": True,
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        inputs = [e for e in form["elements"] if e.get("tag") == "input"]
        assert len(inputs) == 1
        assert inputs[0]["name"] == "notes"

    def test_build_ask_form_card_options_plus_freeform(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "choice",
                "question": "Pick or type",
                "options": [{"label": "A"}, {"label": "B"}],
                "allowFreeformInput": True,
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        selects = [e for e in form["elements"] if e.get("tag") == "select_static"]
        inputs = [e for e in form["elements"] if e.get("tag") == "input"]
        assert len(selects) == 1
        assert selects[0]["name"] == "choice"
        assert len(inputs) == 1
        assert inputs[0]["name"] == "choice_freeform"

    def test_build_ask_form_card_has_submit_button(self):
        adapter = _make_adapter()
        questions = [{"header": "q1", "question": "Q?"}]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        buttons = [e for e in form["elements"] if e.get("tag") == "button"]
        submit_buttons = [b for b in buttons if b.get("form_action_type") == "submit"]
        assert len(submit_buttons) >= 1
        assert submit_buttons[0]["behaviors"][0]["value"]["hermes_ask_form"] is True

    def test_build_ask_form_card_recommended_marker_in_label(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "env",
                "question": "Env?",
                "options": [
                    {"label": "staging", "recommended": True},
                    {"label": "prod"},
                ],
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        select = [e for e in form["elements"] if e.get("tag") == "select_static"][0]
        recommended_opt = select["options"][0]
        assert "⭐" in recommended_opt["text"]["content"]

    def test_build_ask_form_card_multiple_questions(self):
        adapter = _make_adapter()
        questions = [
            {"header": "q1", "question": "First?", "allowFreeformInput": True},
            {"header": "q2", "question": "Second?", "options": [{"label": "A"}]},
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        hrs = [e for e in form["elements"] if e.get("tag") == "hr"]
        assert len(hrs) >= 1

    def test_build_ask_form_card_option_description(self):
        adapter = _make_adapter()
        questions = [
            {
                "header": "env",
                "question": "Env?",
                "options": [
                    {"label": "staging", "description": "For testing"},
                ],
            }
        ]
        card = adapter._build_ask_form_card(questions, "p1")
        form = [e for e in card["body"]["elements"] if e.get("tag") == "form"][0]
        select = [e for e in form["elements"] if e.get("tag") == "select_static"][0]
        assert "For testing" in select["options"][0]["text"]["content"]


# ===========================================================================
# send_clarify_prompt dispatch (V1 vs V2)
# ===========================================================================

class TestFeishuAskFormSend:

    @pytest.mark.asyncio
    async def test_send_clarify_prompt_with_questions_sends_v2_card(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_form_1"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify_prompt(
                chat_id="oc_12345",
                question="",
                choices=None,
                prompt_id="prompt-form-1",
                questions=[
                    {"header": "env", "question": "Which env?", "options": [{"label": "staging"}]}
                ],
            )
        assert result.success is True
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        assert card.get("schema") == "2.0"
        assert "body" in card

    @pytest.mark.asyncio
    async def test_send_clarify_prompt_without_questions_uses_v1(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_v1"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify_prompt(
                chat_id="oc_12345",
                question="Pick one",
                choices=["A", "B"],
                prompt_id="prompt-v1",
            )
        assert result.success is True
        kwargs = mock_send.call_args[1]
        card = json.loads(kwargs["payload"])
        assert "schema" not in card
        assert "elements" in card


# ===========================================================================
# Card V2 form submit callback handling
# ===========================================================================

class TestFeishuAskFormCallback:

    @pytest.mark.asyncio
    async def test_form_submit_resolves_clarify_state(self):
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-form-1"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [
                {"header": "env", "question": "Which env?"},
            ],
            "message_id": "msg_form_1",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-form-1"},
            form_value={
                "env": "staging",
                "hermes_ask_submit": None,
            },
            action_name="hermes_ask_submit",
            token="tok_form_1",
        )
        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Alice", "user_id_alt": None},
            ),
        ):
            result = await adapter._handle_card_action_event(data)
        assert evt.is_set()
        response = adapter._clarify_state["prompt-form-1"]["response"]
        assert isinstance(response, dict)
        assert response["env"] == {"selected": ["staging"], "freeform": None}
        # Verify inline card response is returned
        assert result is not None
        assert getattr(result, "card", None) is not None
        assert result.card.data["header"]["template"] == "green"

    @pytest.mark.asyncio
    async def test_form_submit_with_multi_select(self):
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-ms"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [
                {"header": "tags", "question": "Tags?", "multiSelect": True},
            ],
            "message_id": "msg_ms",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-ms"},
            form_value={"tags": ["bug", "feature"]},
            token="tok_ms",
        )
        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Bob", "user_id_alt": None},
            ),
        ):
            await adapter._handle_card_action_event(data)
        assert evt.is_set()
        response = adapter._clarify_state["prompt-ms"]["response"]
        assert response["tags"] == {"selected": ["bug", "feature"], "freeform": None}

    @pytest.mark.asyncio
    async def test_form_submit_with_freeform(self):
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-ff"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [
                {
                    "header": "choice",
                    "question": "Pick or type",
                    "options": [{"label": "A"}],
                    "allowFreeformInput": True,
                },
            ],
            "message_id": "msg_ff",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-ff"},
            form_value={"choice": "A", "choice_freeform": "custom answer"},
            token="tok_ff",
        )
        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Carol", "user_id_alt": None},
            ),
        ):
            await adapter._handle_card_action_event(data)
        assert evt.is_set()
        response = adapter._clarify_state["prompt-ff"]["response"]
        assert response["choice"] == {"selected": ["A"], "freeform": "custom answer"}

    @pytest.mark.asyncio
    async def test_form_submit_unknown_prompt_id_ignored(self):
        adapter = _make_adapter()
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "nonexistent"},
            form_value={"q": "val"},
            token="tok_unknown",
        )
        await adapter._handle_card_action_event(data)

    @pytest.mark.asyncio
    async def test_form_submit_inline_card_uses_v2_schema(self):
        """Ask form submit inline response must use Card V2 schema (schema: '2.0', body.elements)."""
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-v2"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [
                {"header": "env", "question": "Which env?"},
            ],
            "message_id": "msg_v2",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-v2"},
            form_value={"env": "prod", "hermes_ask_submit": None},
            action_name="hermes_ask_submit",
            token="tok_v2_schema",
        )
        with patch.object(
            adapter, "_resolve_sender_profile", new_callable=AsyncMock,
            return_value={"user_id": "ou_u", "user_name": "Alice", "user_id_alt": None},
        ):
            result = await adapter._handle_card_action_event(data)
        card_data = result.card.data
        assert card_data.get("schema") == "2.0", "Ask form submit card must use V2 schema"
        assert "body" in card_data, "V2 card must have 'body' key"
        assert "elements" in card_data["body"], "V2 card body must have 'elements'"
        assert "config" not in card_data, "V2 card should not have 'config' (V1 field)"


class TestDuplicateTokenIdempotency:
    """Card action duplicate token returns cached inline response."""

    @pytest.mark.asyncio
    async def test_duplicate_token_returns_cached_response(self):
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-dup"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [{"header": "q1", "question": "Q?"}],
            "message_id": "msg_dup",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-dup"},
            form_value={"q1": "answer"},
            action_name="hermes_ask_submit",
            token="tok_dup_test",
        )
        with patch.object(
            adapter, "_resolve_sender_profile", new_callable=AsyncMock,
            return_value={"user_id": "ou_u", "user_name": "User", "user_id_alt": None},
        ):
            first_result = await adapter._handle_card_action_event(data)
        assert first_result is not None

        # Simulate _on_card_action_trigger caching the response
        adapter._card_action_responses["tok_dup_test"] = first_result

        # Second call with same token should return cached response
        second_result = await adapter._handle_card_action_event(data)
        assert second_result is first_result

    @pytest.mark.asyncio
    async def test_response_cached_after_first_handling(self):
        """After handler returns a response, caching it keyed by token enables idempotent replay."""
        adapter = _make_adapter()
        evt = threading.Event()
        adapter._clarify_state["prompt-cache"] = {
            "event": evt,
            "response": None,
            "question": "",
            "questions": [{"header": "q", "question": "Q?"}],
            "message_id": "msg_cache",
        }
        data = _make_card_action_data(
            action_value={"hermes_ask_form": True, "prompt_id": "prompt-cache"},
            form_value={"q": "val"},
            action_name="hermes_ask_submit",
            token="tok_cache_test",
        )
        with patch.object(
            adapter, "_resolve_sender_profile", new_callable=AsyncMock,
            return_value={"user_id": "ou_u", "user_name": "User", "user_id_alt": None},
        ):
            result = await adapter._handle_card_action_event(data)
        assert result is not None
        # Simulate _on_card_action_trigger caching
        adapter._card_action_responses["tok_cache_test"] = result
        assert adapter._card_action_responses["tok_cache_test"] is result

    def test_dedup_prunes_expired_responses(self):
        """Expired tokens should also clean up cached responses."""
        import time
        adapter = _make_adapter()
        adapter._card_action_tokens["old_tok"] = time.time() - 999
        adapter._card_action_responses["old_tok"] = MagicMock()
        # Calling _is_card_action_duplicate should prune the expired entry
        adapter._is_card_action_duplicate("new_tok")
        assert "old_tok" not in adapter._card_action_tokens
        assert "old_tok" not in adapter._card_action_responses


class TestChatMembersFallback:
    """Test _resolve_name_from_chat_members fallback."""

    @pytest.mark.asyncio
    async def test_resolves_name_from_chat_members(self):
        adapter = _make_adapter()
        member = SimpleNamespace(member_id="ou_target", name="张三")
        mock_resp = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(items=[member]),
        )
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            name = await adapter._resolve_name_from_chat_members("ou_target", "oc_chat1")
        assert name == "张三"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        adapter = _make_adapter()
        other = SimpleNamespace(member_id="ou_other", name="李四")
        mock_resp = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(items=[other]),
        )
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            name = await adapter._resolve_name_from_chat_members("ou_target", "oc_chat1")
        assert name is None

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self):
        adapter = _make_adapter()
        with patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("API fail")):
            name = await adapter._resolve_name_from_chat_members("ou_target", "oc_chat1")
        assert name is None
