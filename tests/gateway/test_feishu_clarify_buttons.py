"""Tests for Feishu interactive card clarify buttons.

Mirrors test_feishu_approval_buttons.py for the new ``send_clarify`` and
``hermes_clarify_action`` callback dispatch.
"""

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
import gateway.platforms.feishu as feishu_module
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


# ---------------------------------------------------------------------------
# Card rendering tests
# ---------------------------------------------------------------------------

class TestBuildClarifyCard:
    """Tests for _build_clarify_card static method."""

    def test_multi_choice_card(self):
        card = FeishuAdapter._build_clarify_card(
            question="Pick one", choices=["a", "b", "c"], clarify_id="id1",
        )
        assert card["header"]["template"] == "orange"
        actions = card["elements"][1]["actions"]
        assert len(actions) == 4  # 3 choices + Other
        assert actions[0]["value"]["hermes_clarify_action"] == "choice"
        assert actions[0]["value"]["choice_idx"] == 0
        assert actions[-1]["value"]["hermes_clarify_action"] == "other"

    def test_open_ended_card(self):
        card = FeishuAdapter._build_clarify_card(
            question="Type answer", choices=None, clarify_id="id2",
        )
        assert len(card["elements"]) == 1  # markdown only, no action

    def test_resolved_card(self):
        card = FeishuAdapter._build_resolved_clarify_card(
            choice_text="Apple", user_name="Alice",
        )
        assert card["header"]["template"] == "green"
        assert "Alice" in card["elements"][0]["content"]
        assert "Apple" in card["elements"][0]["content"]


# ---------------------------------------------------------------------------
# send_clarify integration tests
# ---------------------------------------------------------------------------

class TestSendClarify:
    """Tests for the send_clarify async method."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()
        adapter._feishu_send_with_retry = AsyncMock(return_value=MagicMock(
            success=lambda: True,
            data=MagicMock(message_id="msg_1"),
        ))
        result = await adapter.send_clarify(
            chat_id="oc_123", question="Pick", choices=["x", "y"],
            clarify_id="c1", session_key="s1",
        )
        assert result.success
        assert "c1" in adapter._clarify_state
        adapter._feishu_send_with_retry.assert_called_once()
        call_kwargs = adapter._feishu_send_with_retry.call_args[1]
        assert call_kwargs["msg_type"] == "interactive"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_clarify(
            chat_id="oc_123", question="Q", choices=None,
            clarify_id="c1", session_key="s1",
        )
        assert not result.success


# ---------------------------------------------------------------------------
# Card action callback tests
# ---------------------------------------------------------------------------

class TestClarifyCardAction:
    """Tests for _handle_clarify_card_action callback dispatch."""

    @pytest.fixture()
    def _patch_callback_card_types(self):
        """Patch P2CardActionTriggerResponse and CallBackCard for testing."""
        with patch.object(feishu_module, "P2CardActionTriggerResponse", create=True, new=MagicMock):
            with patch.object(feishu_module, "CallBackCard", create=True, new=MagicMock):
                yield

    def test_choice_action_resolves(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._clarify_state["c1"] = {
            "session_key": "s1", "message_id": "m1",
            "chat_id": "oc_123", "choices": ["Apple", "Banana"],
        }
        adapter._submit_on_loop = MagicMock(return_value=True)
        adapter._is_interactive_operator_authorized = MagicMock(return_value=True)
        adapter._get_cached_sender_name = MagicMock(return_value="Alice")

        data = _make_card_action_data({
            "hermes_clarify_action": "choice",
            "clarify_id": "c1",
            "choice_idx": 0,
        })
        event = data.event
        loop = MagicMock()

        with patch("tools.clarify_gateway.resolve_gateway_clarify", create=True) as mock_resolve:
            mock_resolve.return_value = True
            # Import after patching
            import tools.clarify_gateway
            tools.clarify_gateway.resolve_gateway_clarify = mock_resolve

            adapter._handle_clarify_card_action(
                event=event, action_value=data.event.action.value, loop=loop,
            )

        assert "c1" not in adapter._clarify_state  # state popped

    def test_other_action_calls_mark_awaiting_text(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._clarify_state["c2"] = {
            "session_key": "s1", "message_id": "m1",
            "chat_id": "oc_123", "choices": ["A"],
        }
        adapter._is_interactive_operator_authorized = MagicMock(return_value=True)
        adapter._get_cached_sender_name = MagicMock(return_value="Bob")

        data = _make_card_action_data({
            "hermes_clarify_action": "other",
            "clarify_id": "c2",
        })
        event = data.event
        loop = MagicMock()

        with patch("tools.clarify_gateway.mark_awaiting_text", create=True) as mock_mark:
            import tools.clarify_gateway
            tools.clarify_gateway.mark_awaiting_text = mock_mark

            adapter._handle_clarify_card_action(
                event=event, action_value=data.event.action.value, loop=loop,
            )

            mock_mark.assert_called_once_with("c2")

    def test_unauthorized_user_rejected(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._is_interactive_operator_authorized = MagicMock(return_value=False)

        data = _make_card_action_data({
            "hermes_clarify_action": "choice",
            "clarify_id": "c1",
            "choice_idx": 0,
        })
        event = data.event
        loop = MagicMock()

        result = adapter._handle_clarify_card_action(
            event=event, action_value=data.event.action.value, loop=loop,
        )
        # Should return early without resolving
        assert "c1" not in adapter._clarify_state

    def test_fallback_when_entries_missing(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._clarify_state["c3"] = {
            "session_key": "s1", "message_id": "m1",
            "chat_id": "oc_123", "choices": None,  # no choices stored in state
        }
        adapter._submit_on_loop = MagicMock(return_value=True)
        adapter._is_interactive_operator_authorized = MagicMock(return_value=True)
        adapter._get_cached_sender_name = MagicMock(return_value="Alice")

        data = _make_card_action_data({
            "hermes_clarify_action": "choice",
            "clarify_id": "c3",
            "choice_idx": 2,
        })
        event = data.event
        loop = MagicMock()

        # State has entry but no choices; _entries also empty — should use fallback
        with patch.dict("sys.modules", {"tools.clarify_gateway": MagicMock(
            resolve_gateway_clarify=MagicMock(return_value=True),
            _entries={},
        )}):
            adapter._handle_clarify_card_action(
                event=event, action_value=data.event.action.value, loop=loop,
            )

        # Should have scheduled resolution with fallback text "choice 3"
        adapter._submit_on_loop.assert_called_once()

    def test_already_resolved_returns_early(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._is_interactive_operator_authorized = MagicMock(return_value=True)

        data = _make_card_action_data({
            "hermes_clarify_action": "choice",
            "clarify_id": "c_gone",
            "choice_idx": 0,
        })
        event = data.event
        loop = MagicMock()

        # No _clarify_state entry — should return early
        result = adapter._handle_clarify_card_action(
            event=event, action_value=data.event.action.value, loop=loop,
        )
        # Should not have scheduled any resolution
        assert result is not None
