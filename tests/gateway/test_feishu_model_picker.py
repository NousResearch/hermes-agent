"""Tests for Feishu interactive card model picker (send_model_picker).

Covers the two-level provider -> model drill-down, synchronous
P2CardActionTriggerResponse navigation (no subprocess for page turns),
the background model-switch execution + card patch, and the dispatch entry
point in ``_on_card_action_trigger``.
"""

import importlib.util
import json
import sys
import time
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
from plugins.platforms.feishu.adapter import FeishuAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PROVIDERS = [
    {
        "slug": "nvidia",
        "name": "NVIDIA",
        "models": ["deepseek-ai/deepseek-v4-flash", "z-ai/glm-5.2", "minimaxai/minimax-m3"],
        "total_models": 3,
        "is_current": True,
    },
    {
        "slug": "custom:huoshan",
        "name": "火山引擎",
        "models": ["deepseek-v4-flash-ga-260731"],
        "total_models": 1,
    },
]


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    # Authorize the test operator (default group policy is allowlist).
    adapter._allowed_group_users = {"ou_user1"}
    return adapter


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_abc",
) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


def _make_picker_event(open_id: str = "ou_user1", chat_id: str = "oc_12345") -> SimpleNamespace:
    """Build a minimal card-action event carrying operator identity for authz."""
    return SimpleNamespace(
        operator=SimpleNamespace(open_id=open_id, user_id="u_1"),
        context=SimpleNamespace(open_chat_id=chat_id),
    )


def _seed_picker_state(adapter, picker_id="mp_oc_12345", providers=None):
    adapter._model_picker_state[picker_id] = {
        "session_key": "s",
        "on_model_selected": AsyncMock(return_value="Model switched to X\nProvider: Y"),
        "current_model": "deepseek-ai/deepseek-v4-flash",
        "current_provider": "nvidia",
        "message_id": "om_card1",
        "chat_id": "oc_12345",
        "providers": providers if providers is not None else SAMPLE_PROVIDERS,
        "switching": False,
        "created_at": time.time(),
    }


def _close(coro, _loop):
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *a, **k: None)


# ===========================================================================
# send_model_picker — sends a provider-level interactive card
# ===========================================================================
class TestSendModelPicker:
    @pytest.mark.asyncio
    async def test_sends_provider_card_and_stores_state(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_card1"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_model_picker(
                chat_id="oc_12345",
                providers=SAMPLE_PROVIDERS,
                current_model="deepseek-ai/deepseek-v4-flash",
                current_provider="nvidia",
                session_key="agent:main:feishu:oc_12345",
                on_model_selected=AsyncMock(),
            )

        assert result.success is True
        assert result.message_id == "om_card1"
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "blue"
        # Provider buttons carry model_pick_provider actions
        actions = card["elements"][1]["actions"]
        assert len(actions) == 2
        assert all(a["value"]["hermes_action"] == "model_pick_provider" for a in actions)
        assert actions[0]["value"]["provider_slug"] == "nvidia"
        # Current provider is marked primary
        assert actions[0]["type"] == "primary"
        assert actions[1]["type"] == "default"

        # State keyed by picker_id = mp_<chat_id>_<token> (unguessable token)
        picker_keys = [k for k in adapter._model_picker_state if k.startswith("mp_oc_12345")]
        assert picker_keys, "no picker state stored"
        assert adapter._model_picker_state[picker_keys[0]]["message_id"] == "om_card1"

    @pytest.mark.asyncio
    async def test_returns_failure_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_model_picker(
            chat_id="oc_12345",
            providers=SAMPLE_PROVIDERS,
            current_model="m",
            current_provider="p",
            session_key="s",
            on_model_selected=AsyncMock(),
        )
        assert result.success is False


# ===========================================================================
# Card builders — structure contracts
# ===========================================================================
class TestCardBuilders:
    def test_provider_card_marks_current_provider(self):
        adapter = _make_adapter()
        card = adapter._build_provider_list_card(
            providers=SAMPLE_PROVIDERS,
            picker_id="mp_oc_12345",
            current_model="deepseek-ai/deepseek-v4-flash",
            current_provider="nvidia",
        )
        assert card["config"]["wide_screen_mode"] is True
        actions = card["elements"][1]["actions"]
        assert actions[0]["text"]["content"].startswith("✓")  # current marked
        assert actions[0]["type"] == "primary"
        assert actions[1]["text"]["content"] == "火山引擎 (1)"

    def test_model_card_paginates_at_8_per_page(self):
        adapter = _make_adapter()
        provider = {
            "slug": "nvidia",
            "name": "NVIDIA",
            "models": [f"org/model-{i}" for i in range(20)],
            "total_models": 20,
        }
        card = adapter._build_model_list_card(
            provider=provider,
            picker_id="mp_oc_12345",
            current_model="org/model-0",
            current_provider="nvidia",
            page=0,
        )
        model_actions = card["elements"][1]["actions"]
        assert len(model_actions) == 8
        # Header shows page info
        assert "(1/3)" in card["header"]["title"]["content"]
        # nav row: back + next
        nav = card["elements"][2]["actions"]
        nav_acts = [a["value"].get("hermes_action") for a in nav]
        assert "model_pick_back" in nav_acts
        assert "model_pick_page" in nav_acts

    def test_model_card_last_page_no_next_button(self):
        adapter = _make_adapter()
        provider = {"slug": "nvidia", "name": "NVIDIA",
                    "models": [f"org/m{i}" for i in range(20)], "total_models": 20}
        card = adapter._build_model_list_card(
            provider=provider, picker_id="mp", current_model="", current_provider="",
            page=2,
        )
        nav_acts = [a["value"].get("hermes_action") for a in card["elements"][2]["actions"]]
        # back always present; page nav only prev (no next) on last page
        page_navs = [v for v in nav_acts if v == "model_pick_page"]
        assert "model_pick_back" in nav_acts
        assert len(page_navs) == 1  # only "上一页", no "下一页"

    def test_switching_card_shows_target_model(self):
        adapter = _make_adapter()
        card = adapter._build_model_switching_card(model_id="org/m1", provider_slug="nvidia")
        assert "⏳" in card["header"]["title"]["content"]
        assert "org/m1" in card["elements"][0]["content"]


# ===========================================================================
# _handle_model_picker_card_action — synchronous dispatch
# ===========================================================================
class TestModelPickerDispatch:
    def test_provider_selected_returns_model_card(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        resp = adapter._handle_model_picker_card_action(
            event=_make_picker_event(),
            action_value={"hermes_action": "model_pick_provider",
                          "picker_id": "mp_oc_12345", "provider_slug": "nvidia"},
            loop=MagicMock(),
        )
        assert resp is not None
        card = resp.card
        assert card.type == "raw"
        assert card.data["header"]["title"]["content"] == "⚙️ NVIDIA"
        # model buttons
        acts = card.data["elements"][1]["actions"]
        assert [a["value"]["model_id"] for a in acts] == SAMPLE_PROVIDERS[0]["models"]

    def test_unknown_picker_id_returns_empty_response(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        resp = adapter._handle_model_picker_card_action(
            event=SimpleNamespace(),
            action_value={"hermes_action": "model_pick_provider",
                          "picker_id": "mp_missing", "provider_slug": "nvidia"},
            loop=MagicMock(),
        )
        assert resp is not None
        # No card attached (unknown state) — empty P2CardActionTriggerResponse
        assert getattr(resp, "card", None) is None

    def test_back_returns_provider_card(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        resp = adapter._handle_model_picker_card_action(
            event=_make_picker_event(),
            action_value={"hermes_action": "model_pick_back", "picker_id": "mp_oc_12345"},
            loop=MagicMock(),
        )
        card = resp.card
        acts = card.data["elements"][1]["actions"]
        assert all(a["value"]["hermes_action"] == "model_pick_provider" for a in acts)

    def test_page_navigation(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        # provider with 20 models -> 3 pages
        adapter._model_picker_state["mp_oc_12345"]["providers"] = [
            {"slug": "nvidia", "name": "NVIDIA",
             "models": [f"org/m{i}" for i in range(20)], "total_models": 20}
        ]
        resp = adapter._handle_model_picker_card_action(
            event=_make_picker_event(),
            action_value={"hermes_action": "model_pick_page", "picker_id": "mp_oc_12345",
                          "provider_slug": "nvidia", "page": 1},
            loop=MagicMock(),
        )
        card = resp.card
        assert "(2/3)" in card.data["header"]["title"]["content"]

    def test_model_selected_schedules_switch_and_returns_switching_card(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        # We must not actually run the scheduled coroutine synchronously;
        # assert it was scheduled and a "switching" card is returned.
        with patch.object(adapter, "_submit_on_loop", side_effect=_close) as mock_submit:
            resp = adapter._handle_model_picker_card_action(
                event=_make_picker_event(),
                action_value={"hermes_action": "model_pick_model",
                              "picker_id": "mp_oc_12345",
                              "model_id": "z-ai/glm-5.2", "provider_slug": "nvidia"},
                loop=MagicMock(),
            )
        mock_submit.assert_called_once()
        card = resp.card
        assert "⏳" in card.data["header"]["title"]["content"]
        assert "z-ai/glm-5.2" in card.data["elements"][0]["content"]


# ===========================================================================
# _execute_model_switch — background switch + final card patch
# ===========================================================================
class TestExecuteModelSwitch:
    @pytest.mark.asyncio
    async def test_success_patches_green_card_and_cleans_state(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        picker_id = "mp_oc_12345"
        state = adapter._model_picker_state[picker_id]

        with patch.object(adapter, "_patch_model_picker_card", new_callable=AsyncMock) as mock_patch:
            await adapter._execute_model_switch(
                picker_id=picker_id, model_id="z-ai/glm-5.2", provider_slug="nvidia", state=state,
            )

        # Final version patches the switching card first, then the final card.
        assert mock_patch.await_count == 2
        args = mock_patch.await_args
        assert args.args[0] == "oc_12345"
        # final card is the green confirmation
        final_card = args.args[1]
        assert final_card["header"]["template"] == "green"
        assert "已切换为" in final_card["header"]["title"]["content"]
        # message_id passed through so lookup-by-chat never misses
        assert args.kwargs.get("message_id") == "om_card1"
        # state cleaned up
        assert picker_id not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_callback_failure_patches_red_card(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        picker_id = "mp_oc_12345"
        state = adapter._model_picker_state[picker_id]
        state["on_model_selected"] = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(adapter, "_patch_model_picker_card", new_callable=AsyncMock) as mock_patch:
            await adapter._execute_model_switch(
                picker_id=picker_id, model_id="z-ai/glm-5.2", provider_slug="nvidia", state=state,
            )
        final_card = mock_patch.await_args.args[1]
        assert final_card["header"]["template"] == "red"
        assert "切换失败" in final_card["elements"][0]["content"]


# ===========================================================================
# _on_card_action_trigger — dispatch entry point
# ===========================================================================
class TestModelPickerTriggerEntry:
    def test_routes_model_pick_provider_to_sync_handler(self):
        adapter = _make_adapter()
        _seed_picker_state(adapter)
        data = _make_card_action_data(
            {"hermes_action": "model_pick_provider",
             "picker_id": "mp_oc_12345", "provider_slug": "nvidia"},
        )
        with patch.object(adapter, "_loop_accepts_callbacks", return_value=True), \
             patch.object(adapter, "_handle_model_picker_card_action",
                          wraps=adapter._handle_model_picker_card_action) as mock_handle:
            resp = adapter._on_card_action_trigger(data)
        mock_handle.assert_called_once()
        assert resp is not None
        assert resp.card.type == "raw"

    def test_non_picker_actions_delegate_to_normal_path(self):
        adapter = _make_adapter()
        data = _make_card_action_data({"hermes_action": "approve_once", "approval_id": 1})
        with patch.object(adapter, "_loop_accepts_callbacks", return_value=True), \
             patch.object(adapter, "_handle_approval_card_action", return_value=None) as mock_approve:
            adapter._on_card_action_trigger(data)
        mock_approve.assert_called_once()

    def test_unrelated_button_goes_to_synthetic_command_path(self):
        adapter = _make_adapter()
        data = _make_card_action_data({"custom_key": 1})
        with patch.object(adapter, "_loop_accepts_callbacks", return_value=True), \
             patch.object(adapter, "_submit_on_loop", return_value=True) as mock_submit:
            adapter._on_card_action_trigger(data)
        mock_submit.assert_called_once()
