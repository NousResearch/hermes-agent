"""Feishu approval / update-prompt card clicks with no Feishu allowlist or admins.

Setup's default leaves FEISHU_ALLOWED_USERS and the admin list empty (pairing mode). A click
must then pass the gateway authorization inbound messages get, and is refused when that check
denies or was never registered. A configured Feishu allowlist still decides on its own.
"""

import asyncio
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_feishu_mocks():
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in ("lark_oapi", "lark_oapi.api.im.v1", "lark_oapi.event", "lark_oapi.event.callback_type"):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture(autouse=True)
def _patch_callback_card_types(monkeypatch):
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


def _close_submitted_coro(coro, _loop):
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)


def _make_adapter(*, allowed=(), admins=()) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    adapter._loop = MagicMock()
    adapter._loop.is_closed = MagicMock(return_value=False)
    adapter._allowed_group_users = set(allowed)
    adapter._admins = set(admins)
    adapter._approval_state[1] = {"session_key": "sess-1", "message_id": "msg-1", "chat_id": "oc_group",
                                  "request_id": "req-1"}
    adapter._update_prompt_state[2] = {"session_key": "sess-2", "message_id": "msg-2", "chat_id": "oc_group"}
    return adapter


class _GatewayCheck:
    """Stand-in for the runner's ``set_authorization_check`` callback."""

    def __init__(self, allowed_ids):
        self.allowed_ids = set(allowed_ids)
        self.seen = []

    def __call__(self, user_id, chat_type=None, chat_id=None, **_kw):
        self.seen.append((user_id, chat_type, chat_id))
        return user_id in self.allowed_ids


_gateway_check = _GatewayCheck


def _approval_click(open_id, *, user_id=None, union_id=None, chat_id="oc_group"):
    return SimpleNamespace(event=SimpleNamespace(
        token="tok", context=SimpleNamespace(open_chat_id=chat_id),
        operator=SimpleNamespace(open_id=open_id, user_id=user_id, union_id=union_id),
        action=SimpleNamespace(tag="button", value={"hermes_action": "approve_once", "approval_id": 1}),
    ))


def _update_prompt_click(open_id):
    return SimpleNamespace(event=SimpleNamespace(
        token="tok", context=SimpleNamespace(open_chat_id="oc_group"),
        operator=SimpleNamespace(open_id=open_id),
        action=SimpleNamespace(tag="button", value={"hermes_update_prompt_action": "y", "update_prompt_id": 2}),
    ))


class TestNoFeishuAllowlist:
    def test_click_refused_when_gateway_denies(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(_gateway_check({"ou_requester"}))

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_bystander"))

        assert response.card is None
        submit.assert_not_called()
        assert 1 in adapter._approval_state

    def test_click_refused_without_gateway_check(self):
        adapter = _make_adapter()
        assert adapter._authorization_check is None

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_anyone"))

        assert response.card is None
        submit.assert_not_called()

    def test_update_prompt_refused_when_gateway_denies(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(_gateway_check(set()))

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_update_prompt_click("ou_bystander"))

        assert response.card is None
        submit.assert_not_called()
        assert 2 in adapter._update_prompt_state

    def test_gateway_authorized_user_can_approve(self):
        adapter = _make_adapter()
        check = _gateway_check({"ou_requester"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester"))

        assert response.card is not None
        submit.assert_called_once()
        assert check.seen == [("ou_requester", "group", "oc_group")]

    def test_gateway_check_uses_tenant_user_id_when_present(self):
        """The principal matches SessionSource.user_id: tenant user_id, else open_id."""
        adapter = _make_adapter()
        check = _gateway_check({"u_tenant"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(_approval_click("ou_x", user_id="u_tenant"))

        assert response.card is not None
        assert check.seen == [("u_tenant", "group", "oc_group")]

    @pytest.mark.asyncio
    async def test_loop_side_recheck_refuses_unauthorized(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(_gateway_check({"ou_requester"}))

        with patch("tools.approval.resolve_gateway_approval") as resolve:
            await adapter._resolve_approval(1, "once", "Mallory", open_id="ou_bystander", chat_id="oc_group")

        resolve.assert_not_called()
        assert 1 in adapter._approval_state

    @pytest.mark.asyncio
    async def test_loop_side_recheck_resolves_authorized(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(_gateway_check({"u_tenant"}))

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._resolve_approval(
                1, "once", "Alice", open_id="ou_x", chat_id="oc_group", user_id="u_tenant",
            )

        resolve.assert_called_once_with("sess-1", "once", request_id="req-1")

    @pytest.mark.asyncio
    async def test_non_approval_card_action_still_routes(self):
        adapter = _make_adapter()
        adapter._default_group_policy = "open"  # admitted by the chat; only the approval gate is absent
        data = SimpleNamespace(event=SimpleNamespace(
            token="tok_other", context=SimpleNamespace(open_chat_id="oc_group"),
            operator=SimpleNamespace(open_id="ou_bystander"),
            action=SimpleNamespace(tag="button", value={"custom_action": "x"}),
        ))
        with (
            patch.object(adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                         return_value={"user_id": "ou_bystander", "user_name": "B", "user_id_alt": None}),
            patch.object(adapter, "get_chat_info", new_callable=AsyncMock, return_value={"name": "Chat"}),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as handle,
        ):
            await adapter._handle_card_action_event(data)

        handle.assert_called_once()


class TestFeishuAllowlistUnchanged:
    def test_allowlisted_user_approves_without_gateway_check(self):
        adapter = _make_adapter(allowed={"ou_bob"})
        check = _gateway_check(set())
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(_approval_click("ou_bob"))

        assert response.card is not None
        assert check.seen == []

    def test_non_allowlisted_user_refused_even_if_gateway_allows(self):
        adapter = _make_adapter(admins={"ou_admin"})
        adapter.set_authorization_check(_gateway_check({"ou_other"}))

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_other"))

        assert response.card is None
        submit.assert_not_called()


class TestFeishuAllowlistMatchesEveryOperatorId:
    """A Feishu list may name a user by any id intake accepts (open_id / user_id / union_id)."""

    @pytest.mark.parametrize("click_ids", [{"user_id": "u_tenant"}, {"union_id": "on_union"}])
    def test_allowed_users_listed_by_other_id_can_approve(self, click_ids):
        adapter = _make_adapter(allowed={"u_tenant", "on_union"})

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_x", **click_ids))

        assert response.card is not None
        submit.assert_called_once()

    def test_admin_listed_by_tenant_user_id_can_approve(self):
        adapter = _make_adapter(admins={"u_admin"})

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_y", user_id="u_admin"))

        assert response.card is not None
        submit.assert_called_once()

    def test_unlisted_ids_still_refused(self):
        adapter = _make_adapter(allowed={"u_tenant"}, admins={"u_admin"})

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(
                _approval_click("ou_z", user_id="u_other", union_id="on_other"),
            )

        assert response.card is None
        submit.assert_not_called()
        assert 1 in adapter._approval_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize("click_ids", [{"user_id": "u_tenant"}, {"union_id": "on_union"}])
    async def test_loop_side_recheck_matches_other_ids(self, click_ids):
        adapter = _make_adapter(allowed={"u_tenant", "on_union"})

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._resolve_approval(1, "once", "Alice", open_id="ou_x", chat_id="oc_group", **click_ids)

        resolve.assert_called_once_with("sess-1", "once", request_id="req-1")

    @pytest.mark.asyncio
    async def test_update_prompt_recheck_matches_tenant_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter(allowed={"u_tenant"})

        await adapter._resolve_update_prompt(2, "y", "Alice", open_id="ou_x", chat_id="oc_group", user_id="u_tenant")

        assert 2 not in adapter._update_prompt_state


class TestCachedChatTypeResolution:
    """Card clicks resolve dm/group/forum from the cache's raw_mode (conversation mode), NOT
    raw_type (visibility private/public). A private-visibility group must follow group admission,
    not DM. Unknown chats fail closed to "group"."""

    def test_dm_cached_chat_type_passes_dm_admission(self):
        adapter = _make_adapter()
        adapter._chat_info_cache["oc_group"] = {"raw_type": "private", "raw_mode": "p2p"}
        seen = []

        def dm_only(user_id, chat_type=None, chat_id=None, **_kw):
            seen.append((user_id, chat_type, chat_id))
            return chat_type == "dm"

        adapter.set_authorization_check(dm_only)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester"))

        assert response.card is not None
        submit.assert_called_once()
        assert seen == [("ou_requester", "dm", "oc_group")]

    def test_update_prompt_dm_cached_chat_type(self):
        adapter = _make_adapter()
        adapter._chat_info_cache["oc_group"] = {"raw_type": "private", "raw_mode": "p2p"}
        seen = []

        def dm_only(user_id, chat_type=None, chat_id=None, **_kw):
            seen.append(chat_type)
            return chat_type == "dm"

        adapter.set_authorization_check(dm_only)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_update_prompt_click("ou_requester"))

        assert response.card is not None
        submit.assert_called_once()
        assert seen == ["dm"]

    def test_cache_miss_never_gets_dm_admission(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(lambda user_id, chat_type=None, chat_id=None, **_kw: chat_type == "dm")

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester"))

        assert response.card is None
        submit.assert_not_called()
        assert 1 in adapter._approval_state

    @pytest.mark.parametrize(("paired", "admitted"), [(False, False), (True, True)])
    def test_cache_miss_real_gateway_check_admits_only_paired(self, paired, admitted, tmp_path, monkeypatch):
        """Uncached chat through the runner's real check + PairingStore: unpaired is refused,
        a paired operator is still admitted. Pairing recorded the open_id while the click presents
        a tenant user_id too, so both the click gate and the loop-side recheck must try every id
        form before the approval settles."""
        from gateway.config import GatewayConfig, Platform
        from gateway.pairing import PairingStore
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for var in ("FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
                    "GATEWAY_ALLOW_ALL_USERS"):
            monkeypatch.delenv(var, raising=False)
        adapter = _make_adapter()
        store = PairingStore()
        if paired:
            assert store.approve_code("feishu", store.generate_code("feishu", "ou_requester", "R")) is not None
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.FEISHU: adapter.config})
        runner.adapters = {Platform.FEISHU: adapter}
        runner._profile_adapters = {}
        runner.pairing_store, runner.pairing_stores = store, {}
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.FEISHU))

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester", user_id="u_tenant"))

        assert (response.card is not None) is admitted
        assert submit.call_count == int(admitted)
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as settle:
            if admitted:
                asyncio.run(submit.call_args.args[0])
                settle.assert_called_once_with("sess-1", "once", request_id="req-1")
            else:
                settle.assert_not_called()
        assert (1 in adapter._approval_state) is not admitted

    @pytest.mark.parametrize("info", [{"raw_type": "public", "raw_mode": "group"}, {"raw_type": "private"}])
    def test_cached_group_or_missing_mode_stays_group(self, info):
        adapter = _make_adapter()
        adapter._chat_info_cache["oc_group"] = info
        check = _gateway_check({"ou_requester"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            adapter._on_card_action_trigger(_approval_click("ou_requester"))

        assert check.seen == [("ou_requester", "group", "oc_group")]

    def test_private_visibility_group_mode_maps_to_group_not_dm(self):
        """Counterexample: raw_type=private (visibility) + raw_mode=group (conversation mode)
        must be admitted as "group", NOT "dm" — private is about who can see the chat, not
        whether it is a 1:1 DM."""
        adapter = _make_adapter()
        adapter._chat_info_cache["oc_group"] = {"raw_type": "private", "raw_mode": "group"}
        seen = []

        def group_only(user_id, chat_type=None, chat_id=None, **_kw):
            seen.append(chat_type)
            return chat_type == "group"

        adapter.set_authorization_check(group_only)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester"))

        assert response.card is not None
        submit.assert_called_once()
        assert seen == ["group"]

    def test_private_visibility_topic_mode_maps_to_forum(self):
        """Counterexample: raw_type=private + raw_mode=topic must map to "forum", not "dm"."""
        adapter = _make_adapter()
        adapter._approval_state[1]["chat_id"] = "oc_topic"  # the card was posted in the topic chat
        adapter._chat_info_cache["oc_topic"] = {"raw_type": "private", "raw_mode": "topic"}
        seen = []

        def forum_only(user_id, chat_type=None, chat_id=None, **_kw):
            seen.append(chat_type)
            return chat_type == "forum"

        adapter.set_authorization_check(forum_only)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(_approval_click("ou_requester", chat_id="oc_topic"))

        assert response.card is not None
        submit.assert_called_once()
        assert seen == ["forum"]


class TestIdFormFallback:
    """Pairing records one Feishu id form while the event may present another; the gate must try
    every operator id form (open_id/user_id/union_id), like the allowlist branch already does."""

    def test_approved_open_id_passes_when_event_presents_tenant_user_id(self):
        adapter = _make_adapter()
        check = _gateway_check({"ou_requester"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(
                _approval_click("ou_requester", user_id="u_tenant"),
            )

        assert response.card is not None
        submit.assert_called_once()
        assert ("u_tenant", "group", "oc_group") in check.seen
        assert ("ou_requester", "group", "oc_group") in check.seen

    def test_approved_union_id_passes(self):
        adapter = _make_adapter()
        check = _gateway_check({"on_union"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as submit:
            response = adapter._on_card_action_trigger(
                _approval_click("ou_x", user_id="u_other", union_id="on_union"),
            )

        assert response.card is not None
        submit.assert_called_once()

    def test_no_matching_form_still_denied(self):
        adapter = _make_adapter()
        check = _gateway_check({"ou_someone_else"})
        adapter.set_authorization_check(check)

        with patch("asyncio.run_coroutine_threadsafe") as submit:
            response = adapter._on_card_action_trigger(
                _approval_click("ou_x", user_id="u_y", union_id="on_z"),
            )

        assert response.card is None
        submit.assert_not_called()
        assert 1 in adapter._approval_state
