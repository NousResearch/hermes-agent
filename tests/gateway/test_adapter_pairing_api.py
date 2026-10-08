"""Adapter pairing API: ``request_pairing`` issues the code an unauthorized DM would get without a chat
reply, and the gateway's pairing watch reports approvals/revocations through ``on_pairing_changed``."""

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import CODE_TTL_SECONDS, MAX_PENDING_PER_PLATFORM, PairingStore
from gateway.platforms.base import BasePlatformAdapter, PairingOffer, SendResult
from gateway.session import SessionSource

_AUTH_ENV = (
    "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS", "SIGNAL_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
)


class _ScreenAdapter(BasePlatformAdapter):
    """A screen-first adapter: shows codes itself and wants approval news."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.SIGNAL)
        self.sent, self.changes = [], []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def on_pairing_changed(self, user_id, approved):
        self.changes.append((user_id, approved))


class _ChatAdapter(_ScreenAdapter):
    """Keeps the base ``on_pairing_changed``: never watched."""

    on_pairing_changed = BasePlatformAdapter.on_pairing_changed


@pytest.fixture(autouse=True)
def _no_allowlists(monkeypatch):
    for key in _AUTH_ENV:
        monkeypatch.delenv(key, raising=False)


def _store(path) -> PairingStore:
    with patch("gateway.pairing.PAIRING_DIR", path):
        return PairingStore()


def _runner(adapter, store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SIGNAL: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.SIGNAL: adapter}
    runner.pairing_store = store
    runner.pairing_stores = {}
    return runner


def _dm(user_id="+15550000001", *, chat_type="dm", is_bot=False):
    return SessionSource(platform=Platform.SIGNAL, chat_id=user_id, chat_type=chat_type,
                         user_id=user_id, user_name="kitchen", is_bot=is_bot)


def _wired(tmp_path):
    adapter, store = _ScreenAdapter(), _store(tmp_path / "pairing")
    runner = _runner(adapter, store)
    adapter.set_pairing_requester(runner._make_pairing_requester())
    return adapter, store, runner


@pytest.mark.asyncio
async def test_request_pairing_returns_a_code_the_owner_can_approve(tmp_path):
    adapter, store, runner = _wired(tmp_path)

    offer = await adapter.request_pairing(_dm())

    assert isinstance(offer, PairingOffer)
    assert offer.expires_in == CODE_TTL_SECONDS
    assert offer.command.endswith(f"pairing approve signal {offer.code}")
    assert adapter.sent == []
    assert store.approve_code("signal", offer.code)["user_id"] == "+15550000001"
    assert runner._is_user_authorized(_dm()) is True


@pytest.mark.asyncio
async def test_a_second_request_inside_the_rate_limit_window_gets_nothing(tmp_path):
    adapter, store, _runner_ = _wired(tmp_path)

    assert await adapter.request_pairing(_dm()) is not None
    assert await adapter.request_pairing(_dm()) is None
    assert len(store.list_pending("signal")) == 1


@pytest.mark.asyncio
async def test_an_authorized_sender_gets_no_code(tmp_path):
    adapter, store, _runner_ = _wired(tmp_path)
    store.approve_code("signal", store.generate_code("signal", "+15550000001", "kitchen"))

    assert await adapter.request_pairing(_dm()) is None


@pytest.mark.asyncio
async def test_no_code_where_unknown_senders_are_not_paired(tmp_path, monkeypatch):
    """A configured allowlist switches unknown DMs to "ignore": no code is minted either way."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15559999999")
    adapter, store, _runner_ = _wired(tmp_path)

    assert await adapter.request_pairing(_dm()) is None
    assert store.list_pending("signal") == []


@pytest.mark.asyncio
async def test_the_pending_code_cap_applies(tmp_path):
    adapter, _store_, _runner_ = _wired(tmp_path)
    offers = [await adapter.request_pairing(_dm(f"+1555000010{i}")) for i in range(MAX_PENDING_PER_PLATFORM + 1)]

    assert all(offers[:MAX_PENDING_PER_PLATFORM])
    assert offers[MAX_PENDING_PER_PLATFORM] is None


@pytest.mark.asyncio
async def test_only_human_direct_messages_get_a_code(tmp_path):
    adapter, store, _runner_ = _wired(tmp_path)

    assert await adapter.request_pairing(_dm(chat_type="group")) is None
    assert await adapter.request_pairing(_dm(is_bot=True)) is None
    assert store.list_pending("signal") == []


@pytest.mark.asyncio
async def test_without_a_gateway_there_is_no_code():
    assert await _ScreenAdapter().request_pairing(_dm()) is None


@pytest.mark.asyncio
async def test_each_profile_issues_codes_from_its_own_store(tmp_path, monkeypatch):
    """Primary → secondary → primary: every code lands in the store of the profile that owns the bot."""
    alpha_home = tmp_path / "profiles" / "alpha"
    alpha_home.mkdir(parents=True)
    (alpha_home / ".env").write_text("")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: alpha_home)
    primary, alpha = _ScreenAdapter(), _ScreenAdapter()
    default_store, alpha_store = _store(tmp_path / "default-pairing"), _store(alpha_home / "pairing")
    runner = _runner(primary, default_store)
    runner.pairing_stores = {"alpha": alpha_store}
    runner._profile_adapters = {"alpha": {Platform.SIGNAL: alpha}}
    primary.set_pairing_requester(runner._make_pairing_requester())
    alpha.set_pairing_requester(runner._make_pairing_requester("alpha"))

    first = await primary.request_pairing(_dm("+15550000001"))
    second = await alpha.request_pairing(_dm("+15550000002"))
    third = await primary.request_pairing(_dm("+15550000003"))

    assert default_store.approve_code("signal", first.code)["user_id"] == "+15550000001"
    assert alpha_store.approve_code("signal", second.code)["user_id"] == "+15550000002"
    assert default_store.approve_code("signal", third.code)["user_id"] == "+15550000003"
    assert alpha_store.approve_code("signal", first.code) is None


@pytest.mark.asyncio
async def test_watch_reports_approvals_and_revocations_after_a_baseline(tmp_path):
    adapter, store = _ScreenAdapter(), _store(tmp_path / "pairing")
    store.approve_code("signal", store.generate_code("signal", "+15550000001", "earlier"))
    runner, seen = _runner(adapter, store), {}

    await runner._pairing_watch_tick(seen)
    store.approve_code("signal", store.generate_code("signal", "+15550000002", "kitchen"))
    await runner._pairing_watch_tick(seen)
    store.revoke("signal", "+15550000001")
    await runner._pairing_watch_tick(seen)
    await runner._pairing_watch_tick(seen)

    assert adapter.changes == [("+15550000002", True), ("+15550000001", False)]


@pytest.mark.asyncio
async def test_watch_reads_nothing_for_adapters_without_the_hook():
    store = MagicMock()
    store.approved_user_ids.side_effect = AssertionError("store read for an unwatched adapter")
    runner = _runner(_ChatAdapter(), store)

    await runner._pairing_watch_tick({})
    await runner._pairing_watch_tick({})


@pytest.mark.asyncio
async def test_a_failing_hook_does_not_stop_the_watch(tmp_path):
    class _BrokenAdapter(_ScreenAdapter):
        async def on_pairing_changed(self, user_id, approved):
            raise RuntimeError("display offline")

    adapter, store = _BrokenAdapter(), _store(tmp_path / "pairing")
    runner, seen = _runner(adapter, store), {}
    await runner._pairing_watch_tick(seen)
    store.approve_code("signal", store.generate_code("signal", "+15550000002", "kitchen"))

    await runner._pairing_watch_tick(seen)

    assert seen[(None, "signal")] == {"+15550000002"}


@pytest.mark.asyncio
async def test_watch_reports_to_the_adapter_whose_profile_store_changed(tmp_path, monkeypatch):
    alpha_home = tmp_path / "profiles" / "alpha"
    alpha_home.mkdir(parents=True)
    (alpha_home / ".env").write_text("")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: alpha_home)
    primary, alpha = _ScreenAdapter(), _ScreenAdapter()
    default_store, alpha_store = _store(tmp_path / "default-pairing"), _store(alpha_home / "pairing")
    runner, seen = _runner(primary, default_store), {}
    runner.pairing_stores = {"alpha": alpha_store}
    runner._profile_adapters = {"alpha": {Platform.SIGNAL: alpha}}
    await runner._pairing_watch_tick(seen)

    alpha_store.approve_code("signal", alpha_store.generate_code("signal", "+15550000002", "kitchen"))
    await runner._pairing_watch_tick(seen)

    assert alpha.changes == [("+15550000002", True)]
    assert primary.changes == []
