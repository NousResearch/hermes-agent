"""Kanban approvals require an explicit, singular, currently authorized transport."""

import asyncio
import socket
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _Telegram:
    def __init__(self, *, authorized=True, connected=True, send_success=True):
        self.authorized = authorized
        self._bot = object() if connected else None
        self._running = connected
        self.auth_calls = []
        self.sent = []
        self.send_success = send_success
        self.send_attempted = asyncio.Event()

    @property
    def is_connected(self):
        return self._running

    @classmethod
    def supports_exec_approval_buttons(cls):
        return True

    def _is_callback_user_authorized(self, user_id, **kwargs):
        self.auth_calls.append((user_id, kwargs))
        return self.authorized

    async def send_exec_approval(self, **kwargs):
        self.sent.append(kwargs)
        self.send_attempted.set()
        return SimpleNamespace(success=self.send_success, error=None if self.send_success else "not connected")


class _Runner:
    def __init__(self, adapter):
        self.adapter = adapter
        self.config = type("Config", (), {"multiplex_profiles": False})()

    def _authorization_adapter(self, platform, profile):
        return self.adapter if platform.value == "telegram" and profile == "default" else None


def _subscription(**changes):
    from gateway.kanban_approval_broker import APPROVAL_FORWARDING_METADATA_KEY

    sub = {
        "task_id": "t_worker",
        "platform": "telegram",
        "chat_id": "100",
        "thread_id": "7",
        "chat_type": "forum",
        "user_id": "42",
        "notifier_profile": "default",
        "delivery_metadata": {APPROVAL_FORWARDING_METADATA_KEY: "v1"},
    }
    sub.update(changes)
    return sub


def test_finite_scope_requires_explicit_broker_marker():
    from gateway.session_finite import finite_turn_required, finite_turn_scope
    from tools.approval_context import _is_single_query_approval_context

    with finite_turn_scope(True, approval_broker=True):
        assert finite_turn_required() is True  # clarify remains headless
        assert _is_single_query_approval_context() is False
    with finite_turn_scope(True):
        assert finite_turn_required() is True
        assert _is_single_query_approval_context() is True


def test_plain_task_subscription_does_not_enable_approvals():
    from gateway.kanban_approval_broker import select_approval_subscription

    adapter = _Telegram()
    sub = _subscription(delivery_metadata={})
    assert select_approval_subscription(_Runner(adapter), [sub]) is None
    assert adapter.auth_calls == []


def test_exactly_one_authorized_telegram_subscriber_enables_approvals():
    from gateway.kanban_approval_broker import select_approval_subscription

    adapter = _Telegram()
    sub = _subscription()
    selected = select_approval_subscription(_Runner(adapter), [sub])

    assert selected == (adapter, sub)
    assert adapter.auth_calls == [("42", {"chat_id": "100", "chat_type": "forum", "thread_id": "7", "user_name": None})]


def test_multiplex_authorization_uses_notifier_profile_scope(monkeypatch):
    from contextlib import contextmanager
    from gateway.config import Platform
    from gateway.kanban_approval_broker import select_approval_subscription

    adapter = _Telegram()
    runner = _Runner(adapter)
    runner.config.multiplex_profiles = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._active_profile_name = lambda: "default"
    runner._resolve_profile_home_for_source = lambda source: "/notifier/home"
    entered = []

    @contextmanager
    def scope(home):
        entered.append(home)
        yield

    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_profile_runtime_scope=scope))
    sub = _subscription()
    assert select_approval_subscription(runner, [sub]) == (adapter, sub)
    assert entered == ["/notifier/home"]


def test_inherited_subscription_does_not_delegate_approval_to_child_task(tmp_path):
    import contextlib
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_notify import add_notify_sub
    from hermes_cli.kanban_db_connect import connect
    from gateway.kanban_approval_broker import APPROVAL_FORWARDING_METADATA_KEY, approval_subscription_for_task

    db = tmp_path / "board.db"
    kb.init_db(db)
    with contextlib.closing(connect(db)) as conn:
        parent = kb.create_task(conn, title="parent", assignee="implementer")
        add_notify_sub(
            conn, task_id=parent, platform="telegram", chat_id="100", user_id="42", notifier_profile="default",
            delivery_metadata={APPROVAL_FORWARDING_METADATA_KEY: "v1", "kanban_approval_task_id": parent})
        child = kb.create_task(conn, title="child", assignee="implementer", parents=[parent])

    runner = _Runner(_Telegram())
    assert approval_subscription_for_task(runner, db, child) is None
    assert approval_subscription_for_task(runner, db, parent) is not None


def test_unauthorized_disconnected_or_ambiguous_subscribers_fail_closed():
    from gateway.kanban_approval_broker import select_approval_subscription

    for adapter, subs in (
        (_Telegram(authorized=False), [_subscription()]),
        (_Telegram(connected=False), [_subscription()]),
        (_Telegram(), [_subscription(), _subscription(chat_id="other")]),
        (_Telegram(), [_subscription(notifier_profile=None)]),
        (_Telegram(), [_subscription(user_id=None)]),
    ):
        assert select_approval_subscription(_Runner(adapter), subs) is None


def _running_owner(tmp_path, monkeypatch, adapter):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_notify as kbn
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    monkeypatch.setitem(sys.modules, "hermes_bootstrap",
                        SimpleNamespace(_happy_eyeballs_create_connection=socket.create_connection))
    board_db = tmp_path / "board.db"
    with connect_closing(board_db) as conn:
        task_id = kb.create_task(conn, title="worker", assignee="implementer")
        kbn.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="100", thread_id="7", chat_type="thread",
            user_id="42", notifier_profile="default",
            delivery_metadata={"kanban_approval_forwarding": "v1", "kanban_approval_task_id": task_id})
    db = SessionDB(tmp_path / "state.db")
    db.create_session("worker-session", source="kanban")
    epoch = begin_runtime_epoch(db, instance_id="owner")
    card = {"board": "default", "task_id": task_id, "run_id": 1, "db": str(board_db)}
    admit_session_input(
        db, epoch=epoch, principal_id="dispatcher", session_id="worker-session", request_id="kanban:test",
        payload={"text": "work", "finite": True, "kanban_card_v1": card, "kanban_approval_broker_v1": True})
    row = claim_session_input(db, epoch=epoch, session_id="worker-session")
    runner = _Runner(adapter)
    runner._draining = False
    authority = SessionAuthority(runner, profile_id="worker-profile", instance_id="owner", db=db, epoch=epoch)
    live = authority.sessions["worker-session"] = LiveSession(None, "worker-route")
    live.event_stream.execution = {"admission_id": row["admission_id"], "execution_generation": row["generation"]}
    authority.executing["worker-session"] = row["generation"]
    responses = []
    live.controls.remote_responders["request-1"] = lambda kind, prompt_id, value: responses.append((kind, prompt_id, value))
    ref = SessionRef(authority.profile_id, "worker-session")
    return SimpleNamespace(authority=authority, row=row, ref=ref, responses=responses, db=db, adapter=adapter)


@pytest.mark.asyncio
async def test_forwarded_prompt_binds_exact_worker_and_subscriber(tmp_path, monkeypatch):
    adapter = _Telegram()
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    from gateway.kanban_approval_broker import FORWARDING_METADATA_KEY, respond_forwarded

    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write",
             "allow_session": True, "allow_permanent": True}, timeout_s=30)
        await asyncio.wait_for(adapter.send_attempted.wait(), 2)
        sent = adapter.sent[0]
        assert sent["chat_id"] == "100"
        assert sent["session_key"] == "worker-route"
        assert sent["request_id"] == "request-1"
        assert sent["allow_session"] is False and sent["allow_permanent"] is False
        assert sent["metadata"]["thread_id"] == "7"
        binding = sent["metadata"][FORWARDING_METADATA_KEY]
        from hermes_constants import get_process_hermes_home
        assert binding.notifier_profile == "default"
        assert binding.notifier_home == str(get_process_hermes_home())

        common = {"user_id": "42", "chat_id": "100", "thread_id": "7"}
        assert await respond_forwarded(binding, "once", **{**common, "user_id": "intruder"}) == 0
        assert await respond_forwarded(binding, "once", **{**common, "chat_id": "intruder"}) == 0
        assert await respond_forwarded(binding, "once", **{**common, "thread_id": "intruder"}) == 0
        assert owner.responses == []

        assert await respond_forwarded(binding, "once", **common) == 1
        assert owner.responses == [("approval", "request-1", "once")]
        assert await respond_forwarded(binding, "once", **common) == 0
        assert owner.responses == [("approval", "request-1", "once")]
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_in_process_waiter_receives_forwarded_choice(tmp_path, monkeypatch):
    adapter = _Telegram()
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    from gateway.kanban_approval_broker import FORWARDING_METADATA_KEY, respond_forwarded
    from tools.approval_gateway_wait import _await_gateway_decision

    owner.authority.sessions["worker-session"].controls.remote_responders.clear()
    owner.authority.sessions["worker-session"].task = asyncio.current_task()

    def notify(data):
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route", data, timeout_s=30)

    try:
        waiting = asyncio.create_task(asyncio.to_thread(
            _await_gateway_decision, "worker-route", notify,
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write",
             "allow_session": False, "allow_permanent": False}))
        await asyncio.wait_for(adapter.send_attempted.wait(), 2)
        binding = adapter.sent[0]["metadata"][FORWARDING_METADATA_KEY]
        assert await respond_forwarded(binding, "once", user_id="42", chat_id="100", thread_id="7") == 1
        decision = await asyncio.wait_for(waiting, 2)
        assert decision["resolved"] is True and decision["choice"] == "once"
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_stale_worker_response_never_consumes_a_later_prompt(tmp_path, monkeypatch):
    adapter = _Telegram()
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    from gateway.kanban_approval_broker import FORWARDING_METADATA_KEY, respond_forwarded
    from hermes_state_runtime import settle_session_input

    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
            timeout_s=30)
        await asyncio.wait_for(adapter.send_attempted.wait(), 2)
        binding = adapter.sent[0]["metadata"][FORWARDING_METADATA_KEY]
        settle_session_input(
            owner.db, epoch=owner.authority.epoch, admission_id=owner.row["admission_id"],
            generation=owner.row["generation"], outcome="completed")

        assert await respond_forwarded(binding, "once", user_id="42", chat_id="100", thread_id="7") == 0
        assert owner.responses == []
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_worker_loss_during_withdrawal_cleans_owner_projection(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError

    adapter = _Telegram(send_success=False)
    owner = _running_owner(tmp_path, monkeypatch, adapter)

    def lost(*args):
        raise RuntimeStoreError("managed_worker_lost")

    owner.authority.sessions["worker-session"].controls.remote_responders["request-1"] = lost

    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
            timeout_s=30)
        tasks = list(getattr(owner.authority, "_kanban_approval_tasks", ()))
        assert tasks
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert owner.authority.sessions["worker-session"].controls.pending == {}
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_delivery_failure_withdraws_exact_prompt_instead_of_timing_out(tmp_path, monkeypatch):
    adapter = _Telegram(send_success=False)
    owner = _running_owner(tmp_path, monkeypatch, adapter)

    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
            timeout_s=30)
        await asyncio.wait_for(adapter.send_attempted.wait(), 2)
        tasks = list(getattr(owner.authority, "_kanban_approval_tasks", ()))
        assert tasks
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert owner.responses == [("approval", "request-1", "cancelled:authorized Telegram approval delivery failed")]
        assert owner.authority.sessions["worker-session"].controls.pending == {}
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_real_telegram_adapter_round_trip_returns_choice_to_worker(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    bot = AsyncMock()
    adapter._bot = bot
    adapter._app = MagicMock()
    adapter._running = True
    send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    bot.send_message = send_message
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    owner = _running_owner(tmp_path, monkeypatch, adapter)

    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
            timeout_s=30)
        tasks = list(getattr(owner.authority, "_kanban_approval_tasks", ()))
        assert tasks
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert send_message.await_count == 1, owner.responses
        assert owner.responses == []

        approval_id = next(iter(adapter._approval_forwarded))
        query = MagicMock()
        query.data = f"ea:once:{approval_id}"
        query.from_user = SimpleNamespace(id=42, first_name="John")
        query.message.chat_id = 100
        query.message.message_thread_id = 7
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())

        assert owner.responses == [("approval", "request-1", "once")]
    finally:
        owner.db.close()


def _cross_profile_real_owner(tmp_path, monkeypatch, *, worker_users=""):
    """Real adapter/owner with distinct notifier and worker secret scopes."""
    monkeypatch.setitem(sys.modules, "hermes_bootstrap",
                        SimpleNamespace(_happy_eyeballs_create_connection=socket.create_connection))
    from agent import secret_scope as ss
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    notifier_home = tmp_path / "default-home"
    worker_home = tmp_path / "implementer-home"
    notifier_home.mkdir()
    worker_home.mkdir()
    (notifier_home / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=42\nGATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")
    (worker_home / ".env").write_text(
        f"TELEGRAM_ALLOWED_USERS={worker_users}\nGATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._running = True
    bot = adapter._bot
    assert bot is not None
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    # Exercise the adapter's scoped fallback as well as the real owner and callback path.
    adapter._authorization_check = None
    adapter._message_handler = None
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    runner = owner.authority.runner
    runner.config.multiplex_profiles = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._active_profile_name = lambda: "default"
    runner._resolve_profile_home_for_source = lambda source: notifier_home
    return SimpleNamespace(owner=owner, adapter=adapter, bot=bot,
                           notifier_home=notifier_home, worker_home=worker_home)


async def _deny_forwarded(rig):
    from gateway.run import _profile_runtime_scope

    owner, adapter = rig.owner, rig.adapter
    with _profile_runtime_scope(rig.worker_home, hydrate_secrets=False):
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
            timeout_s=30)
        await asyncio.wait_for(asyncio.gather(*owner.authority._kanban_approval_tasks), 2)
        assert rig.bot.send_message.await_count == 1, owner.responses
        approval_id = next(iter(adapter._approval_forwarded))
        query = MagicMock()
        query.data = f"ea:deny:{approval_id}"
        query.from_user = SimpleNamespace(id=42, first_name="John")
        query.message.chat_id = 100
        query.message.message_thread_id = 7
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())
    return approval_id, query


@pytest.mark.asyncio
async def test_forwarded_deny_rechecks_notifier_allowlist_not_worker_scope(tmp_path, monkeypatch):
    """A real owner/adapter callback must not inherit the worker's empty allowlist."""
    rig = _cross_profile_real_owner(tmp_path, monkeypatch)
    try:
        _approval_id, query = await _deny_forwarded(rig)
        assert rig.owner.responses == [("approval", "request-1", "deny")]
        assert query.answer.await_args_list[-1].kwargs["text"]
    finally:
        rig.owner.db.close()


@pytest.mark.asyncio
async def test_forwarded_callback_honors_notifier_allowlist_revocation(tmp_path, monkeypatch):
    """The worker's still-valid grant must not override notifier revocation after card delivery."""
    rig = _cross_profile_real_owner(tmp_path, monkeypatch, worker_users="42")
    try:
        # Post first, revoke the notifier grant, then tap under the worker's still-authorized scope.
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(rig.worker_home, hydrate_secrets=False):
            rig.owner.authority.register_approval(
                "worker-session", rig.owner.row["generation"], "worker-route",
                {"request_id": "request-1", "command": "edit AGENTS.md", "description": "protected write"},
                timeout_s=30)
            await asyncio.wait_for(asyncio.gather(*rig.owner.authority._kanban_approval_tasks), 2)
        (rig.notifier_home / ".env").write_text("GATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")

        approval_id = next(iter(rig.adapter._approval_forwarded))
        query = MagicMock()
        query.data = f"ea:deny:{approval_id}"
        query.from_user = SimpleNamespace(id=42, first_name="John")
        query.message.chat_id = 100
        query.message.message_thread_id = 7
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        with _profile_runtime_scope(rig.worker_home, hydrate_secrets=False):
            await rig.adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())

        assert rig.owner.responses == []
        assert "request-1" in rig.owner.authority.sessions["worker-session"].controls.pending
        assert approval_id in rig.adapter._approval_state
    finally:
        rig.owner.db.close()


async def _persisted_default_dm_forwarded(rig, tmp_path, monkeypatch):
    """Deliver through a native no-thread subscription and real injected runner auth."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from hermes_cli import kanban_db_notify as kbn
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setenv("HERMES_HOME", str(rig.notifier_home))
    task_id = rig.owner.row["payload"]["kanban_card_v1"]["task_id"]
    with connect_closing(tmp_path / "board.db") as conn:
        assert kbn.remove_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="100", thread_id="7")
        kbn.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="42", user_id="42",
            chat_type="dm", notifier_profile="default",
            delivery_metadata={"kanban_approval_forwarding": "v1", "kanban_approval_task_id": task_id})
        assert kbn.list_notify_subs(conn, task_id)[0]["thread_id"] == ""

    with _profile_runtime_scope(rig.notifier_home, hydrate_secrets=False):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.profile_routes = []
        runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
        runner.pairing_store = PairingStore(profile="default")
        runner.pairing_stores = {"default": runner.pairing_store}
        runner._primary_profile_name = "default"
        runner._profile_adapters = {}
        runner.adapters = {Platform.TELEGRAM: rig.adapter}
        runner._draining = False
        rig.owner.authority.runner = runner
        rig.adapter._message_handler = runner._primary_message_handler()
        rig.adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))

    with _profile_runtime_scope(rig.worker_home, hydrate_secrets=False):
        rig.owner.authority.register_approval(
            "worker-session", rig.owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "SECRET_TOOL_PAYLOAD", "description": "protected write"},
            timeout_s=30)
        await asyncio.wait_for(asyncio.gather(*rig.owner.authority._kanban_approval_tasks), 2)
    assert rig.bot.send_message.await_count == 1
    return next(iter(rig.adapter._approval_forwarded.items()))


async def _tap_default_dm(rig, approval_id, *, thread_id=None):
    from gateway.run import _profile_runtime_scope

    query = MagicMock()
    query.data = f"ea:deny:{approval_id}"
    query.from_user = SimpleNamespace(id=42, first_name="John")
    query.message.chat_id = 42
    query.message.message_thread_id = thread_id
    query.message.chat.type = "private"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    with _profile_runtime_scope(rig.worker_home, hydrate_secrets=False):
        await rig.adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())
    return query


@pytest.mark.asyncio
async def test_persisted_default_dm_deny_once_and_duplicate(tmp_path, monkeypatch):
    rig = _cross_profile_real_owner(tmp_path, monkeypatch)
    try:
        approval_id, binding = await _persisted_default_dm_forwarded(rig, tmp_path, monkeypatch)
        assert binding.thread_id is None

        await _tap_default_dm(rig, approval_id)
        await _tap_default_dm(rig, approval_id)  # duplicate delivery cannot resolve twice

        assert rig.owner.responses == [("approval", "request-1", "deny")]
        assert approval_id not in rig.adapter._approval_state
    finally:
        rig.owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stale", "unexpected_thread", "revoked"])
async def test_persisted_default_dm_refusal_controls(tmp_path, monkeypatch, control):
    rig = _cross_profile_real_owner(
        tmp_path, monkeypatch, worker_users="42" if control == "revoked" else "")
    try:
        approval_id, binding = await _persisted_default_dm_forwarded(rig, tmp_path, monkeypatch)
        assert binding.thread_id is None
        if control == "stale":
            from hermes_state_runtime import settle_session_input
            settle_session_input(
                rig.owner.db, epoch=rig.owner.authority.epoch,
                admission_id=rig.owner.row["admission_id"],
                generation=rig.owner.row["generation"], outcome="completed")
        elif control == "revoked":
            (rig.notifier_home / ".env").write_text(
                "TELEGRAM_ALLOWED_USERS=\nGATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")

        await _tap_default_dm(
            rig, approval_id, thread_id=7 if control == "unexpected_thread" else None)

        assert rig.owner.responses == []
        if control in {"unexpected_thread", "revoked"}:
            assert approval_id in rig.adapter._approval_state
            assert "request-1" in rig.owner.authority.sessions["worker-session"].controls.pending
    finally:
        rig.owner.db.close()


@pytest.mark.asyncio
async def test_delayed_old_generation_preflight_cannot_withdraw_new_prompt(tmp_path, monkeypatch):
    from gateway import kanban_approval_broker as broker
    from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input

    owner = _running_owner(tmp_path, monkeypatch, _Telegram())
    live = owner.authority.sessions["worker-session"]
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def delayed_preflight(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2)

    monkeypatch.setattr(broker, "approval_subscription_for_task", delayed_preflight)
    live.controls.register(
        "worker-session", "worker-route", owner.row["generation"],
        {"request_id": "request-1", "command": "old", "description": "old"})
    forwarding = asyncio.create_task(
        broker.forward_approval(owner.authority, owner.row, {"request_id": "request-1"}))
    new_responses = []

    try:
        await asyncio.wait_for(started.wait(), 2)
        settle_session_input(
            owner.db, epoch=owner.authority.epoch, admission_id=owner.row["admission_id"],
            generation=owner.row["generation"], outcome="completed")
        admit_session_input(
            owner.db, epoch=owner.authority.epoch, principal_id="dispatcher",
            session_id="worker-session", request_id="next", payload={"text": "next", "finite": True})
        current = claim_session_input(owner.db, epoch=owner.authority.epoch, session_id="worker-session")
        assert current is not None and not isinstance(current, bool)
        live.event_stream.execution = {
            "admission_id": current["admission_id"], "execution_generation": current["generation"]}
        owner.authority.executing["worker-session"] = current["generation"]
        live.controls.remote_responders.clear()
        live.controls.pending.clear()
        live.controls.register(
            "worker-session", "worker-route", current["generation"],
            {"request_id": "request-2", "command": "new", "description": "new"})
        live.controls.remote_responders["request-2"] = (
            lambda kind, prompt_id, value: new_responses.append((kind, prompt_id, value)))

        release.set()
        await asyncio.wait_for(forwarding, 2)

        assert "request-2" in live.controls.pending
        assert "request-2" in live.controls.remote_responders
        assert new_responses == []
    finally:
        release.set()
        await asyncio.gather(forwarding, return_exceptions=True)
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["normal", "disconnect", "disconnect_reconnect"])
async def test_real_adapter_disconnect_fences_inflight_forwarded_send(tmp_path, monkeypatch, transition):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._running = True
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(message_id=42)

    bot = adapter._bot
    assert bot is not None
    bot.send_message = AsyncMock(side_effect=delayed_send)
    try:
        owner.authority.register_approval(
            "worker-session", owner.row["generation"], "worker-route",
            {"request_id": "request-1", "command": "edit fixture AGENTS.md", "description": "protected write"},
            timeout_s=30)
        await asyncio.wait_for(started.wait(), 2)
        if transition != "normal":
            adapter._mark_disconnected()
        if transition == "disconnect_reconnect":
            adapter._mark_connected()
        release.set()
        await asyncio.wait_for(asyncio.gather(*owner.authority._kanban_approval_tasks), 2)
        live = owner.authority.sessions["worker-session"]

        if transition == "normal":
            assert owner.responses == []
            assert "request-1" in live.controls.pending
            approval_id = next(iter(adapter._approval_forwarded))
            query = MagicMock()
            query.data = f"ea:deny:{approval_id}"
            query.from_user = SimpleNamespace(id=42, first_name="John")
            query.message.chat_id = 100
            query.message.message_thread_id = 7
            query.message.chat.type = "supergroup"
            query.answer = AsyncMock()
            query.edit_message_text = AsyncMock()
            await adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())
            assert owner.responses == [("approval", "request-1", "deny")]
        else:
            assert owner.responses == [
                ("approval", "request-1", "cancelled:Telegram approval transport disconnected")]
            assert live.controls.pending == {}
            assert adapter._approval_forwarded == {}
            assert adapter._approval_state == {}
    finally:
        release.set()
        owner.db.close()


@pytest.mark.asyncio
async def test_disconnected_real_adapter_with_live_bot_cannot_mint_approval_route(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.kanban_approval_broker import approval_subscription_for_task
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    adapter._bot = AsyncMock()  # asynchronous teardown has not cleared the bot yet
    adapter._running = False
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    owner = _running_owner(tmp_path, monkeypatch, adapter)
    try:
        card = owner.row["payload"]["kanban_card_v1"]
        assert approval_subscription_for_task(owner.authority.runner, card["db"], card["task_id"]) is None
        bot = adapter._bot
        assert bot is not None
        assert bot.send_message.await_count == 0
    finally:
        owner.db.close()
