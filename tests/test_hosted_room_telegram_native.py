"""Native dispatch/config/SQLite/worker exclusion, with only Telegram I/O replaced."""
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("telegram")
from telegram import Update
from telegram.ext import Application, TypeHandler
from telegram.request import BaseRequest

from gateway.hosted_rooms import create_room
from plugins.platforms.telegram import hosted_room_transport as transport


@pytest.fixture
def binding_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    identities = {"alpha": {"id": 1, "username": "alpha_test_bot"},
                  "beta": {"id": 2, "username": "beta_test_bot"}}
    for profile in identities:
        folder = home / "profiles" / profile
        folder.mkdir(parents=True)
        (folder / ".env").write_text(f"TELEGRAM_BOT_TOKEN={profile}\n")
    binding = {"enabled": True, "room_id": "native-room", "chat_id": -999,
               "owner_id": 42, "queue_db": str(tmp_path / "queue.db"),
               "control_profile": "alpha", "bots": identities}
    config = home / "binding.json"
    config.write_text(json.dumps(binding))
    (home / "config.yaml").write_text(
        "gateway:\n  hosted_rooms:\n    telegram:\n      binding_file: binding.json\n")
    db_path = tmp_path / "room.db"
    create_room(db_path, room_id=binding["room_id"], name="Native transport",
                authority_gateway_id="test-owner", members=[
                    {"member_id": "one", "profile": "alpha", "handle": "first"},
                    {"member_id": "two", "profile": "beta", "handle": "second"}])
    return binding, config, SimpleNamespace(db_path=db_path)


class LocalRequest(BaseRequest):
    def __init__(self, pending=None, *, fail_first=False):
        self.pending = pending
        self.fail_first = fail_first
        self.drops = []
        self.drained = asyncio.Event()
        self.block = asyncio.Event()

    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        if self.pending is not None:
            if url.endswith("/deleteWebhook"):
                assert request_data is not None
                drop = request_data.parameters.get("drop_pending_updates")
                self.drops.append(drop)
                if drop:
                    self.pending.clear()
                return 200, b'{"ok":true,"result":true}'
            if url.endswith("/getUpdates"):
                if self.drained.is_set():
                    await self.block.wait()
                pending, self.pending = self.pending, []
                self.drained.set()
                return 200, json.dumps({"ok": True, "result": pending}).encode()
        assert url.endswith("/getMe"), "unexpected Telegram network operation"
        if self.fail_first:
            self.fail_first = False
            raise OSError("offline transient initialization failure")
        return 200, b'{"ok":true,"result":{"id":1,"is_bot":true,"first_name":"Test","username":"alpha_test_bot"}}'


@pytest.mark.parametrize("mode", ["polling", "webhook"])
@pytest.mark.parametrize("is_reconnect", [False, True])
@pytest.mark.parametrize("binding_state", ["active", "disabled", "unbound", "unwired"])
def test_startup_backlog_policy_uses_wired_binding(binding_env, monkeypatch, mode,
                                                  is_reconnect, binding_state):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    binding, path, _ = binding_env
    binding["enabled"] = binding_state != "disabled"
    path.write_text(json.dumps(binding), encoding="utf-8")
    if binding_state == "unbound":
        (path.parent / "config.yaml").write_text("{}", encoding="utf-8")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:local-test"))
    assert adapter._hosted_room_ingress_active is False
    app = Application.builder().token("123:local-test").request(LocalRequest()).build()
    if binding_state != "unwired":
        # Rewiring must replace an earlier active decision, not leave it latched on.
        adapter._hosted_room_ingress_active = True
        adapter._wire_plugin_handlers(app)
    else:
        # Existing installed fixtures bypass __init__ and call startup directly.
        vars(adapter).pop("_hosted_room_ingress_active", None)
    # Moving config must not select a policy different from the handlers just wired.
    binding["enabled"] = not binding["enabled"]
    path.write_text(json.dumps(binding), encoding="utf-8")
    starter = AsyncMock(return_value=True)
    adapter._app = app
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock())
    monkeypatch.setattr(adapter, "_start_polling_resilient", starter)
    monkeypatch.setattr(type(app.updater), "start_webhook", starter)
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name: "offline-secret")
    if mode == "polling":
        asyncio.run(adapter._start_polling_mode(is_reconnect=is_reconnect))
    else:
        asyncio.run(adapter._start_webhook_mode("https://offline.invalid/telegram",
                                               is_reconnect=is_reconnect))
    assert starter.call_args.kwargs["drop_pending_updates"] is (
        not is_reconnect and binding_state != "active")


@pytest.mark.parametrize("rebuild", [False, True])
def test_real_ptb_initial_polling_captures_pending_room_message(binding_env, monkeypatch, rebuild):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    binding, _, _ = binding_env

    async def exercise():
        pending = [{"update_id": 1, "message": {
            "message_id": 1, "date": 1, "chat": {"id": -999, "type": "supergroup"},
            "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
            "text": "sent during cutover"}}, {"update_id": 2, "message": {
            "message_id": 2, "date": 1, "chat": {"id": -555, "type": "supergroup"},
            "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
            "text": "other chat backlog"}}]
        request = LocalRequest(pending, fail_first=rebuild)
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:local-test"))
        builder = (Application.builder().token("123:local-test").request(request)
                   .get_updates_request(adapter._instrument_polling_request(request)))
        adapter._app = original = builder.build()
        adapter._bot = original.bot
        adapter._wire_plugin_handlers(original)
        # Sentinel core handler detects any owned-room fallthrough after a retry rebuild.
        fallback = []

        async def other(update, context):
            fallback.append(update.update_id)

        monkeypatch.setattr(adapter, "_register_handlers", lambda app: app.add_handler(
            TypeHandler(Update, other), group=0))
        adapter._register_handlers(original)
        await adapter._initialize_app_with_retries(builder)
        app = adapter._app
        assert app is not None and app.updater is not None
        assert (app is not original) is rebuild
        await app.start()
        try:
            await adapter._start_polling_mode(is_reconnect=False)
            await asyncio.wait_for(request.drained.wait(), timeout=2)
            await asyncio.wait_for(app.update_queue.join(), timeout=2)
            with sqlite3.connect(binding["queue_db"]) as db:
                assert db.execute("SELECT message_id,text FROM inbox").fetchall() == [
                    (1, "sent during cutover")]
            assert request.drops and all(drop is False for drop in request.drops)
            assert fallback == [2]
        finally:
            request.block.set()
            if app.updater.running:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()

    asyncio.run(exercise())


def test_real_ptb_rebuild_reserves_room_and_never_falls_through(binding_env):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    binding, config, _ = binding_env
    fallback = []

    async def exercise():
        for generation in range(2):
            adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:local-test"))
            app = Application.builder().token("123:local-test").request(LocalRequest()).build()
            adapter._wire_plugin_handlers(app)

            async def other(update, context):
                fallback.append(update.update_id)

            app.add_handler(TypeHandler(Update, other), group=0)
            await app.initialize()
            try:
                for update_id, chat, owner in ((1, -999, 42), (2, -999, 666), (3, -555, 42)):
                    update = Update.de_json({"update_id": generation * 10 + update_id, "message": {
                        "message_id": generation * 10 + update_id, "date": 1,
                        "chat": {"id": chat, "type": "supergroup", "title": "Test"},
                        "from": {"id": owner, "is_bot": False, "first_name": "Owner"},
                        "text": "native input"}}, app.bot)
                    await app.process_update(update)
                binding["enabled"] = False
                config.write_text(json.dumps(binding))
                update = Update.de_json({"update_id": 99, "message": {
                    "message_id": 99, "date": 1, "chat": {"id": -999, "type": "supergroup"},
                    "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
                    "text": "disabled input"}}, app.bot)
                await app.process_update(update)
            finally:
                await app.shutdown()
    asyncio.run(exercise())
    with sqlite3.connect(binding["queue_db"]) as db:
        assert db.execute("SELECT message_id,user_id,text FROM inbox").fetchall() == [(1, 42, "native input")]
    assert fallback == [3, 13]


def test_live_owner_excludes_both_processes_before_recovering_receipts(binding_env, monkeypatch):
    binding, _, service = binding_env
    entered, release = threading.Event(), threading.Event()
    closed = []

    class Bot:
        def __init__(self, token):
            self.profile = token

        async def initialize(self):
            pass

        async def get_me(self):
            return SimpleNamespace(**binding["bots"][self.profile])

        async def shutdown(self):
            if self.profile == "alpha":
                entered.set()
                assert await asyncio.to_thread(release.wait, 10)
            closed.append(self.profile)

    monkeypatch.setattr("telegram.Bot", Bot)
    owner = transport.Transport(service, binding)
    contender = transport.Transport(service, binding)
    try:
        owner.start()
        assert owner.member_profiles == {"one": "alpha", "two": "beta"}
        assert owner.profile_handles == {"alpha": "@first", "beta": "@second"}
        with owner.db() as db:
            db.execute("INSERT INTO deliveries(event_id,chunk_index,profile,thread_id,status)"
                       " VALUES('in-flight',0,'alpha','thread','sending')")
        with pytest.raises(RuntimeError, match="already owned"):
            contender.start()
        code = """import json,sys
from pathlib import Path
from types import SimpleNamespace
from plugins.platforms.telegram.hosted_room_transport import Transport
item=Transport(SimpleNamespace(db_path=Path(sys.argv[1])),json.loads(sys.argv[2]))
def poison():
    item.error='unexpected admission'; item.ready.set()
item._run=poison
try:
    item.start()
except RuntimeError as exc:
    assert 'already owned' in str(exc), type(exc).__name__
    print('EXCLUDED')
else:
    raise AssertionError('second consumer admitted')
finally:
    item.stop(timeout=10)
"""
        result = subprocess.run([sys.executable, "-c", code, str(service.db_path), json.dumps(binding)],
                                cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "EXCLUDED"
        with owner.db() as db:
            assert db.execute("SELECT status FROM deliveries WHERE event_id='in-flight'").fetchone()[0] == "sending"
        owner.halt.set()
        assert entered.wait(10)
        assert owner.stop(timeout=0) is False
        assert owner._mutex is not None
        with pytest.raises(RuntimeError, match="already owned"):
            contender.start()
    finally:
        release.set()
        assert owner.stop(timeout=10)
        assert contender.stop(timeout=10)
    assert closed == ["alpha", "beta"]
    replacement = transport.Transport(service, binding)
    try:
        replacement.start()
        with replacement.db() as db:
            assert db.execute("SELECT status FROM deliveries WHERE event_id='in-flight'").fetchone()[0] == "uncertain"
    finally:
        assert replacement.stop(timeout=10)


def test_startup_failure_closes_every_client_and_keeps_errors_redacted(binding_env, monkeypatch, caplog):
    binding, _, service = binding_env
    closed = []

    class Bot:
        def __init__(self, token):
            self.profile = token

        async def initialize(self):
            if self.profile == "beta":
                raise ValueError("must-not-appear-in-logs")

        async def get_me(self):
            return SimpleNamespace(**binding["bots"][self.profile])

        async def shutdown(self):
            closed.append(self.profile)
            if self.profile == "alpha":
                raise RuntimeError("must-not-appear-in-logs")

    monkeypatch.setattr("telegram.Bot", Bot)
    item = transport.Transport(service, binding)
    try:
        with pytest.raises(RuntimeError, match="initialization failed"):
            item.start()
    finally:
        assert item.stop(timeout=10)
    assert closed == ["alpha", "beta"]
    assert "must-not-appear-in-logs" not in caplog.text


@pytest.mark.parametrize("change", [
    {"enabled": "false"}, {"owner_id": True}, {"chat_id": "-999"},
    {"queue_db": "relative.db"}, {"control_profile": "missing"},
    {"bots": {"../escape": {"id": 1, "username": "alpha_test_bot"}}},
])
def test_malformed_binding_never_mutates_the_queue(binding_env, change):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    binding, path, _ = binding_env
    binding.update(change)
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        transport.load_binding(path)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:local-test"))
    adapter._hosted_room_ingress_active = True
    app = Application.builder().token("123:local-test").request(LocalRequest()).build()
    with pytest.raises(ValueError):
        adapter._wire_plugin_handlers(app)
    assert adapter._hosted_room_ingress_active is False
    assert not app.handlers
    assert not Path(binding["queue_db"]).exists()


@pytest.mark.parametrize("available", [True, False])
def test_transport_prepares_native_dependency_before_sdk_import(tmp_path, monkeypatch, available):
    from plugins.platforms.telegram import adapter

    sdk = sys.modules["telegram"]
    calls = []
    monkeypatch.setitem(sys.modules, "telegram", None)

    def prepare():
        calls.append("platform.telegram")
        if available:
            monkeypatch.setitem(sys.modules, "telegram", sdk)
        return available

    monkeypatch.setattr(adapter, "check_telegram_requirements", prepare)
    # Exercise only dependency initialization, without accounts or Telegram I/O.
    item = transport.Transport(None, {"queue_db": str(tmp_path / "queue.db"),
                                     "room_id": "room", "chat_id": -999, "bots": {}})
    item.halt.set()
    if available:
        asyncio.run(item.run())
        assert item.ready.is_set()
    else:
        with pytest.raises(RuntimeError, match="Telegram dependency unavailable"):
            asyncio.run(item.run())
        assert not item.ready.is_set()
    assert calls == ["platform.telegram"]
    assert not item.path.exists()


class DeliveryRequest(LocalRequest):
    """Real PTB serialization/error decoding; no request can leave this scripted boundary."""

    def __init__(self, fail_part=1, response=None):
        super().__init__()
        self.fail_part, self.response = fail_part, response
        self.sends = []
        self.on_send = None

    async def do_request(self, url, method, request_data=None, **kwargs):
        if url.endswith('/getMe'):
            return await super().do_request(url, method, request_data, **kwargs)
        assert url.rsplit('/', 1)[-1] in {'sendMessage', 'sendDocument', 'sendPhoto'}
        self.sends.append((request_data.parameters, request_data.multipart_data))
        if self.on_send:
            self.on_send()
        if len(self.sends) - 1 == self.fail_part:
            if self.response:
                return self.response
            from telegram.error import TimedOut
            raise TimedOut('response lost; unsafe exception detail must not be persisted')
        return 200, json.dumps({'ok': True, 'result': {
            'message_id': 100 + len(self.sends), 'date': 1,
            'chat': {'id': -999, 'type': 'supergroup'}, 'text': 'scripted'}}).encode()


@pytest.fixture
def recovery(binding_env, monkeypatch):
    from gateway import hosted_rooms
    from tui_gateway import methods_groups, server
    from tui_gateway.hosted_room_service import HostedRoomService

    binding, _, original = binding_env
    monkeypatch.setattr(hosted_rooms, 'local_authority_gateway_id', lambda: 'test-owner')
    monkeypatch.setattr(hosted_rooms, 'default_db_path', lambda: original.db_path)
    service = HostedRoomService(SimpleNamespace(), db_path=original.db_path)
    data = b'immutable committed source blob\n'
    manifests = []
    for index in range(2):
        uploaded = service.attachments.put(
            room_id=binding['room_id'], upload_id=f'upload-{index}', kind='file',
            name=f'output-{index}.txt', mime='text/plain', data=data + str(index).encode())
        manifests.append({key: uploaded[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')})
    service.attachments.commit_message(
        room_id=binding['room_id'], event_id='output', manifest=manifests,
        recipient_member_ids=['one', 'two'], viewer_access=True, hold_until_event=True)
    common = dict(room_id=binding['room_id'], authority_gateway_id='test-owner', authority_epoch=1)
    hosted_rooms.append_event(
        service.db_path, **common, event_id='output', kind='message.member',
        actor={'kind': 'member', 'id': 'one'}, payload={
            'text': 'a' * 1800 + 'b' * 1800 + 'c' * 100, 'member_id': 'one',
            'thread_id': 'thread', 'attachments': manifests})
    hosted_rooms.append_event(
        service.db_path, **common, event_id='settled', kind='turn.settled',
        actor={'kind': 'gateway', 'id': 'test-owner'}, payload={'message_event_id': 'output'})
    item = transport.Transport(service, binding)

    def parked_worker():
        # The real lifetime lock/start/recovery are exercised; publication is stepped explicitly
        # through real PTB below, rather than letting the native loop race test assertions.
        item.ready.set()
        item.halt.wait(10)

    monkeypatch.setattr(item, '_run', parked_worker)
    monkeypatch.setattr(methods_groups, '_transport', item)
    monkeypatch.setattr(server, 'get_hosted_room_service', lambda: service)
    monkeypatch.setattr(service, 'status', lambda room_id: {'running': False})
    item.start()

    def resolve(**changes):
        params = dict(room_id=item.room, event_id='output', chunk_index=1, attempt=1,
                      decision='retry', confirm=True, authority_gateway_id='test-owner', authority_epoch=1)
        params.update(changes)
        return server._methods['groups.telegram.resolve_delivery'](123, params)

    try:
        yield item, resolve, server
    finally:
        if methods_groups._transport is not item:
            methods_groups._transport.stop(timeout=10)
        item.stop(timeout=10)


@pytest.mark.parametrize('part', [1, 3], ids=['text', 'document'])
@pytest.mark.parametrize('decision', ['retry', 'confirmed-delivered'])
def test_native_delivery_reconciliation_survives_lost_rpc_response_and_restart(recovery, monkeypatch, part, decision):
    from telegram import Bot
    from telegram.error import NetworkError
    from gateway.hosted_room_driver import list_tasks
    from tui_gateway import methods_groups

    item, resolve, server = recovery
    before = item.events()
    blobs = item.published_media(before[-2])
    tasks = list_tasks(item.service.db_path, room_id=item.room)
    request = DeliveryRequest(fail_part=part)
    args = dict(chunk_index=part, decision=decision,
                **({'message_id': 999} if decision == 'confirmed-delivered' else {}))

    async def first():
        async with Bot('123:local-test', request=request) as bot:
            with pytest.raises(NetworkError):
                await item.publish({'alpha': bot})
            with pytest.raises(RuntimeError, match='requires readback'):
                await item.publish({'alpha': bot})
    asyncio.run(first())
    state = server._methods['groups.state'](1, {'room_id': item.room})['result']['telegram_status']
    assert state['blocked'] is True
    assert state['attention']['chunk_index'] == part
    assert state['attention']['failure_kind'] == 'ambiguous'
    assert len(request.sends) == part + 1
    cursor = state['cursor']
    ack = resolve(**args)  # Imagine this response is lost after the SQLite commit.
    assert 'error' not in ack, ack
    assert resolve(**args) == ack
    assert item.status()['cursor'] == cursor  # Reconciliation itself never moves the cursor.
    assert item.stop(timeout=10)
    item = transport.Transport(item.service, item.config)

    def parked_replacement():
        item.ready.set()
        item.halt.wait(10)

    monkeypatch.setattr(item, '_run', parked_replacement)
    monkeypatch.setattr(methods_groups, '_transport', item)
    item.start()
    assert resolve(**args) == ack

    async def finish():
        async with Bot('123:local-test', request=request) as bot:
            await item.publish({'alpha': bot})
            await item.publish({'alpha': bot})
    asyncio.run(finish())
    # Five parts total. The already-sent prefix is not repeated; only authorized part replay is extra.
    assert len(request.sends) == 5 + (decision == 'retry')
    expected_parts = ['a' * 1800, 'b' * 1800, 'c' * 100, *[data for _, data in blobs]]
    if decision == 'retry':
        expected_parts.insert(part, expected_parts[part])
    actual_parts = [params.get('text') if 'text' in params else next(iter(files.values()))[1]
                    for params, files in request.sends]
    assert actual_parts == expected_parts
    if decision == 'retry':
        # Existing reply policy can now quote the delivered prefix in the SAME thread.
        first_params, first_files = request.sends[part]
        retry_params, retry_files = request.sends[part + 1]
        assert {k: v for k, v in first_params.items() if k != 'reply_parameters'} == {
            k: v for k, v in retry_params.items() if k != 'reply_parameters'}
        assert list(first_files.values()) == list(retry_files.values())
    with item.db() as db:
        rows = db.execute('SELECT * FROM deliveries ORDER BY chunk_index').fetchall()
        assert len(rows) == 5 and all(row['status'] == 'sent' for row in rows)
        mapped = rows[part]['message_id']
        evidence = db.execute('SELECT * FROM delivery_resolutions').fetchone()
        assert evidence['previous_status'] == 'uncertain' and evidence['failure_kind'] == 'ambiguous'
    assert item.thread_for({'reply_to': mapped, 'event_id': 'reply'}) == 'thread'
    assert resolve(**args)['result']['decision'] == decision
    assert item.status()['cursor'] == before[-1]['seq']
    assert item.events() == before
    assert item.published_media(before[-2]) == blobs
    assert list_tasks(item.service.db_path, room_id=item.room) == tasks == []
    for _, multipart in request.sends:
        for _, content, _ in multipart.values():
            assert content in [data for _, data in blobs]


@pytest.mark.parametrize('code,parameters,expected', [
    (400, {}, 'BadRequest'), (403, {}, 'Forbidden'), (401, {}, 'InvalidToken'),
    (429, {'retry_after': 30}, 'RetryAfter'),
])
def test_real_ptb_negative_responses_remain_blocked_and_retain_refusal(recovery, monkeypatch, code, parameters, expected):
    from telegram import Bot
    from telegram.error import TelegramError

    item, resolve, _ = recovery
    request = DeliveryRequest(response=(code, json.dumps({
        'ok': False, 'error_code': code, 'description': 'scripted refusal', 'parameters': parameters}).encode()))

    async def fail():
        async with Bot('123:local-test', request=request) as bot:
            with pytest.raises(TelegramError):
                await item.publish({'alpha': bot})
            with pytest.raises(RuntimeError):
                await item.publish({'alpha': bot})
    asyncio.run(fail())
    attention = item.status()['attention']
    assert attention['status'] == 'rejected' and attention['failure_kind'] == expected
    assert len(request.sends) == 2
    if expected == 'RetryAfter':
        assert attention['retry_after'] > transport.time.time()
        assert item.stop(timeout=10)
        item.halt.clear()
        item.start()
        assert item.status()['attention']['retry_after'] == attention['retry_after']
        assert 'cooldown' in resolve()['error']['message']
        monkeypatch.setattr(transport.time, 'time', lambda: attention['retry_after'] + 1)
    assert 'error' not in resolve()
    assert item.status()['attention']['status'] == 'retry_authorized'
    with item.db() as db:
        evidence = db.execute('SELECT * FROM delivery_resolutions').fetchone()
        assert evidence['previous_status'] == 'rejected'
        assert evidence['failure_kind'] == expected
        assert evidence['retry_after'] == attention['retry_after']
    assert len(request.sends) == 2  # No RPC calls the Bot or starts an agent.


@pytest.mark.parametrize('decision', ['retry', 'confirmed-delivered'])
@pytest.mark.parametrize('refused', [False, True], ids=['uncertain', 'rejected'])
@pytest.mark.parametrize('unreadable', [False, True], ids=['readable', 'db-error'])
def test_run_parks_failed_publication_until_durable_resolution(
        recovery, monkeypatch, caplog, decision, refused, unreadable):
    from telegram.error import BadRequest, TimedOut

    item, resolve, _ = recovery
    sends, closed, sleeps = [], [], []
    original_status = item.status
    for name in ('publish', 'published_media', 'events', 'status'):
        original = getattr(item, name)
        spy = AsyncMock(wraps=original) if name == 'publish' else Mock(wraps=original)
        monkeypatch.setattr(item, name, spy)
    cursor = original_status()['cursor']

    class Bot:
        def __init__(self, token):
            self.profile = token

        async def initialize(self):
            pass

        async def get_me(self):
            return SimpleNamespace(**item.config['bots'][self.profile])

        async def shutdown(self):
            closed.append(self.profile)

        async def send_message(self, **kwargs):
            sends.append(kwargs['text'])
            if len(sends) == 2:
                raise BadRequest('refused') if refused else TimedOut('lost response')
            return SimpleNamespace(message_id=100 + len(sends))

        async def send_document(self, **kwargs):
            sends.append(kwargs['document'])
            return SimpleNamespace(message_id=100 + len(sends))

    monkeypatch.setattr('telegram.Bot', Bot)
    # Ingest uses the native queue and canonical service; no worker/model is started.
    monkeypatch.setattr(item.service, 'prepare_room', lambda binding: None)

    async def step(delay):
        sleeps.append(delay)
        turn = len(sleeps)
        assert turn <= 6
        if turn <= 4:
            assert delay == 2
            assert len(sends) == 2
            assert item.publish.call_count == 1
            assert item.published_media.call_count == 1
            assert item.events.call_count == 1
            assert item.status.call_count == turn - 1
            with item.db() as db:
                assert db.execute('SELECT seq FROM cursor').fetchone()[0] == cursor
                assert db.execute("SELECT count(*) FROM inbox WHERE state='accepted'").fetchone()[0] == turn - 1
                if unreadable and turn == 2:
                    db.execute('ALTER TABLE hidden_deliveries RENAME TO deliveries')
                if turn < 4:
                    db.execute('INSERT INTO inbox(event_id,chat_id,message_id,user_id,text,received_at)'
                               ' VALUES(?,-999,?,42,?,1)', (f'input-{turn}', turn, f'input {turn}'))
                if unreadable and turn == 1:
                    db.execute('ALTER TABLE deliveries RENAME TO hidden_deliveries')
            if turn == 4:
                assert original_status()['blocked']
                args = {'decision': decision}
                if decision == 'confirmed-delivered':
                    args['message_id'] = 999
                assert 'error' not in resolve(**args)
                assert original_status()['cursor'] == cursor
        else:
            assert delay == 0.25
            assert item.status.call_count == 4  # No receipt scan on the healthy loop.
            assert len(sends) == 5 + (decision == 'retry')
            assert original_status()['blocked'] is False
            with item.db() as db:
                assert all(row[0] == 'sent' for row in db.execute('SELECT status FROM deliveries'))
            if turn == 6:
                item.halt.set()

    monkeypatch.setattr(transport.asyncio, 'sleep', step)
    asyncio.run(item.run())
    assert closed == ['alpha', 'beta']
    errors = [record.message for record in caplog.records if record.levelname == 'ERROR']
    assert errors == [f"hosted room transport iteration failed: {'BadRequest' if refused else 'TimedOut'}"] + (
        ['hosted room transport iteration failed: OperationalError'] if unreadable else [])
    expected: list[str | bytes] = ['a' * 1800, 'b' * 1800]
    if decision == 'retry':
        expected.append('b' * 1800)
    expected += ['c' * 100, b'immutable committed source blob\n0', b'immutable committed source blob\n1']
    assert sends == expected
    assert original_status()['cursor'] == item.events()[-1]['seq']


def test_reconciliation_guards_sending_identity_authority_and_stale_attempt(recovery):
    from telegram import Bot
    from telegram.error import NetworkError

    item, resolve, server = recovery
    request = DeliveryRequest()
    in_flight = []
    request.on_send = lambda: in_flight.append(resolve()) if len(request.sends) == 2 else None

    async def fail():
        async with Bot('123:local-test', request=request) as bot:
            with pytest.raises(NetworkError):
                await item.publish({'alpha': bot})
    asyncio.run(fail())
    assert 'currently sending' in in_flight[0]['error']['message']
    before = item.status()
    bad = [{'room_id': 'wrong'}, {'event_id': 'missing'}, {'chunk_index': 99}, {'chunk_index': True},
           {'attempt': True}, {'attempt': 2}, {'authority_gateway_id': 'other'}, {'authority_epoch': 2},
           {'authority_epoch': True}, {'confirm': False}, {'decision': 'skip'}, {'message_id': 9}]
    bad += [dict(decision='confirmed-delivered', message_id=value)
            for value in [None, True, '999', 1.5, 0, -1, 2**53, 101]]
    for params in bad:
        assert 'error' in resolve(**params), params
        assert item.status() == before
    with item.db() as db:
        db.execute("INSERT INTO inbox(event_id,chat_id,message_id,user_id,text,received_at)"
                   " VALUES('mapped-input',-999,998,42,'input',1)")
    assert 'already mapped' in resolve(decision='confirmed-delivered', message_id=998)['error']['message']
    with item.db() as db:
        db.execute("UPDATE deliveries SET profile='beta' WHERE chunk_index=1")
    assert 'identity mismatch' in resolve()['error']['message']
    with item.db() as db:
        db.execute("UPDATE deliveries SET profile='alpha' WHERE chunk_index=1")
        db.execute("UPDATE deliveries SET thread_id='wrong' WHERE chunk_index=1")
    assert 'identity mismatch' in resolve()['error']['message']
    with item.db() as db:
        db.execute("UPDATE deliveries SET thread_id='thread' WHERE chunk_index=1")
    assert 'error' not in resolve()
    request.fail_part = len(request.sends)  # Explicit replay also loses its response.
    request.on_send = None
    asyncio.run(fail())
    second = item.status()
    assert second['attention']['attempt'] == 2 and second['blocked']
    assert 'error' not in resolve()  # Old response readback, NOT a second authorization.
    assert item.status() == second
    assert 'different decision' in resolve(decision='confirmed-delivered', message_id=999)['error']['message']
    assert 'groups.telegram.resolve_delivery' in server._LONG_HANDLERS
    item.halt.set()
    assert 'unavailable' in resolve(attempt=2)['error']['message']


def test_restart_recovers_legacy_sending_then_native_resolution_is_fenced_by_room_state(recovery):
    from gateway import hosted_rooms

    item, resolve, _ = recovery
    # Old queue rows migrate with attempt=1, and only the exclusive replacement can recover them.
    with item.db() as db:
        db.execute("INSERT INTO deliveries(event_id,chunk_index,profile,thread_id,status)"
                   " VALUES('output',1,'alpha','thread','sending')")
    assert 'currently sending' in resolve()['error']['message']
    assert item.stop(timeout=10)
    item.halt.clear()
    item.start()
    assert item.status()['attention']['status'] == 'uncertain'
    # Canonical authority is checked even if the caller supplies the old, otherwise valid identity.
    with sqlite3.connect(item.service.db_path) as db:
        db.execute("UPDATE hosted_rooms SET authority_gateway_id='other' WHERE room_id=?", (item.room,))
    assert 'authority changed' in resolve()['error']['message']
    with sqlite3.connect(item.service.db_path) as db:
        db.execute("UPDATE hosted_rooms SET authority_gateway_id='test-owner' WHERE room_id=?", (item.room,))
    ack = resolve(decision='confirmed-delivered', message_id=999)
    assert 'error' not in ack
    assert resolve(decision='confirmed-delivered', message_id=999) == ack
    hosted_rooms.begin_room_disband(
        item.service.db_path, room_id=item.room, expected_gateway_id='test-owner', expected_epoch=1)
    assert 'being disbanded' in resolve(decision='confirmed-delivered', message_id=999)['error']['message']


def test_queue_migration_preserves_old_ambiguous_delivery_and_cursor(tmp_path):
    path = tmp_path / 'legacy-queue.db'
    with sqlite3.connect(path) as db:
        db.executescript(transport.SCHEMA)
        db.execute("INSERT INTO deliveries VALUES('old-output',1,'alpha','thread','uncertain',NULL)")
        db.execute('UPDATE cursor SET seq=12')
    transport.initialize_queue(path)
    transport.initialize_queue(path, recover=True)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM deliveries').fetchone() == (
            'old-output', 1, 'alpha', 'thread', 'uncertain', None, 1, None, None)
        assert db.execute('SELECT seq FROM cursor').fetchone() == (12,)
        assert db.execute('SELECT count(*) FROM delivery_resolutions').fetchone() == (0,)
