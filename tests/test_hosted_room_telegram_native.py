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
    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        assert url.endswith("/getMe"), "unexpected Telegram network operation"
        return 200, b'{"ok":true,"result":{"id":1,"is_bot":true,"first_name":"Test","username":"alpha_test_bot"}}'


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
            db.execute("INSERT INTO deliveries VALUES('in-flight',0,'alpha','thread','sending',NULL)")
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
    binding, path, _ = binding_env
    binding.update(change)
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        transport.load_binding(path)
    assert not Path(binding["queue_db"]).exists()
