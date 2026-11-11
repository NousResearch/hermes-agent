"""Error text that names an exception's class names a store refusal ``RuntimeStoreError`` and any other storage
error by the SQLite error it chains: the turn-failure notice and the control socket's error. A gateway refusal
subclass and any other error keep their own class name."""
import contextlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent.i18n import t
from gateway.config import Platform
from gateway.control_socket import GatewayControlServer
from storage_user_copy import error_type_name
from gateway.parked_route_refusal import ParkedProfileRefusal
from persistence.core.errors import AuthorityRefusal, FenceStale, StaleFenceKind, UnknownExecution
from persistence.sqlite._backend import translated


def _translated(error):
    try:
        with translated():
            raise error
    except Exception as exc:
        return exc


CASES = [
    (AuthorityRefusal("session_busy", field="turn"), "RuntimeStoreError"),
    (FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton"), "RuntimeStoreError"),
    (UnknownExecution("session", "s1"), "RuntimeStoreError"),
    (ParkedProfileRefusal("work"), "ParkedProfileRefusal"),
    (ValueError("bad value"), "ValueError"),
]
IDS = ["refusal", "fence", "unknown-execution", "gateway-subclass", "other"]
# A storage error the store raised over a SQLite error; the viewer automation wire never catches one.
STORE_CASES = [
    (_translated(sqlite3.OperationalError("disk I/O error")), "OperationalError"),
    (_translated(sqlite3.IntegrityError("UNIQUE constraint failed: sessions.id")), "IntegrityError"),
]
STORE_IDS = ["store-error", "store-integrity"]


@pytest.mark.parametrize("exc,name", CASES + STORE_CASES, ids=IDS + STORE_IDS)
def test_error_type_name(exc, name):
    assert error_type_name(exc) == name


@pytest.mark.parametrize("exc,name", CASES + STORE_CASES, ids=IDS + STORE_IDS)
def test_control_socket_error(tmp_path, exc, name):
    def boom():
        raise exc

    server = GatewayControlServer(tmp_path, verb_handlers={"boom": boom})
    response = json.loads(server.handle_request_line(b'{"verb": "boom", "id": 7}'))
    assert response["ok"] is False and response["error"] == f"{name}: {exc}"


@pytest.mark.parametrize("exc,name", CASES + STORE_CASES, ids=IDS + STORE_IDS)
@pytest.mark.asyncio
async def test_turn_failure_notice(exc, name):
    from gateway.platforms.base import BasePlatformAdapter
    sent = []

    class _Adapter:
        name = "fake"

        def _media_delivery_scope(self, _source):
            return contextlib.nullcontext()

        def warning_text(self, detailed, _fallback, **_kw):
            return detailed

        async def send(self, chat_id, content, metadata=None):
            sent.append(content)

    event = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="c1", thread_id=None),
                            raw_message=None, internal=False, metadata=None)
    await BasePlatformAdapter._notify_turn_error(_Adapter(), event, exc)
    assert sent == [t("gateway.notify.turn_error_detailed", error_type=name, error_detail=str(exc)[:300])]
