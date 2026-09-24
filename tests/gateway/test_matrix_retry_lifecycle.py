"""Causal Matrix transport and lifecycle regressions (no homeserver)."""
import asyncio
import sys
from json import JSONDecodeError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mautrix.api import Method
from mautrix.client import Client
from mautrix.errors import EncryptionError

from plugins.platforms.matrix import adapter as matrix


class Response:
    def __init__(self, status=429, delay=1000):
        self.status = status
        self.data = ({"errcode": "M_LIMIT_EXCEEDED", "error": "limited", "retry_after_ms": delay}
                     if status == 429 else {"event_id": "$ok"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return self.data

    async def text(self):
        return "limited"


@pytest.mark.asyncio
async def test_429_retries_only_exact_replay_safe_put_and_honors_delay(monkeypatch):
    calls, delays = [], []
    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return Response(429) if len(calls) == 1 else Response(200)
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr(matrix.asyncio, "sleep", sleep)
    api = matrix._create_matrix_http_api(
        base_url="https://example.invalid", token="", client_session=SimpleNamespace(request=request),
        default_retry_count=0)
    assert await api.request(Method.PUT, "_matrix/client/v3/rooms/!r/send/m.room.message/txn", {"body": "hello"}) == {"event_id": "$ok"}
    assert calls[0] == calls[1]
    assert delays == [1.0]
    calls.clear()
    with pytest.raises(Exception):
        await api.request(Method.POST, "_matrix/client/v3/rooms/!r/send/m.room.message", {"body": "hello"})
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(Exception):
        await api.request(Method.PUT, "_matrix/client/v3/rooms/!r/send/m.room.message", {"body": "hello"})
    assert len(calls) == 1  # missing transaction ID must not retry


@pytest.mark.asyncio
async def test_429_retry_freezes_mutable_body_before_first_attempt(monkeypatch):
    body = bytearray(b'{"body":"original"}')
    calls = []
    def request(method, url, **kwargs):
        calls.append((str(url), kwargs["data"]))
        return Response(429) if len(calls) == 1 else Response(200)
    async def sleep(_delay):
        body[:] = b'{"body":"modified"}'
    monkeypatch.setattr(matrix.asyncio, "sleep", sleep)
    api = matrix._create_matrix_http_api(
        base_url="https://example.invalid", token="",
        client_session=SimpleNamespace(request=request), default_retry_count=0)
    url = SimpleNamespace(raw_path="/_matrix/client/v3/rooms/!r/send/m.room.message/txn")
    result, _response = await api._send(Method.PUT, url, body, None, {})
    assert result == {"event_id": "$ok"}
    assert calls == [(str(url), b'{"body":"original"}')] * 2
    assert all(type(payload) is bytes for _path, payload in calls)


@pytest.mark.asyncio
async def test_429_retry_is_bounded_and_ambiguous_timeout_has_no_encrypted_fallback(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(args)
        return Response(429, 1) if len(calls) < 5 else Response(200)
    async def sleep(_delay):
        pass
    monkeypatch.setattr(matrix.asyncio, "sleep", sleep)
    api = matrix._create_matrix_http_api(base_url="https://example.invalid", token="",
                                          client_session=SimpleNamespace(request=request), default_retry_count=0)
    with pytest.raises(Exception):
        await api.request(Method.PUT, "_matrix/client/v3/rooms/!r/send/m.room.message/txn", {"body": "hello"})
    assert len(calls) == 3

    calls.clear()
    def too_late(*args, **kwargs):
        calls.append(args)
        return Response(429, 31000)
    api.session.request = too_late
    with pytest.raises(Exception):
        await api.request(Method.PUT, "_matrix/client/v3/rooms/!r/send/m.room.message/txn", {"body": "hello"})
    assert len(calls) == 1  # retry_after_ms exceeds the 30s recovery deadline

    adapter = object.__new__(matrix.MatrixAdapter)
    adapter.platform = matrix.Platform.MATRIX
    adapter._encryption = True
    shares = []
    async def share():
        shares.append(True)
    adapter._client = SimpleNamespace(crypto=SimpleNamespace(share_keys=share))
    adapter.format_message = lambda text: text
    adapter.truncate_message = lambda text, _limit: [text]
    adapter.max_message_length = 1000
    adapter._build_text_message_content = lambda text: {"msgtype": "m.text", "body": text}
    adapter._apply_relation_metadata = lambda *args, **kwargs: None
    async def timeout(*args):
        raise asyncio.TimeoutError("delivery unknown")
    adapter._send_room_message = timeout
    result = await adapter.send("!r", "hello")
    assert not result.success
    assert not shares
    sends = []
    async def failed_send(*args, **kwargs):
        sends.append(True)
        return result
    adapter.send = failed_send
    assert not (await adapter._send_with_retry("!r", "hello")).success
    assert sends == [True]  # base must not create a fresh transaction after ambiguous timeout
    from mautrix.errors import make_request_error
    limit = make_request_error(http_status=429, text="limited", errcode="M_LIMIT_EXCEEDED", message="limited")
    async def rate_limited(*args):
        raise limit
    adapter._send_room_message = rate_limited
    # Exercise the real adapter result through the base retry/fallback path.
    del adapter.send
    sends.clear()
    original = adapter.send
    async def counted_send(*args, **kwargs):
        sends.append(True)
        return await original(*args, **kwargs)
    adapter.send = counted_send
    limited = await adapter._send_with_retry("!r", "hello")
    assert limited.error_kind == "rate_limited"
    assert sends == [True]
    assert not shares


@pytest.mark.asyncio
async def test_connect_disconnect_serialized_and_cancelled_connect_settles(monkeypatch):
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._lifecycle_lock = asyncio.Lock()
    entered, release = asyncio.Event(), asyncio.Event()
    order = []
    async def connect_impl(**kwargs):
        order.append("connect-start")
        entered.set()
        await release.wait()
        order.append("connect-end")
        return True
    async def disconnect_impl():
        order.append("disconnect")
    adapter._connect_impl = connect_impl
    adapter._disconnect_impl = disconnect_impl
    connecting = asyncio.create_task(adapter.connect())
    try:
        await asyncio.wait_for(entered.wait(), 1)
    except asyncio.TimeoutError:
        connecting.cancel()
        await asyncio.gather(connecting, return_exceptions=True)
        pytest.fail("connect did not enter serialized implementation")
    disconnecting = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0)
    assert order == ["connect-start"]
    release.set()
    await asyncio.gather(connecting, disconnecting)
    assert order == ["connect-start", "connect-end", "disconnect"]


@pytest.mark.asyncio
async def test_partial_session_closed_once_on_constructor_failure(monkeypatch):
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._lifecycle_lock = asyncio.Lock()
    adapter._client = adapter._crypto_db = adapter._sync_task = adapter._opening_session = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._homeserver = "https://example.invalid"
    adapter._proxy_url = None
    adapter._access_token = "token"
    adapter._user_id = "@bot:example.invalid"
    adapter._device_id = ""
    adapter._encryption = False
    adapter._resolve_store_dir = lambda: SimpleNamespace(mkdir=lambda **kwargs: None)
    closes = []
    class Session:
        async def close(self):
            closes.append(True)
    monkeypatch.setattr(matrix, "_create_matrix_session", lambda _: Session())
    monkeypatch.setattr(matrix, "_create_matrix_http_api", lambda **kw: (_ for _ in ()).throw(RuntimeError("constructor")), raising=False)
    with pytest.raises(RuntimeError, match="constructor"):
        await adapter.connect()
    assert closes == [True]
    assert adapter._client is None


@pytest.mark.asyncio
async def test_cancelled_partial_connect_owns_session_until_teardown(monkeypatch):
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._lifecycle_lock = asyncio.Lock()
    adapter._client = adapter._crypto_db = adapter._sync_task = adapter._opening_session = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._homeserver = "https://example.invalid"
    adapter._proxy_url = None
    adapter._access_token = "token"
    adapter._user_id = "@bot:example.invalid"
    adapter._device_id = ""
    adapter._encryption = False
    adapter._resolve_store_dir = lambda: SimpleNamespace(mkdir=lambda **kwargs: None)
    entered, cleanup_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    closes = []
    class Session:
        async def close(self):
            cleanup_started.set()
            await release.wait()
            closes.append(True)
    session = Session()
    monkeypatch.setattr(matrix, "_create_matrix_session", lambda _: session)
    monkeypatch.setattr(matrix, "_create_matrix_http_api", lambda **kwargs: SimpleNamespace(session=session))
    class Client:
        def __init__(self, **kwargs):
            self.api = kwargs["api"]
        async def whoami(self):
            entered.set()
            await asyncio.Event().wait()
    fake_client = SimpleNamespace(Client=Client)
    fake_store = SimpleNamespace(MemoryStateStore=MagicMock, MemorySyncStore=MagicMock)
    monkeypatch.setitem(sys.modules, "mautrix.client", fake_client)
    monkeypatch.setitem(sys.modules, "mautrix.client.state_store", fake_store)
    connecting = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(entered.wait(), 2)
    connecting.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 2)
    connecting.cancel()
    assert not connecting.done() and not closes
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(connecting, 2)
    assert closes == [True]
    assert adapter._client is None
    await adapter.disconnect()
    assert closes == [True]


def _http_backed_adapter(response, calls):
    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    api = matrix._create_matrix_http_api(
        base_url="https://example.invalid", token="",
        client_session=SimpleNamespace(request=request), default_retry_count=0)
    client = Client(api=api)
    client.state_store = SimpleNamespace(is_encrypted=_false_encryption)
    client.crypto = SimpleNamespace(share_keys=_unexpected_share)
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter.platform = matrix.Platform.MATRIX
    adapter._encryption = True
    adapter._client = client
    adapter.format_message = lambda text: text
    adapter.truncate_message = lambda text, _limit: [text]
    adapter.max_message_length = 1000
    adapter._build_text_message_content = lambda text: {"msgtype": "m.text", "body": text}
    adapter._apply_relation_metadata = lambda *args, **kwargs: None
    return adapter, client


async def _false_encryption(_room):
    return False


async def _unexpected_share():
    pytest.fail("no key share is allowed after a possible HTTP send")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_json", [False, True], ids=["missing-event-id", "malformed-json"])
async def test_accepted_send_with_unusable_response_never_creates_another_transaction(bad_json):
    calls = []
    response = Response(200)
    if bad_json:
        async def malformed_json():
            raise JSONDecodeError("invalid success JSON", "{", 1)
        response.json = malformed_json
    else:
        response.data = {}
    adapter, _client = _http_backed_adapter(response, calls)
    result = await adapter._send_with_retry("!r", "hello", max_retries=2, base_delay=0)
    assert not result.success
    assert result.raw_response == {"matrix_send_final": True, "matrix_send_disposition": "unknown"}
    assert len(calls) == 1
    assert "/send/" in str(calls[0][0][1])


@pytest.mark.asyncio
async def test_protocol_rejection_does_not_trigger_key_share_or_fallback():
    calls = []
    response = Response(403)
    response.data = {"errcode": "M_FORBIDDEN", "error": "forbidden"}
    adapter, _client = _http_backed_adapter(response, calls)
    result = await adapter._send_with_retry("!r", "hello", max_retries=2, base_delay=0)
    assert not result.success
    assert result.raw_response == {"matrix_send_final": True, "matrix_send_disposition": "rejected"}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pre_send_crypto_failure_can_recover_after_key_share():
    calls = []
    adapter, client = _http_backed_adapter(Response(200), calls)
    crypto_steps = []
    class Crypto:
        async def encrypt_megolm_event(self, *_args):
            crypto_steps.append("encrypt")
            if len(crypto_steps) < 4:
                raise EncryptionError("no group session")
            return {"algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "local"}

        async def share_group_session(self, *_args):
            crypto_steps.append("group-share")

        async def share_keys(self):
            crypto_steps.append("key-share")

    client.crypto = Crypto()
    client.state_store = SimpleNamespace(
        is_encrypted=_true_encryption, has_full_member_list=_true_encryption,
        get_members=_empty_members)
    result = await adapter._send_with_retry("!r", "hello", max_retries=2, base_delay=0)
    assert result.success and result.message_id == "$ok"
    assert crypto_steps == ["encrypt", "group-share", "encrypt", "key-share", "encrypt"]
    assert len(calls) == 1  # the first send fails during encryption, before HTTP


async def _true_encryption(_room):
    return True


async def _empty_members(_room):
    return []


@pytest.mark.asyncio
async def test_disconnect_attempts_session_close_after_crypto_stop_failure_and_retains_failed_owner():
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._sync_task = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._opening_session = None
    calls = []
    class CryptoDB:
        async def stop(self):
            calls.append("stop")
            if calls.count("stop") == 1:
                raise RuntimeError("crypto stop failed")
    class Session:
        async def close(self):
            calls.append("close")
    db, session = CryptoDB(), Session()
    adapter._crypto_db = db
    adapter._client = SimpleNamespace(api=SimpleNamespace(session=session))
    with pytest.raises(RuntimeError, match="crypto stop failed"):
        await adapter._disconnect_impl()
    assert calls == ["stop", "close"]
    assert adapter._crypto_db is db and adapter._client is None
    await adapter._disconnect_impl()
    assert calls == ["stop", "close", "stop"]
    assert adapter._crypto_db is None


@pytest.mark.asyncio
async def test_disconnect_reports_both_failures_and_retains_both_owners():
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._sync_task = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._opening_session = None
    calls = []
    class CryptoDB:
        async def stop(self):
            calls.append("stop")
            raise RuntimeError("stop failed")
    class Session:
        async def close(self):
            calls.append("close")
            raise RuntimeError("close failed")
    db, session = CryptoDB(), Session()
    client = SimpleNamespace(api=SimpleNamespace(session=session))
    adapter._crypto_db, adapter._client = db, client
    with pytest.raises(ExceptionGroup) as raised:
        await adapter._disconnect_impl()
    assert [str(exc) for exc in raised.value.exceptions] == ["stop failed", "close failed"]
    assert calls == ["stop", "close"]
    assert adapter._crypto_db is db and adapter._client is client


@pytest.mark.asyncio
async def test_reconnect_settles_or_retains_orphaned_crypto_before_new_session(monkeypatch):
    adapter = object.__new__(matrix.MatrixAdapter)
    adapter._lifecycle_lock = asyncio.Lock()
    adapter._sync_task = adapter._opening_session = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._homeserver = "https://example.invalid"
    adapter._proxy_url = None
    adapter._resolve_store_dir = lambda: SimpleNamespace(mkdir=lambda **kwargs: None)
    events = []
    class CryptoDB:
        async def stop(self):
            events.append("stop")
            if events.count("stop") <= 3:
                raise RuntimeError("stop still failed")
    db = CryptoDB()
    adapter._crypto_db = db
    class Session:
        async def close(self):
            events.append("close")
    adapter._client = SimpleNamespace(api=SimpleNamespace(session=Session()))
    def create_session(_proxy):
        events.append("new session")
        raise RuntimeError("session sentinel")
    monkeypatch.setattr(matrix, "_create_matrix_session", create_session)
    with pytest.raises(RuntimeError, match="stop still failed"):
        await adapter.disconnect()
    assert events == ["stop", "close"]
    assert adapter._crypto_db is db and adapter._client is None
    # The next connect and its defensive cleanup both fail to stop: neither may allocate.
    with pytest.raises(RuntimeError, match="stop still failed"):
        await adapter.connect(is_reconnect=True)
    assert events == ["stop", "close", "stop", "stop"]
    assert adapter._crypto_db is db and adapter._client is None
    # Once stop can settle, the next connect may allocate, never before release.
    with pytest.raises(RuntimeError, match="session sentinel"):
        await adapter.connect(is_reconnect=True)
    assert events == ["stop", "close", "stop", "stop", "stop", "new session"]
    assert adapter._crypto_db is None
