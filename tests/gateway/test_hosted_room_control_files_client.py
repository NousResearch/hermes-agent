"""Real hostile HTTP framing and credential-boundary tests."""

import base64
import copy
import hashlib
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gateway.hosted_room_control_client import RoomControlHTTPClient
from gateway.hosted_room_controls import StoredPeerRoomControl
from gateway.hosted_room_file_contract import FileAccessError


DATA = b"0123456789"
ATTACHMENT_ID = "att_" + "1" * 32
SCOPE = {
    "room_id": "room-1",
    "member_id": "peer",
    "target_profile": "reviewer",
    "authority_gateway_id": "install:home",
    "authority_epoch": 1,
}
METADATA = {
    "attachment_id": ATTACHMENT_ID,
    "event_id": "event-1",
    "kind": "file",
    "name": "result.txt",
    "mime": "text/plain",
    "size": len(DATA),
    "sha256": hashlib.sha256(DATA).hexdigest(),
}


def page():
    item = {key: value for key, value in METADATA.items() if key != "sha256"}
    item.update(
        seq=1,
        shared_at=1.0,
        producer={"kind": "member", "id": "peer", "label": "Reviewer"},
    )
    return {
        "items": [item],
        "has_more": False,
        "next_cursor": None,
        "snapshot_seq": 1,
        "authority": {"gateway_id": "install:home", "epoch": 1},
        "scope": dict(SCOPE),
    }


@contextmanager
def host(mode="ok", *, catalog=False, redirect_to=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", redirect_to or "/redirect-target")
                self.end_headers()
                return
            if mode == "old":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"old host: not found")
                return
            if mode in {"large_error", "slow_error"}:
                self.send_response(503)
                self.end_headers()
                try:
                    if mode == "large_error":
                        self.wfile.write(b"x" * 4097)
                    else:
                        for _index in range(10):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            time.sleep(0.08)
                except OSError:
                    pass
                return
            body = DATA
            expected = copy.deepcopy(SCOPE)
            metadata = copy.deepcopy(METADATA)
            if mode == "bool_epoch":
                expected["authority_epoch"] = True
            if mode.startswith("scope_"):
                field = mode.removeprefix("scope_")
                expected[field] = 2 if field == "authority_epoch" else "wrong"
            if mode == "event":
                metadata["event_id"] = "wrong-event"
            if mode == "id":
                metadata["attachment_id"] = "att_" + "2" * 32
            if mode == "digest":
                metadata["sha256"] = "0" * 64
            if mode == "name":
                metadata["name"] = "../private"
            if mode == "oversize_metadata":
                metadata["size"] = 15_000_001
            receipt = {"scope": expected, "attachment": metadata}
            if catalog:
                payload = page()
                payload["scope"] = expected
                if mode == "latest":
                    payload["latest_seq"] = 2
                if mode == "extra":
                    payload["items"][0]["path"] = "/private"
                if mode == "duplicate":
                    payload["items"] *= 2
                if mode == "newer":
                    payload["items"][0]["seq"] = 2
                if mode == "many":
                    payload["items"] *= 33
                body = json.dumps(payload).encode()
                if mode == "malformed":
                    body = b"not-json"
                if mode == "large_catalog":
                    body = b"x" * (128 * 1024 + 1)
            elif mode == "truncated":
                body = DATA[:3]
            elif mode == "overbody":
                body = DATA + b"x"
            self.send_response(200)
            self.send_header(
                "Content-Type", "application/json" if catalog else "text/plain"
            )
            if mode == "encoding":
                self.send_header("Content-Encoding", "gzip")
            if not catalog and mode != "missing_receipt":
                encoded = base64.urlsafe_b64encode(
                    json.dumps(receipt).encode()
                ).decode()
                self.send_header(
                    "X-Hermes-Room-File",
                    "invalid" if mode == "bad_receipt" else encoded,
                )
            if mode not in {"overbody", "slow"}:
                length = len(body)
                if mode == "truncated":
                    length = len(DATA)
                if mode == "overlength":
                    length = len(DATA) + 1
                self.send_header(
                    "Content-Length", "invalid" if mode == "bad_length" else str(length)
                )
            self.end_headers()
            try:
                if mode == "slow":
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.08)
                else:
                    self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    link = StoredPeerRoomControl(
        room_id="room-1",
        member_id="peer",
        target_profile="reviewer",
        home_url=f"http://127.0.0.1:{server.server_port}",
        transport_security="loopback",
        authority_gateway_id="install:home",
        authority_epoch=1,
        room_name="Files",
        member_count=2,
        control_token="A" * 43,
        status="active",
        created_at=1,
        updated_at=1,
        expires_at=time.time() + 300,
    )
    try:
        yield (
            RoomControlHTTPClient(
                link, timeout_seconds=0.2 if mode in {"slow", "slow_error"} else 5
            ),
            requests,
        )
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def read(client):
    return client.read_file(
        target_profile="reviewer", event_id="event-1", attachment_id=ATTACHMENT_ID
    )


def test_verified_binary_receipt_and_token_only_in_headers():
    with host() as (client, requests):
        result = read(client)
    assert result.data == DATA and result.attachment == METADATA
    assert len(requests) == 1
    path, headers = requests[0]
    assert "A" * 43 not in path
    assert headers["Authorization"] == "HermesRoomControl " + "A" * 43
    assert headers["X-Hermes-Room-Profile"] == "reviewer"


@pytest.mark.parametrize(
    "mode",
    [
        "scope_room_id",
        "scope_member_id",
        "scope_target_profile",
        "scope_authority_gateway_id",
        "scope_authority_epoch",
        "bool_epoch",
        "event",
        "id",
        "digest",
        "name",
        "oversize_metadata",
        "truncated",
        "overbody",
        "overlength",
        "bad_length",
        "encoding",
        "bad_receipt",
        "missing_receipt",
    ],
)
def test_invalid_binary_never_returns_partial_or_unverified_data(mode):
    with host(mode) as (client, _requests):
        with pytest.raises(FileAccessError) as error:
            read(client)
    assert "A" * 43 not in str(error.value)
    assert "../private" not in str(error.value)


def test_trickled_binary_has_a_total_deadline():
    with host("slow") as (client, _requests):
        started = time.monotonic()
        with pytest.raises(FileAccessError) as error:
            read(client)
        assert time.monotonic() - started < 2
    assert error.value.code == "file_timeout"


def test_no_redirect_is_followed_even_on_the_same_origin():
    with host("redirect") as (client, requests):
        with pytest.raises(FileAccessError):
            read(client)
    assert len(requests) == 1
    assert all("redirect-target" not in path for path, _headers in requests)


def test_cross_origin_redirect_cannot_send_token_to_another_server():
    with host() as (target, target_requests):
        with host("redirect", redirect_to=target.link.home_url) as (client, requests):
            with pytest.raises(FileAccessError):
                read(client)
    assert len(requests) == 1
    assert target_requests == []


def test_old_host_is_typed_unsupported():
    with host("old") as (client, _requests):
        for call in (
            lambda: read(client),
            lambda: client.list_files(target_profile="reviewer"),
        ):
            with pytest.raises(FileAccessError) as error:
                call()
            assert error.value.code == "file_access_unsupported"


def test_catalogue_response_is_validated():
    with host(catalog=True) as (client, _requests):
        assert client.list_files(target_profile="reviewer") == page()


def test_additive_latest_sequence_is_preserved_for_parent_catalogue_v2():
    with host("latest", catalog=True) as (client, _requests):
        assert client.list_files(target_profile="reviewer")["latest_seq"] == 2


@pytest.mark.parametrize("mode", ["large_error", "slow_error"])
def test_error_responses_have_the_same_body_and_time_bounds(mode):
    with host(mode) as (client, _requests):
        started = time.monotonic()
        with pytest.raises(FileAccessError):
            read(client)
        assert time.monotonic() - started < 2


@pytest.mark.parametrize(
    "mode",
    [
        "scope_room_id",
        "scope_target_profile",
        "extra",
        "duplicate",
        "newer",
        "many",
        "malformed",
        "large_catalog",
    ],
)
def test_malformed_catalogue_is_not_a_file_list(mode):
    with host(mode, catalog=True) as (client, _requests):
        with pytest.raises(FileAccessError):
            client.list_files(target_profile="reviewer")
