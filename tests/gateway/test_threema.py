"""Invariant tests for the Threema gateway platform plugin.

Covers plugins/platforms/threema — the E2E container, the callback MAC, byte-accurate
chunking, the file-message envelope, and the adapter's inbound decisions. No network:
the REST client takes an injected transport and the callback path is exercised through
``process_callback``, which is deliberately free of aiohttp so the security-critical
branches run in CI whether or not the optional SDKs are installed.

The protocol facts pinned here come from https://gateway.threema.ch/en/developer/e2e
and /developer/api.
"""
from __future__ import annotations

import binascii
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.threema import adapter as threema_adapter
from plugins.platforms.threema import api as threema_api
from plugins.platforms.threema import crypto

SECRET = "s3cr3t"
GATEWAY_ID = "*HERMES1"
SENDER_ID = "ECHOECHO"


@pytest.fixture
def keypair():
    """(gateway_private, gateway_public, sender_private, sender_public).

    PyNaCl is an optional platform dependency, so only the tests that actually need a
    box skip without it — the padding, MAC, chunking and registration contracts still
    run on a bare install.
    """
    PrivateKey = pytest.importorskip("nacl.public", reason="PyNaCl not installed").PrivateKey
    ours, theirs = PrivateKey.generate(), PrivateKey.generate()
    return bytes(ours), bytes(ours.public_key), bytes(theirs), bytes(theirs.public_key)


# ---------------------------------------------------------------------------
# Fake transports
# ---------------------------------------------------------------------------


@dataclass
class FakeResponse:
    status_code: int = 200
    text: str = ""
    content: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTP:
    """Records (method, url, kwargs) and replies from a {path: response} table."""

    def __init__(self, routes: Dict[str, FakeResponse]):
        self.routes = routes
        self.calls = []

    async def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        path = url.split("msgapi.threema.ch", 1)[-1]
        self.calls.append((method, path, kwargs))
        for prefix, response in self.routes.items():
            if path.startswith(prefix):
                return response
        return FakeResponse(status_code=404, text="not found")

    async def aclose(self) -> None:  # pragma: no cover - client is injected, never owned
        pass


def make_client(routes: Dict[str, FakeResponse]) -> tuple:
    http = FakeHTTP(routes)
    return threema_api.ThreemaClient(GATEWAY_ID, SECRET, client=http), http


def make_adapter(keypair, tmp_path, routes=None, **extra) -> threema_adapter.ThreemaAdapter:
    ours, _ours_pub, _theirs, theirs_pub = keypair
    key_file = tmp_path / "threema.key"
    key_file.write_text("private:" + binascii.hexlify(ours).decode(), encoding="utf-8")
    config = PlatformConfig(enabled=True, extra={
        "gateway_id": GATEWAY_ID, "api_secret": SECRET,
        "private_key_path": str(key_file), **extra,
    })
    adapter = threema_adapter.ThreemaAdapter(config)
    adapter._private_key = crypto.load_private_key(str(key_file))
    client, http = make_client(routes or {})
    client.cache_public_key(SENDER_ID, theirs_pub)
    adapter._client = client
    adapter._http = http
    return adapter


def callback_form(box: bytes, nonce: bytes, *, secret: str = SECRET, message_id: str = "0011223344556677") -> Dict[str, str]:
    form = {
        "from": SENDER_ID, "to": GATEWAY_ID, "messageId": message_id, "date": "1758400000",
        "nonce": binascii.hexlify(nonce).decode(), "box": binascii.hexlify(box).decode(),
        "nickname": "Echo",
    }
    form["mac"] = crypto.callback_mac(form, secret)
    return form


# ---------------------------------------------------------------------------
# Plugin packaging
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_without_core_changes():
    """The adapter is a bundled plugin: Platform("threema") comes from the directory scan."""
    assert Platform("threema").value == "threema"
    assert Platform("threema") is Platform("threema")


def test_plugin_manifest_declares_its_env_contract():
    import yaml
    from pathlib import Path
    manifest = yaml.safe_load((Path(threema_adapter.__file__).parent / "plugin.yaml").read_text())
    assert manifest["kind"] == "platform"
    required = {entry["name"] for entry in manifest["requires_env"]}
    assert required == {"THREEMA_GATEWAY_ID", "THREEMA_API_SECRET", "THREEMA_PRIVATE_KEY_PATH"}
    secrets = {e["name"] for e in manifest["requires_env"] + manifest["optional_env"] if e["password"]}
    assert "THREEMA_API_SECRET" in secrets and "THREEMA_PRIVATE_KEY" in secrets


def test_lazy_deps_entry_uses_pynacl_not_the_system_libsodium_sdk():
    from tools.lazy_deps import LAZY_DEPS
    packages = LAZY_DEPS["platform.threema"]
    assert any(pkg.startswith("pynacl==") for pkg in packages)
    assert not any("threema" in pkg or "libnacl" in pkg for pkg in packages)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_key_parsing_accepts_panel_format_and_bare_hex(keypair):
    ours = keypair[0]
    hex_key = binascii.hexlify(ours).decode()
    assert crypto.parse_key(f"private:{hex_key}") == ours
    assert crypto.parse_key(hex_key) == ours
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.parse_key(f"public:{hex_key}", expect="private")
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.parse_key("private:notahexkey")
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.parse_key("private:" + "ab" * 16)  # 16 bytes, not 32


def test_private_key_is_read_from_a_file_so_it_stays_out_of_the_environment(tmp_path, keypair):
    ours = keypair[0]
    path = tmp_path / "gateway.key"
    path.write_text(f"private:{binascii.hexlify(ours).decode()}\n", encoding="utf-8")
    assert crypto.load_private_key(str(path)) == ours
    assert crypto.derive_public_key(ours) == keypair[1]
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.load_private_key("")


# ---------------------------------------------------------------------------
# Container: padding and boxes
# ---------------------------------------------------------------------------


def test_padding_is_pkcs7_and_never_shorter_than_32_bytes():
    """Threema widens the random pad so short messages do not leak their length."""
    padded = crypto.pad(b"hi", pad_length=3)
    assert len(padded) == 32  # 2 + 3 would be 5; the spec bumps it to exactly 32
    assert padded[-1] == 30 and crypto.unpad(padded) == b"hi"

    long_enough = crypto.pad(b"x" * 100, pad_length=7)
    assert long_enough == b"x" * 100 + bytes([7]) * 7
    assert crypto.unpad(long_enough) == b"x" * 100

    for _ in range(20):  # random path stays within spec
        blob = crypto.pad(b"y" * 40)
        assert 1 <= blob[-1] <= 255 and len(blob) >= 32
        assert crypto.unpad(blob) == b"y" * 40


def test_unpad_rejects_a_zero_or_oversized_length():
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.unpad(b"")
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.unpad(b"abc\x00")
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.unpad(b"ab\x09")


def test_container_round_trip_carries_the_type_byte(keypair):
    ours, ours_pub, theirs, theirs_pub = keypair
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, "grüezi 👋".encode(), ours, theirs_pub)
    assert len(nonce) == crypto.NONCE_LENGTH
    kind, inner = crypto.decrypt_container(box, nonce, theirs, ours_pub)
    assert kind == crypto.TYPE_TEXT
    assert inner.decode() == "grüezi 👋"


def test_container_decrypt_fails_with_the_wrong_key(keypair):
    PrivateKey = pytest.importorskip("nacl.public").PrivateKey
    ours, _ours_pub, theirs, theirs_pub = keypair
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, b"secret", ours, theirs_pub)
    stranger = bytes(PrivateKey.generate().public_key)
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.decrypt_container(box, nonce, theirs, stranger)


def test_blob_crypto_uses_the_protocol_fixed_nonces(keypair):
    del keypair  # only here to gate on PyNaCl
    key = crypto.random_blob_key()
    payload = b"a file" * 100
    assert crypto.BLOB_FILE_NONCE == b"\x00" * 23 + b"\x01"
    assert crypto.BLOB_THUMBNAIL_NONCE == b"\x00" * 23 + b"\x02"
    assert crypto.decrypt_blob(crypto.encrypt_blob(payload, key), key) == payload
    # A thumbnail encrypted under the thumbnail nonce must not open as a file blob.
    with pytest.raises(crypto.ThreemaCryptoError):
        crypto.decrypt_blob(crypto.encrypt_blob(payload, key, thumbnail=True), key)


# ---------------------------------------------------------------------------
# Callback MAC
# ---------------------------------------------------------------------------


def test_callback_mac_matches_the_documented_field_order():
    fields = {"from": "ECHOECHO", "to": "*HERMES1", "messageId": "0102030405060708",
              "date": "1758400000", "nonce": "aa" * 24, "box": "bb" * 40, "nickname": "ignored"}
    import hashlib
    import hmac as _hmac
    expected = _hmac.new(SECRET.encode(), b"ECHOECHO*HERMES101020304050607081758400000" + b"aa" * 24 + b"bb" * 40,
                         hashlib.sha256).hexdigest()
    assert crypto.callback_mac(fields, SECRET) == expected
    assert crypto.verify_callback(dict(fields, mac=expected), SECRET)


def test_callback_verification_rejects_tampering_and_missing_macs():
    fields = {"from": "ECHOECHO", "to": "*HERMES1", "messageId": "01", "date": "1", "nonce": "aa", "box": "bb"}
    good = crypto.callback_mac(fields, SECRET)
    assert crypto.verify_callback(dict(fields, mac=good), SECRET)
    assert not crypto.verify_callback(dict(fields, mac=good, box="cc"), SECRET)   # body swapped
    assert not crypto.verify_callback(dict(fields, mac=good), "other-secret")     # wrong secret
    assert not crypto.verify_callback(dict(fields), SECRET)                       # no mac at all
    assert not crypto.verify_callback(dict(fields, mac=good), "")                 # unconfigured secret


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_e2e_posts_hex_encoded_nonce_and_box():
    client, http = make_client({"/send_e2e": FakeResponse(text="0ce8e0e0b0f1d5c1\n")})
    message_id = await client.send_e2e(SENDER_ID, b"\x01" * 24, b"\x02" * 50)
    assert message_id == "0ce8e0e0b0f1d5c1"
    method, path, kwargs = http.calls[-1]
    assert (method, path) == ("POST", "/send_e2e")
    data = kwargs["data"]
    assert data["from"] == GATEWAY_ID and data["to"] == SENDER_ID and data["secret"] == SECRET
    assert data["nonce"] == "01" * 24 and data["box"] == "02" * 50
    assert "noDeliveryReceipts" not in data and "noPush" not in data


@pytest.mark.asyncio
async def test_send_e2e_refuses_an_oversized_box_before_spending_a_credit():
    client, http = make_client({"/send_e2e": FakeResponse(text="id")})
    with pytest.raises(threema_api.ThreemaAPIError) as exc:
        await client.send_e2e(SENDER_ID, b"\x00" * 24, b"\x00" * (threema_api.MAX_BOX_BYTES + 1))
    assert exc.value.status == 413
    assert http.calls == []


@pytest.mark.asyncio
async def test_status_codes_carry_an_actionable_hint():
    client, _http = make_client({"/credits": FakeResponse(status_code=402, text="")})
    with pytest.raises(threema_api.ThreemaAPIError) as exc:
        await client.credits()
    assert exc.value.out_of_credits and "credits" in str(exc.value).lower()

    client, _http = make_client({"/send_e2e": FakeResponse(status_code=429)})
    with pytest.raises(threema_api.ThreemaAPIError) as exc:
        await client.send_e2e(SENDER_ID, b"\x00" * 24, b"\x00" * 10)
    assert exc.value.rate_limited


@pytest.mark.asyncio
async def test_api_secret_never_reaches_an_error_message():
    client, _http = make_client({"/credits": FakeResponse(status_code=401)})
    with pytest.raises(threema_api.ThreemaAPIError) as exc:
        await client.credits()
    assert SECRET not in str(exc.value)


@pytest.mark.asyncio
async def test_public_keys_are_cached_because_threema_asks_callers_to_cache(keypair):
    theirs_pub = keypair[3]
    routes = {"/pubkeys/": FakeResponse(text=binascii.hexlify(theirs_pub).decode())}
    client, http = make_client(routes)
    assert await client.public_key(SENDER_ID) == theirs_pub
    assert await client.public_key(SENDER_ID.lower()) == theirs_pub  # normalized, still cached
    assert len([c for c in http.calls if c[1].startswith("/pubkeys/")]) == 1
    await client.public_key(SENDER_ID, refresh=True)
    assert len([c for c in http.calls if c[1].startswith("/pubkeys/")]) == 2


@pytest.mark.asyncio
async def test_blob_upload_sends_auth_in_the_query_and_the_blob_as_multipart():
    client, http = make_client({"/upload_blob": FakeResponse(text="0a1b2c3d")})
    assert await client.upload_blob(b"payload") == "0a1b2c3d"
    _method, path, kwargs = http.calls[-1]
    assert path.startswith("/upload_blob")
    assert kwargs["params"] == {"from": GATEWAY_ID, "secret": SECRET}
    assert "blob" in kwargs["files"] and "data" not in kwargs


def test_identity_validation():
    assert threema_api.valid_identity("ECHOECHO")
    assert threema_api.valid_identity("*HERMES1")
    assert threema_api.valid_identity("echoecho")       # normalized: users type lowercase
    assert not threema_api.valid_identity("SHORT")      # 8 characters exactly
    assert not threema_api.valid_identity("ECHOECHO9X")
    assert not threema_api.valid_identity("ECHO-ECH")   # only A-Z0-9, and * only in front
    assert not threema_api.valid_identity("E*HOECHO")
    assert not threema_api.valid_identity("")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunking_counts_bytes_not_characters():
    """3500 emoji are 14,000 bytes; Threema's cap is on bytes, so they must split."""
    emoji = "🔒" * 1200  # 4800 bytes, 1200 characters
    chunks = threema_adapter._text_chunks(emoji)
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= threema_adapter.MAX_TEXT_BYTES for chunk in chunks)
    assert "".join(chunks) == emoji  # nothing dropped, no code point split


def test_chunking_leaves_a_short_message_alone_and_prefers_line_breaks():
    assert threema_adapter._text_chunks("hello") == ["hello"]
    assert threema_adapter._text_chunks("") == []
    text = ("a" * 3000) + "\n" + ("b" * 1000)
    chunks = threema_adapter._text_chunks(text)
    assert chunks[0] == "a" * 3000 and chunks[1] == "b" * 1000


# ---------------------------------------------------------------------------
# Adapter: inbound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_text_becomes_a_message_event(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, "hoi Hermes".encode(), theirs, ours_pub)
    status, event = await adapter.process_callback(callback_form(box, nonce))
    assert status == 200
    assert event.text == "hoi Hermes"
    assert event.source.chat_id == SENDER_ID and event.source.chat_type == "dm"
    assert event.user_name == "Echo"


@pytest.mark.asyncio
async def test_a_forged_callback_is_rejected_before_any_decryption(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, b"hello", theirs, ours_pub)
    form = callback_form(box, nonce, secret="not-the-secret")
    status, event = await adapter.process_callback(form)
    assert (status, event) == (401, None)


@pytest.mark.asyncio
async def test_a_retried_callback_is_delivered_once(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, b"only once", theirs, ours_pub)
    form = callback_form(box, nonce)
    first_status, first_event = await adapter.process_callback(form)
    second_status, second_event = await adapter.process_callback(form)
    assert first_status == 200 and first_event is not None
    # Threema retries a failed callback 3x at 5-minute intervals — the ack must stay 200
    # so the retries stop, but the message must not reach the agent twice.
    assert (second_status, second_event) == (200, None)
    assert threema_adapter.DEDUP_TTL_SECONDS > 3 * 5 * 60


@pytest.mark.asyncio
async def test_a_delivery_receipt_never_wakes_the_agent(keypair, tmp_path):
    """0x80 is the recipient's read/received checkmark, not a user turn."""
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    inner = bytes([crypto.RECEIPT_READ]) + b"\x01" * 8
    nonce, box = crypto.encrypt_container(crypto.TYPE_DELIVERY_RECEIPT, inner, theirs, ours_pub)
    status, event = await adapter.process_callback(callback_form(box, nonce))
    assert (status, event) == (200, None)


@pytest.mark.asyncio
async def test_an_undecryptable_but_authentic_callback_is_acked_not_retried(keypair, tmp_path):
    adapter = make_adapter(keypair, tmp_path)
    form = callback_form(b"\x00" * 64, b"\x00" * 24)  # MAC is valid, the box is junk
    status, event = await adapter.process_callback(form)
    assert (status, event) == (200, None)


@pytest.mark.asyncio
async def test_a_transient_api_failure_asks_threema_to_retry(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    adapter._client._pubkeys.clear()  # force a /pubkeys lookup, which will fail
    adapter._client._client = FakeHTTP({"/pubkeys/": FakeResponse(status_code=500)})
    nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, b"hi", theirs, ours_pub)
    status, event = await adapter.process_callback(callback_form(box, nonce))
    assert (status, event) == (500, None)


@pytest.mark.asyncio
async def test_inbound_file_is_downloaded_decrypted_and_cached(keypair, tmp_path, monkeypatch):
    ours_pub, theirs = keypair[1], keypair[2]
    blob_key = crypto.random_blob_key()
    image = b"\xff\xd8\xff" + b"pretend jpeg" * 10
    encrypted = crypto.encrypt_blob(image, blob_key)
    adapter = make_adapter(keypair, tmp_path)
    adapter._client._client = FakeHTTP({"/blobs/": FakeResponse(content=encrypted)})

    cached = {}

    async def _fake_cache(data, ext=".jpg"):
        cached["data"], cached["ext"] = data, ext
        return str(tmp_path / f"cached{ext}")

    monkeypatch.setattr(threema_adapter, "cache_image_from_bytes_async", _fake_cache)
    meta = {"j": 1, "i": 1, "k": binascii.hexlify(blob_key).decode(), "b": "cafebabe",
            "m": "image/jpeg", "n": "photo.jpg", "s": len(image), "d": "look at this"}
    nonce, box = crypto.encrypt_container(
        crypto.TYPE_FILE, json.dumps(meta).encode(), theirs, ours_pub)
    status, event = await adapter.process_callback(callback_form(box, nonce))
    assert status == 200
    assert cached["data"] == image  # decrypted with the key from the container, not the blob
    assert event.media_urls == [str(tmp_path / "cached.jpg")]
    assert event.text == "look at this"


@pytest.mark.asyncio
async def test_inbound_location_is_rendered_as_text(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    inner = "47.3769,8.5417,12.5\nHauptbahnhof\nBahnhofplatz, 8001 Zürich".encode()
    nonce, box = crypto.encrypt_container(crypto.TYPE_LOCATION, inner, theirs, ours_pub)
    _status, event = await adapter.process_callback(callback_form(box, nonce))
    assert "47.3769" in event.text and "8.5417" in event.text
    assert "Hauptbahnhof" in event.text


# ---------------------------------------------------------------------------
# Adapter: outbound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_encrypts_per_recipient_and_returns_the_message_id(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    http = FakeHTTP({"/send_e2e": FakeResponse(text="abcd1234")})
    adapter._client._client = http
    result = await adapter.send(SENDER_ID, "grüezi")
    assert result.success and result.message_id == "abcd1234"
    data = http.calls[-1][2]["data"]
    box = binascii.unhexlify(data["box"])
    nonce = binascii.unhexlify(data["nonce"])
    kind, inner = crypto.decrypt_container(box, nonce, theirs, ours_pub)
    assert (kind, inner.decode()) == (crypto.TYPE_TEXT, "grüezi")


@pytest.mark.asyncio
async def test_a_long_answer_is_split_and_every_part_is_reported(keypair, tmp_path):
    adapter = make_adapter(keypair, tmp_path)
    adapter._client._client = FakeHTTP({"/send_e2e": FakeResponse(text="id")})
    result = await adapter.send(SENDER_ID, "x" * 9000)
    assert result.success
    assert len(result.continuation_message_ids) == 2  # 3 sends -> 3 credits
    assert len([c for c in adapter._client._client.calls if c[1] == "/send_e2e"]) == 3


@pytest.mark.asyncio
async def test_send_rejects_an_invalid_recipient_without_calling_the_api(keypair, tmp_path):
    adapter = make_adapter(keypair, tmp_path)
    http = FakeHTTP({"/send_e2e": FakeResponse(text="id")})
    adapter._client._client = http
    result = await adapter.send("not-an-id", "hello")
    assert not result.success and "not a valid Threema ID" in result.error
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_rate_limited_send_is_marked_retryable(keypair, tmp_path):
    adapter = make_adapter(keypair, tmp_path)
    adapter._client._client = FakeHTTP({"/send_e2e": FakeResponse(status_code=429)})
    result = await adapter.send(SENDER_ID, "hello")
    assert not result.success and result.retryable


@pytest.mark.asyncio
async def test_sending_a_file_uploads_an_encrypted_blob_and_references_it(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    http = FakeHTTP({"/upload_blob": FakeResponse(text="feedface"), "/send_e2e": FakeResponse(text="msg-1")})
    adapter._client._client = http
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.7 report body")

    result = await adapter.send_document(SENDER_ID, str(document), caption="the report")
    assert result.success and result.message_id == "msg-1"

    upload = next(c for c in http.calls if c[1].startswith("/upload_blob"))
    uploaded = upload[2]["files"]["blob"][1]
    assert uploaded != b"%PDF-1.7 report body"  # the blob is encrypted before it leaves

    data = http.calls[-1][2]["data"]
    kind, inner = crypto.decrypt_container(
        binascii.unhexlify(data["box"]), binascii.unhexlify(data["nonce"]), theirs, ours_pub)
    assert kind == crypto.TYPE_FILE
    meta = json.loads(inner)
    assert meta["b"] == "feedface" and meta["m"] == "application/pdf" and meta["n"] == "report.pdf"
    assert meta["s"] == len(b"%PDF-1.7 report body") and meta["d"] == "the report"
    assert meta["j"] == 0 and meta["i"] == 0  # a document renders as a file, not as media
    assert crypto.decrypt_blob(uploaded, binascii.unhexlify(meta["k"])) == b"%PDF-1.7 report body"


@pytest.mark.asyncio
async def test_an_image_is_sent_as_media_rendering(keypair, tmp_path):
    ours_pub, theirs = keypair[1], keypair[2]
    adapter = make_adapter(keypair, tmp_path)
    adapter._client._client = FakeHTTP({"/upload_blob": FakeResponse(text="b10b"), "/send_e2e": FakeResponse(text="m")})
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert (await adapter.send_image_file(SENDER_ID, str(image))).success
    data = adapter._client._client.calls[-1][2]["data"]
    _kind, inner = crypto.decrypt_container(
        binascii.unhexlify(data["box"]), binascii.unhexlify(data["nonce"]), theirs, ours_pub)
    meta = json.loads(inner)
    assert meta["j"] == 1 and meta["i"] == 1 and meta["m"] == "image/png"


@pytest.mark.asyncio
async def test_typing_indicator_is_a_no_op_because_it_would_cost_a_credit(keypair, tmp_path):
    adapter = make_adapter(keypair, tmp_path)
    http = FakeHTTP({})
    adapter._client._client = http
    assert await adapter.send_typing(SENDER_ID) is None
    assert http.calls == []


# ---------------------------------------------------------------------------
# Configuration and registration
# ---------------------------------------------------------------------------


def test_missing_configuration_is_reported_by_name(tmp_path):
    adapter = threema_adapter.ThreemaAdapter(PlatformConfig(enabled=True, extra={}))
    missing = adapter._missing_config()
    assert "THREEMA_GATEWAY_ID" in missing and "THREEMA_API_SECRET" in missing
    assert any("PRIVATE_KEY" in item for item in missing)

    bad_id = threema_adapter.ThreemaAdapter(PlatformConfig(enabled=True, extra={
        "gateway_id": "nope", "api_secret": "s", "private_key_path": "/tmp/k"}))
    assert any("not an 8-character Threema ID" in item for item in bad_id._missing_config())


def test_validate_config_needs_all_three_credentials(monkeypatch):
    for var in ("THREEMA_GATEWAY_ID", "THREEMA_API_SECRET", "THREEMA_PRIVATE_KEY_PATH", "THREEMA_PRIVATE_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not threema_adapter.validate_config(PlatformConfig(extra={"gateway_id": GATEWAY_ID}))
    full = PlatformConfig(extra={"gateway_id": GATEWAY_ID, "api_secret": SECRET, "private_key_path": "/k"})
    assert threema_adapter.validate_config(full) and threema_adapter.is_connected(full)


def test_env_enablement_seeds_extra_for_an_env_only_setup(monkeypatch):
    monkeypatch.setenv("THREEMA_GATEWAY_ID", GATEWAY_ID)
    monkeypatch.setenv("THREEMA_API_SECRET", SECRET)
    monkeypatch.setenv("THREEMA_PUBLIC_URL", "https://hermes.example.com")
    monkeypatch.setenv("THREEMA_CALLBACK_PORT", "9443")
    seeded = threema_adapter._env_enablement()
    assert seeded["gateway_id"] == GATEWAY_ID and seeded["public_url"] == "https://hermes.example.com"
    assert seeded["callback_port"] == 9443
    monkeypatch.setenv("THREEMA_CALLBACK_PORT", "not-a-port")
    assert "callback_port" not in threema_adapter._env_enablement()
    monkeypatch.delenv("THREEMA_GATEWAY_ID")
    assert threema_adapter._env_enablement() is None


def test_register_declares_the_hooks_the_gateway_needs():
    captured = {}

    class Ctx:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    threema_adapter.register(Ctx())
    assert captured["name"] == "threema"
    assert captured["check_fn"] is threema_adapter.check_requirements
    assert captured["ensure_deps_fn"] is threema_adapter.ensure_requirements
    assert captured["standalone_sender_fn"] is threema_adapter._standalone_send
    assert captured["cron_deliver_env_var"] == "THREEMA_HOME_CHANNEL"
    assert captured["allowed_users_env"] == "THREEMA_ALLOWED_USERS"
    assert captured["max_message_length"] == threema_adapter.MAX_TEXT_BYTES
    # The hint has to warn about markdown and about per-message cost.
    hint = captured["platform_hint"].lower()
    assert "markdown" in hint and "credit" in hint


@pytest.mark.asyncio
async def test_standalone_send_works_without_a_running_gateway(keypair, tmp_path, monkeypatch):
    """A detached cron job only needs the key pair and the secret — no callback listener."""
    ours_pub, theirs = keypair[1], keypair[2]
    key_file = tmp_path / "k.key"
    key_file.write_text("private:" + binascii.hexlify(keypair[0]).decode(), encoding="utf-8")
    http = FakeHTTP({
        "/pubkeys/": FakeResponse(text=binascii.hexlify(keypair[3]).decode()),
        "/send_e2e": FakeResponse(text="cron-1"),
    })
    real_client_cls = threema_api.ThreemaClient
    monkeypatch.setattr(threema_adapter, "ThreemaClient",
                        lambda ident, secret, **kw: real_client_cls(ident, secret, client=http))
    config = PlatformConfig(extra={"gateway_id": GATEWAY_ID, "api_secret": SECRET,
                                   "private_key_path": str(key_file)})
    result = await threema_adapter._standalone_send(config, SENDER_ID, "nightly digest")
    assert result == {"success": True, "message_id": "cron-1"}

    bad = await threema_adapter._standalone_send(config, "nope", "x")
    assert "not a valid Threema ID" in bad["error"]


def test_config_yaml_reaches_the_adapter_end_to_end(tmp_path, monkeypatch):
    """Real loader, real registry, real adapter — no mocks in the resolution chain.

    A plugin platform is only usable if ``gateway.platforms.threema`` in config.yaml
    lands in ``PlatformConfig.extra``, the platform shows up as connected, and the
    adapter reads its own settings back. Unit-mocking any of those three steps hides
    a wiring break.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    threema:\n"
        "      enabled: true\n"
        "      extra:\n"
        '        gateway_id: "*HERMES1"\n'
        '        private_key_path: "key.private"\n'
        '        public_url: "https://hermes.example.com"\n'
        "        callback_port: 9443\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("THREEMA_API_SECRET", SECRET)

    from hermes_cli.plugins import discover_plugins
    from gateway.config import load_gateway_config
    from gateway.platform_registry import platform_registry

    discover_plugins(force=True)
    assert platform_registry.is_registered("threema")
    entry = platform_registry.get("threema")
    assert entry.source == "plugin" and entry.plugin_name == "threema-platform"

    config = load_gateway_config()
    platform_config = config.platforms[Platform("threema")]
    assert platform_config.enabled
    assert Platform("threema") in config.get_connected_platforms()

    adapter = threema_adapter.ThreemaAdapter(platform_config)
    assert adapter._port == 9443                      # extra.callback_port, not the default
    assert adapter._path == "/threema/callback"
    assert adapter._public_url == "https://hermes.example.com"
    assert adapter._missing_config() == []            # secret came from the environment
