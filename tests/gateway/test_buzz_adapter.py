"""Tests for the Buzz platform adapter plugin."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.platforms.base import CachedMedia, MessageType
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/buzz/adapter.py under a unique module name
# (plugin_adapter_buzz) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same xdist worker.
_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter
hex_to_npub = _buzz_mod.hex_to_npub
npub_to_hex = _buzz_mod.npub_to_hex
_normalize_user_ref = _buzz_mod._normalize_user_ref
_cli_error_message = _buzz_mod._cli_error_message
_resolve_private_key = _buzz_mod._resolve_private_key
check_requirements = _buzz_mod.check_requirements
validate_config = _buzz_mod.validate_config
register = _buzz_mod.register
_env_enablement = _buzz_mod._env_enablement
_standalone_send = _buzz_mod._standalone_send

# Real key pair (Chip's public identity — public information, not a secret)
SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
OTHER_PUBKEY = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
# Real DM conversation as materialized by a hosted relay: `dms list` returns
# [] for it (#68871) while `channels list` shows it as name "DM", empty
# description, indistinguishable from a channel except via message p-tags.
DM_CHANNEL = "6468cc16-a114-4f23-8b8c-02c1655cbf6b"

_ENV_VARS = (
    "BUZZ_RELAY_URL",
    "BUZZ_PRIVATE_KEY",
    "BUZZ_CHANNELS",
    "BUZZ_HOME_CHANNEL",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "BUZZ_POLL_INTERVAL",
    "BUZZ_CLI_PATH",
    "BUZZ_CREDENTIALS_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Keep tests hermetic: no ambient Buzz env vars or real credentials."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _event(event_id, pubkey=OTHER_PUBKEY, content="hello", created_at=1000, kind=9):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": [["h", CHANNEL]],
    }


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = SELF_NPUB
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    # GatewayRunner installs this callback before intake starts. Attachment
    # tests are authorized by default and override the callback at the boundary.
    adapter.set_authorization_check(lambda *_args: True)
    return adapter


class _ScriptedCli:
    """Fake ``_run_cli`` that routes on the buzz subcommand and records calls."""

    def __init__(self):
        self.responses = {}  # (group, cmd) -> list of (code, stdout, stderr)
        self.calls = []

    def script(self, group, cmd, payload, code=0, stderr=""):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        self.responses.setdefault((group, cmd), []).append((code, stdout, stderr))

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        queue = self.responses.get((args[0], args[1]), [])
        if len(queue) > 1:
            return queue.pop(0)
        if queue:
            return queue[0]
        return 0, "[]", ""


# ── bech32 / identity helpers ─────────────────────────────────────────────


class TestBech32Helpers:

    def test_hex_to_npub_known_pair(self):
        assert hex_to_npub(SELF_PUBKEY) == SELF_NPUB

    def test_npub_to_hex_known_pair(self):
        assert npub_to_hex(SELF_NPUB) == SELF_PUBKEY


# ── Adapter init / config precedence ──────────────────────────────────────


class TestBuzzAdapterInit:


    def test_init_from_config_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://cfg.relay",
                "channels": ["ccc"],
                "poll_interval": 2,
                "home_channel": "ccc",
            },
        )
        adapter = BuzzAdapter(cfg)
        assert adapter.relay_url == "https://cfg.relay"
        assert adapter.channels == ["ccc"]
        assert adapter.poll_interval == 2.0
        assert adapter.home_channel == "ccc"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://env.relay")
        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://cfg.relay"}))
        assert adapter.relay_url == "https://env.relay"


# ── CLI error contract ────────────────────────────────────────────────────


class TestCliErrorContract:

    def test_parses_json_error(self):
        msg = _cli_error_message('{"error":"relay_error","message":"boom","retryable":false}', 2)
        assert "relay_error" in msg and "boom" in msg and "exit 2" in msg


# ── Seeding / high-water mark / de-dupe ───────────────────────────────────


class TestPollingDedupe:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_seed_sets_high_water_mark_without_dispatch(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event("e1", content="@Chip old history", created_at=100),
            _event("e2", content="@Chip newer history", created_at=200),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        state = adapter._channel_state[CHANNEL]
        assert state["last_ts"] == 200
        assert set(state["seen"]) == {"e1", "e2"}
        # Seeding must never replay history into the agent
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_new_event_dispatched_once(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [_event("e1", content="@Chip hi", created_at=100)])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        # Poll 1: seeded event + a genuinely new mention
        cli.responses.clear()
        cli.script("messages", "get", [
            _event("e1", content="@Chip hi", created_at=100),
            _event("e2", content="hey @Chip, ping", created_at=150),
        ])
        await adapter._poll_channel(CHANNEL)
        assert [d["message_id"] for d in adapter._dispatched] == ["e2"]
        assert adapter._dispatched[0]["text"] == "hey @Chip, ping"
        assert adapter._channel_state[CHANNEL]["last_ts"] == 150

        # Poll 2: identical response — the seen-id set must de-dupe
        await adapter._poll_channel(CHANNEL)
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_malformed_attachment_url_does_not_abort_following_event(self, adapter):
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        malformed = _event("malformed-attachment", content="background chatter", created_at=159)
        malformed["tags"].append([
            "imeta",
            "url https://[invalid/media.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename invalid.bin",
        ])
        valid = _event("following-valid", content="@Chip still there?", created_at=160)
        cli = _ScriptedCli()
        cli.script("messages", "get", [malformed, valid])
        adapter._run_cli = cli

        await adapter._poll_channel(CHANNEL)

        assert [item["message_id"] for item in adapter._dispatched] == ["following-valid"]

    @pytest.mark.asyncio
    async def test_addressed_attachment_is_cached_and_dispatched_to_agent(self, adapter):
        attachment = CachedMedia(
            path="/agent/cache/doc_handoff.txt",
            media_type="text/plain",
            kind="document",
            display_name="handoff.txt",
        )
        adapter._download_attachment = AsyncMock(return_value=attachment)
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("attachment-event", content="@Chip inspect this", created_at=160)
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/abc.bin",
            "m text/plain",
            "x " + "a" * 64,
            "size 12",
            "filename handoff.txt",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        adapter._download_attachment.assert_awaited_once()
        dispatched = adapter._dispatched[-1]
        assert dispatched["media_urls"] == [attachment.path]
        assert dispatched["media_types"] == ["text/plain"]
        assert dispatched["message_type"] is MessageType.DOCUMENT
        assert attachment.context_note() not in dispatched["text"]
        assert dispatched["raw_message"] is event


class TestInboundAttachments:

    def test_imeta_total_declared_bytes_are_bounded(self):
        event = _event("bounded", content="@Chip files")
        per_file_size = 6 * 1024 * 1024
        for index in range(4):
            event["tags"].append([
                "imeta",
                f"url https://test.relay/media/{index}.bin",
                "m application/octet-stream",
                "x " + format(index + 1, "064x"),
                f"size {per_file_size}",
                f"filename {index}.bin",
            ])

        attachments = BuzzAdapter._imeta_attachments(event)

        assert len(attachments) == 3
        assert sum(item["size"] for item in attachments) <= 20 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_download_caches_only_exact_size_and_sha256(self, monkeypatch):
        import httpx

        payload = b"verified Buzz attachment"
        digest = hashlib.sha256(payload).hexdigest()
        real_async_client = httpx.AsyncClient

        def handler(request):
            assert request.url.host == "test.relay"
            return httpx.Response(
                200,
                content=payload,
                headers={"content-length": str(len(payload))},
            )

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            ),
        )
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://test.relay/media/verified.bin",
            "sha256": digest,
            "size": len(payload),
            "filename": "verified.txt",
            "mime_type": "text/plain",
        })

        assert cached is not None
        assert cached.kind == "document"
        assert cached.media_type == "text/plain"
        assert Path(cached.path).read_bytes() == payload

    @pytest.mark.asyncio
    async def test_download_rejects_untrusted_attachment_host_before_network(self, monkeypatch):
        import httpx

        def must_not_create_client(**kwargs):
            raise AssertionError("network client must not be created")

        monkeypatch.setattr(httpx, "AsyncClient", must_not_create_client)
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://untrusted.example/media/file.bin",
            "sha256": "a" * 64,
            "size": 12,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is None

    @pytest.mark.asyncio
    async def test_download_rejects_unconfigured_nondefault_port_before_network(self, monkeypatch):
        import httpx

        def must_not_create_client(**kwargs):
            raise AssertionError("network client must not be created")

        monkeypatch.setattr(httpx, "AsyncClient", must_not_create_client)
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://test.relay:8443/media/file.bin",
            "sha256": "a" * 64,
            "size": 12,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is None

    @pytest.mark.asyncio
    async def test_download_allows_explicitly_configured_nondefault_port(self, monkeypatch):
        import httpx

        payload = b"x"
        real_async_client = httpx.AsyncClient

        def handler(request):
            assert request.url.port == 8443
            return httpx.Response(200, content=payload, headers={"content-length": "1"})

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            ),
        )
        adapter = _make_adapter({"attachment_hosts": ["test.relay:8443"]})

        cached = await adapter._download_attachment({
            "url": "https://test.relay:8443/media/file.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": 1,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is not None

    @pytest.mark.asyncio
    async def test_unaddressed_channel_attachment_is_not_downloaded(self):
        adapter = _make_adapter()
        authorization_check = MagicMock(return_value=True)
        adapter.set_authorization_check(authorization_check)
        adapter._cache_inbound_attachments = AsyncMock()
        adapter._download_attachment = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("unaddressed-attachment", content="shared file")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 12",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        authorization_check.assert_not_called()
        adapter._cache_inbound_attachments.assert_not_awaited()
        adapter._download_attachment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_attachment_download_is_visible_to_agent(self):
        adapter = _make_adapter()
        adapter._download_attachment = AsyncMock(return_value=None)
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        dispatched = []

        async def capture(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture
        event = _event("failed-attachment", content="@Chip inspect this")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 12",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        assert "could not be downloaded" in dispatched[-1]["text"]
        assert dispatched[-1]["media_urls"] == []

    @pytest.mark.asyncio
    async def test_dispatch_builds_document_message_event_with_cached_path(self):
        adapter = _make_adapter()
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._dispatch_message(
            text="inspect",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="Other",
            message_id="document-event",
            created_at=1000,
            media_urls=["/agent/cache/doc_report.pdf"],
            media_types=["application/pdf"],
            message_type=MessageType.DOCUMENT,
            raw_message={"id": "document-event"},
        )

        call = adapter.handle_message.await_args
        assert call is not None
        dispatched_event = call.args[0]
        assert dispatched_event.message_type is MessageType.DOCUMENT
        assert dispatched_event.media_urls == ["/agent/cache/doc_report.pdf"]
        assert dispatched_event.media_types == ["application/pdf"]
        assert dispatched_event.media_text_inlined == [False]
        assert dispatched_event.raw_message == {"id": "document-event"}

    def test_imeta_sanitizes_filename_and_rejects_incomplete_metadata(self):
        event = _event("metadata", content="@Chip files")
        event["tags"].extend([
            [
                "imeta",
                "url https://test.relay/media/valid.bin",
                "m text/plain",
                "x " + "a" * 64,
                "size 12",
                "filename ../../private/report.txt",
            ],
            [
                "imeta",
                "url http://test.relay/media/insecure.bin",
                "x " + "b" * 64,
                "size 12",
                "filename insecure.bin",
            ],
            [
                "imeta",
                "url https://test.relay/media/no-hash.bin",
                "size 12",
                "filename no-hash.bin",
            ],
        ])

        attachments = BuzzAdapter._imeta_attachments(event)

        assert len(attachments) == 1
        assert attachments[0]["filename"] == "report.txt"

    @pytest.mark.asyncio
    async def test_attachment_only_dm_is_downloaded_and_dispatched(self):
        adapter = _make_adapter()
        attachment = CachedMedia(
            path="/agent/cache/doc_attachment.bin",
            media_type="application/octet-stream",
            kind="document",
            display_name="attachment.bin",
        )
        adapter._download_attachment = AsyncMock(return_value=attachment)
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[DM_CHANNEL] = {
            "chat_type": "dm",
            "last_ts": 0,
            "seen": {},
        }
        dispatched = []

        async def capture(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture
        event = _event("attachment-only", content="")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 12",
            "filename attachment.bin",
        ])

        await adapter._handle_event(
            DM_CHANNEL,
            adapter._channel_state[DM_CHANNEL],
            event,
        )

        adapter._download_attachment.assert_awaited_once()
        assert dispatched[-1]["media_urls"] == [attachment.path]
        assert dispatched[-1]["message_type"] is MessageType.DOCUMENT

    @pytest.mark.asyncio
    async def test_malformed_attachment_only_dm_dispatches_once_with_bounded_note(self):
        adapter = _make_adapter()
        adapter._download_attachment = AsyncMock()
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[DM_CHANNEL] = {
            "chat_type": "dm",
            "last_ts": 0,
            "seen": {},
        }
        adapter._dispatch_message = AsyncMock()
        event = _event("malformed-attachment-only", content="")
        event["tags"].append(["imeta", "url definitely-not-a-url"])

        await adapter._handle_event(DM_CHANNEL, adapter._channel_state[DM_CHANNEL], event)
        await adapter._handle_event(DM_CHANNEL, adapter._channel_state[DM_CHANNEL], event)

        adapter._download_attachment.assert_not_awaited()
        adapter._dispatch_message.assert_awaited_once()
        call = adapter._dispatch_message.await_args
        assert call is not None
        dispatched = call.kwargs
        assert "1 Buzz attachment(s) rejected" in dispatched["text"]
        assert len(dispatched["text"]) <= 80
        assert dispatched["media_urls"] == []

    @pytest.mark.asyncio
    async def test_excess_imeta_is_reported_while_accepted_files_remain_attached(self):
        adapter = _make_adapter()
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        adapter._download_attachment = AsyncMock(
            side_effect=lambda metadata: CachedMedia(
                f"/cache/{metadata['filename']}",
                metadata["mime_type"],
                "document",
                metadata["filename"],
            )
        )
        adapter._dispatch_message = AsyncMock()
        event = _event("excess-imeta", content="@Chip inspect")
        for index in range(5):
            event["tags"].append([
                "imeta",
                f"url https://test.relay/media/{index}.bin",
                "m application/octet-stream",
                "x " + format(index + 1, "064x"),
                "size 1",
                f"filename {index}.bin",
            ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        assert adapter._download_attachment.await_count == 4
        call = adapter._dispatch_message.await_args
        assert call is not None
        dispatched = call.kwargs
        assert "1 Buzz attachment(s) rejected" in dispatched["text"]
        assert len(dispatched["media_urls"]) == 4

    def test_imeta_bounds_filename_to_filesystem_safe_utf8_length(self):
        event = _event("long-name", content="@Chip file")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename " + ("é" * 180) + ".pdf",
        ])

        filename = BuzzAdapter._imeta_attachments(event)[0]["filename"]

        assert len(filename.encode("utf-8")) <= 120
        assert filename.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_cache_write_failure_is_treated_as_failed_attachment(self, monkeypatch):
        import httpx

        payload = b"x"
        digest = hashlib.sha256(payload).hexdigest()
        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        content=payload,
                        headers={"content-length": "1"},
                    )
                ),
                **kwargs,
            ),
        )
        monkeypatch.setattr(
            _buzz_mod,
            "cache_media_bytes",
            MagicMock(side_effect=OSError(36, "File name too long")),
        )
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://test.relay/media/file.bin",
            "sha256": digest,
            "size": 1,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is None

    @pytest.mark.asyncio
    async def test_download_has_total_deadline(self, monkeypatch):
        import httpx

        payload = b"x"
        real_async_client = httpx.AsyncClient

        async def slow_handler(_request):
            await asyncio.sleep(0.05)
            return httpx.Response(200, content=payload, headers={"content-length": "1"})

        monkeypatch.setattr(_buzz_mod, "_ATTACHMENT_DOWNLOAD_TIMEOUT", 0.01)
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(slow_handler),
                **kwargs,
            ),
        )
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://test.relay/media/file.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": 1,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is None

    @pytest.mark.asyncio
    async def test_multiple_mixed_attachments_use_document_semantics(self):
        adapter = _make_adapter()
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        cached = [
            CachedMedia("/cache/image.png", "image/png", "image", "image.png"),
            CachedMedia("/cache/audio.mp3", "audio/mpeg", "audio", "audio.mp3"),
        ]
        adapter._cache_inbound_attachments = AsyncMock(return_value=cached)
        dispatched = []

        async def capture(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture
        event = _event("mixed", content="@Chip inspect")
        for index, mime_type in enumerate(("image/png", "audio/mpeg")):
            event["tags"].append([
                "imeta",
                f"url https://test.relay/media/{index}.bin",
                f"m {mime_type}",
                "x " + format(index + 1, "064x"),
                "size 1",
                f"filename {index}.bin",
            ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        assert dispatched[-1]["message_type"] is MessageType.DOCUMENT
        assert dispatched[-1]["media_types"] == ["image/png", "audio/mpeg"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "payload", "headers", "declared_size", "expected_digest"),
        [
            (302, b"", {}, 1, "a" * 64),
            (200, b"x", {"content-length": "invalid"}, 1, hashlib.sha256(b"x").hexdigest()),
            (200, b"x", {}, 2, hashlib.sha256(b"x").hexdigest()),
            (200, b"xx", {}, 1, hashlib.sha256(b"xx").hexdigest()),
            (200, b"x", {}, 1, "a" * 64),
        ],
    )
    async def test_download_rejects_invalid_response_or_integrity(
        self,
        monkeypatch,
        status,
        payload,
        headers,
        declared_size,
        expected_digest,
    ):
        import httpx

        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(status, content=payload, headers=headers)
                ),
                **kwargs,
            ),
        )
        adapter = _make_adapter()

        cached = await adapter._download_attachment({
            "url": "https://test.relay/media/file.bin",
            "sha256": expected_digest,
            "size": declared_size,
            "filename": "file.bin",
            "mime_type": "application/octet-stream",
        })

        assert cached is None

    def test_imeta_rejects_url_credentials_and_fragments_and_caps_items(self):
        event = _event("url-safety", content="@Chip files")
        event["tags"].extend([
            [
                "imeta",
                "url https://user:password@test.relay/media/private.bin",
                "x " + "a" * 64,
                "size 1",
                "filename private.bin",
            ],
            [
                "imeta",
                "url https://test.relay/media/fragment.bin#hidden",
                "x " + "b" * 64,
                "size 1",
                "filename fragment.bin",
            ],
        ])
        for index in range(6):
            event["tags"].append([
                "imeta",
                f"url https://test.relay/media/{index}.bin",
                "x " + format(index + 1, "064x"),
                "size 1",
                f"filename {index}.bin",
            ])

        attachments = BuzzAdapter._imeta_attachments(event)

        assert len(attachments) == 4
        assert all("@" not in item["url"] and "#" not in item["url"] for item in attachments)

    @pytest.mark.asyncio
    async def test_self_attachment_stops_before_authorization_or_cache(self):
        adapter = _make_adapter()
        authorization_check = MagicMock(return_value=True)
        adapter.set_authorization_check(authorization_check)
        adapter._cache_inbound_attachments = AsyncMock()
        adapter._dispatch_message = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("self-attachment", pubkey=SELF_PUBKEY, content="@Chip inspect")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        authorization_check.assert_not_called()
        adapter._cache_inbound_attachments.assert_not_awaited()
        adapter._dispatch_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_sender_attachment_is_not_downloaded(self):
        adapter = _make_adapter()
        adapter._allowed_pubkeys = {"f" * 64}
        authorization_check = MagicMock(return_value=True)
        adapter.set_authorization_check(authorization_check)
        adapter._cache_inbound_attachments = AsyncMock()
        adapter._download_attachment = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("unauthorized-attachment", content="@Chip inspect")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        authorization_check.assert_not_called()
        adapter._cache_inbound_attachments.assert_not_awaited()
        adapter._download_attachment.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("authorization", [False, None, "raise", "truthy"])
    async def test_non_true_gateway_authority_never_caches_even_for_locally_allowed_sender(
        self,
        authorization,
    ):
        adapter = _make_adapter()
        adapter._run_cli = AsyncMock(side_effect=AssertionError("authorization test invoked Buzz CLI"))
        adapter._resolve_user_name = AsyncMock(return_value="Other")
        adapter._allowed_pubkeys = {OTHER_PUBKEY}
        if authorization is None:
            adapter.set_authorization_check(None)
        elif authorization == "raise":
            def raise_unknown(*_args):
                raise RuntimeError("authorization backend unavailable")

            adapter.set_authorization_check(raise_unknown)
        elif authorization == "truthy":
            adapter.set_authorization_check(lambda *_args: "AUTHORIZED")
        else:
            adapter.set_authorization_check(lambda *_args: False)
        adapter._cache_inbound_attachments = AsyncMock()
        adapter._dispatch_message = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event(f"gateway-{authorization}-attachment", content="@Chip inspect")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        adapter._cache_inbound_attachments.assert_not_awaited()
        adapter._dispatch_message.assert_awaited_once()
        call = adapter._dispatch_message.await_args
        assert call is not None
        assert call.kwargs["media_urls"] == []

    @pytest.mark.asyncio
    async def test_explicit_true_gateway_authority_caches_attachment(self):
        adapter = _make_adapter()
        adapter._run_cli = AsyncMock(side_effect=AssertionError("authorization test invoked Buzz CLI"))
        adapter._resolve_user_name = AsyncMock(return_value="Other")
        authorization_check = MagicMock(return_value=True)
        adapter.set_authorization_check(authorization_check)
        cached = CachedMedia(
            "/cache/authorized.bin",
            "application/octet-stream",
            "document",
            "authorized.bin",
        )
        adapter._cache_inbound_attachments = AsyncMock(return_value=[cached])
        adapter._dispatch_message = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("gateway-authorized-attachment", content="@Chip inspect")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        authorization_check.assert_called_once_with(OTHER_PUBKEY, "group", CHANNEL)
        adapter._cache_inbound_attachments.assert_awaited_once()
        call = adapter._dispatch_message.await_args
        assert call is not None
        assert call.kwargs["media_urls"] == [cached.path]

    @pytest.mark.asyncio
    async def test_real_gateway_auth_callback_defaults_to_no_attachment_side_effects(
        self,
        monkeypatch,
    ):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.adapters = {}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False

        adapter = _make_adapter()
        adapter._run_cli = AsyncMock(side_effect=AssertionError("authorization test invoked Buzz CLI"))
        adapter._resolve_user_name = AsyncMock(return_value="Other")
        adapter._allowed_pubkeys = {OTHER_PUBKEY}
        adapter.set_authorization_check(runner._make_adapter_auth_check(adapter.platform))
        adapter._cache_inbound_attachments = AsyncMock()
        adapter._dispatch_message = AsyncMock()
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        event = _event("gateway-default-denied-attachment", content="@Chip inspect")
        event["tags"].append([
            "imeta",
            "url https://test.relay/media/file.bin",
            "m application/octet-stream",
            "x " + "a" * 64,
            "size 1",
            "filename file.bin",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        runner.pairing_store.is_approved.assert_called_once_with("buzz", OTHER_PUBKEY)
        adapter._cache_inbound_attachments.assert_not_awaited()
        adapter._dispatch_message.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kind", "mime_type", "expected_type"),
        [
            ("image", "image/png", MessageType.PHOTO),
            ("video", "video/mp4", MessageType.VIDEO),
            ("audio", "audio/mpeg", MessageType.AUDIO),
            ("document", "application/pdf", MessageType.DOCUMENT),
        ],
    )
    async def test_homogeneous_attachment_kind_sets_message_type(
        self,
        kind,
        mime_type,
        expected_type,
    ):
        adapter = _make_adapter()
        adapter._user_names[OTHER_PUBKEY] = "Other"
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        cached = CachedMedia(
            f"/cache/file-{kind}",
            mime_type,
            kind,
            f"file-{kind}",
        )
        adapter._cache_inbound_attachments = AsyncMock(return_value=[cached])
        dispatched = []

        async def capture(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture
        event = _event(f"homogeneous-{kind}", content="@Chip inspect")
        event["tags"].append([
            "imeta",
            f"url https://test.relay/media/{kind}",
            f"m {mime_type}",
            "x " + "a" * 64,
            "size 1",
            f"filename file-{kind}",
        ])

        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        assert dispatched[-1]["message_type"] is expected_type


# ── Mention gating / DMs / authorization ──────────────────────────────────


class TestMentionGating:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)

    @pytest.mark.asyncio
    async def test_unaddressed_channel_message_ignored(self, adapter):
        await self._poll_with(adapter, _event("e1", content="just chatting", created_at=10))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_name_mention_dispatched(self, adapter):
        await self._poll_with(adapter, _event("e1", content="hey @Chip can you help?", created_at=10))
        assert len(adapter._dispatched) == 1


    @pytest.mark.asyncio
    async def test_allowlist_blocks_unauthorized(self, adapter):
        adapter._allowed_pubkeys = {"b" * 64}
        await self._poll_with(adapter, _event("e1", content="@Chip hello", created_at=10))
        assert adapter._dispatched == []


# ── DM classification via p-tags (issue #68871) ──────────────────────────
#
# `buzz dms list` returns [] on some hosted relays, so DM conversations leak
# in via `channels list` and get seeded chat_type="group".  The adapter must
# reclassify them from the Nostr tags of real traffic: DM messages are
# p-tagged to our own pubkey WITHOUT the text mentioning us, while channel
# messages only ever p-tag us when the text visibly @mentions us.


def _tagged_event(event_id, channel, *, content, pubkey=OTHER_PUBKEY,
                  created_at=1000, kind=9, p=None, reply_to=None):
    """Event with the tag shapes observed on a live relay (h/p/e tags)."""
    tags = [["h", channel]]
    if reply_to:
        tags.append(["e", reply_to, "", "reply"])
    if p:
        tags.append(["p", p])
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
    }


class TestDmClassification:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        # Metadata exactly as `channels list` returns it on the hosted relay.
        a._channel_meta = {
            DM_CHANNEL: {"channel_id": DM_CHANNEL, "name": "DM", "description": ""},
            CHANNEL: {
                "channel_id": CHANNEL,
                "name": "general",
                "description": "General conversation and community updates.",
            },
        }
        a._channel_names = {DM_CHANNEL: "DM", CHANNEL: "general"}
        # Both leaked in as group — the bug under test.
        a._channel_state[DM_CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, channel, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(channel)

    @pytest.mark.asyncio
    async def test_unmentioned_ptagged_dm_latches_and_dispatches(self, adapter):
        """The reported bug: a DM without an @mention must dispatch."""
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e1", DM_CHANNEL, content="here's a test message", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[DM_CHANNEL]["chat_type"] == "dm"
        assert [d["message_id"] for d in adapter._dispatched] == ["e1"]
        assert adapter._dispatched[0]["chat_type"] == "dm"


    @pytest.mark.asyncio
    async def test_general_reply_ptagging_self_stays_channel(self, adapter):
        """A #general reply to us p-tags our pubkey (observed live) — that
        must NOT reclassify the channel; mention gating still applies."""
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="@chip what's up?",
                          p=SELF_PUBKEY, reply_to="root-event"),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        # It carried a mention, so it dispatches — but as a group message.
        assert [d["chat_type"] for d in adapter._dispatched] == ["group"]

        # And once the mention is absent, the channel gate drops the message
        # even though the earlier reply p-tagged us.
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e2", CHANNEL, content="thanks everyone", created_at=1001),
        )
        assert len(adapter._dispatched) == 1


    @pytest.mark.asyncio
    async def test_channel_like_metadata_blocks_latch_even_without_mention(self, adapter):
        """Second guard on its own: even a p-tagged, un-mentioned message
        cannot reclassify a conversation whose metadata says real channel."""
        adapter._channel_meta[CHANNEL]["description"] = ""
        adapter._channel_meta[CHANNEL]["name"] = "announcements"
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="fyi everyone", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        assert adapter._dispatched == []


    @pytest.mark.asyncio
    async def test_dm_shaped_channel_discovered_when_dms_list_empty(self):
        """Fallback discovery: with `dms list` broken (returns []), a
        DM-shaped `channels list` entry gets watched; real channels not
        already watched are left alone."""
        a = _make_adapter()
        cli = _ScriptedCli()
        cli.script("dms", "list", [])
        cli.script("channels", "list", [
            {"channel_id": DM_CHANNEL, "name": "DM", "description": "", "created_at": 1},
            {"channel_id": CHANNEL, "name": "general",
             "description": "General conversation and community updates.", "created_at": 2},
        ])
        a._run_cli = cli
        await a._discover_dms(seed=False)
        # Watched as group; the p-tag latch flips it on the first real DM.
        assert a._channel_state[DM_CHANNEL]["chat_type"] == "group"
        assert a._may_reclassify_as_dm(DM_CHANNEL) is True
        assert CHANNEL not in a._channel_state
        assert a._may_reclassify_as_dm(CHANNEL) is False


# ── Sending ───────────────────────────────────────────────────────────────


class TestBuzzAdapterSend:

    @pytest.mark.asyncio
    async def test_send_success_via_stdin(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt123", "message": ""})
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "hello **markdown**")
        assert result.success is True
        assert result.message_id == "evt123"

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "send"]
        assert args[args.index("--channel") + 1] == CHANNEL
        # Content travels via stdin (--content -), never argv
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "hello **markdown**"
        # Our own event id is marked seen for echo suppression
        assert "evt123" in adapter._channel_state[CHANNEL]["seen"]


    @pytest.mark.asyncio
    async def test_send_image_local_file_uses_file_flag(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt126", "message": ""})
        adapter._run_cli = cli
        result = await adapter.send_image(CHANNEL, str(img), caption="screenshot")
        assert result.success is True
        args, _stdin = cli.calls[0]
        assert args[args.index("--file") + 1] == str(img)


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestBuzzAdapterLifecycle:


    @pytest.mark.asyncio
    async def test_disconnect_releases_scoped_lock(self, monkeypatch):
        """The identity lock taken in connect() must be released on disconnect."""
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        adapter = _make_adapter()
        adapter._lock_key = "wss://relay.example:" + SELF_PUBKEY
        await adapter.disconnect()
        assert released == [("buzz", "wss://relay.example:" + SELF_PUBKEY)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """A second profile using the same relay+pubkey must fail fast."""
        import gateway.status as gateway_status

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: False
        )
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        cli = _ScriptedCli()
        cli.script(
            "users", "get",
            [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}],
        )
        adapter._run_cli = cli
        assert await adapter.connect() is False
        assert adapter._lock_key is None


# ── Credentials / requirements ────────────────────────────────────────────


class TestCredentialResolution:

    def test_env_key_wins(self, monkeypatch):
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1fromenv")
        assert _resolve_private_key() == "nsec1fromenv"

    def test_credentials_file_fallback(self, monkeypatch, tmp_path):
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "npub": "npub1x"}), encoding="utf-8")
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        assert _resolve_private_key() == "nsec1fromfile"


# ── Env enablement / registration / standalone send ──────────────────────


class TestEnvEnablement:

    def test_returns_none_when_unconfigured(self):
        assert _env_enablement() is None


class TestBuzzPluginRegistration:

    def test_register_platform_contract(self):
        from gateway.platform_registry import platform_registry

        platform_registry.unregister("buzz")
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "buzz"
        assert kwargs["cron_deliver_env_var"] == "BUZZ_HOME_CHANNEL"
        assert kwargs["allowed_users_env"] == "BUZZ_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "BUZZ_ALLOW_ALL_USERS"
        assert callable(kwargs["standalone_sender_fn"])
        assert callable(kwargs["env_enablement_fn"])
        assert set(kwargs["required_env"]) == {"BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"}


class TestStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_success(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))

        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0):
            captured.update(cli_path=cli_path, args=args, relay_url=relay_url, input_text=input_text)
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), CHANNEL, "cron says hi")
        assert result == {"success": True, "message_id": "evt-cron"}
        assert captured["args"][:2] == ["messages", "send"]
        assert captured["input_text"] == "cron says hi"
        # The private key must never be part of argv
        assert all("nsec1x" not in str(a) for a in captured["args"])


