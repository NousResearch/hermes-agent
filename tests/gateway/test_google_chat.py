"""
Tests for Google Chat platform adapter.

Covers: platform registration, env config loading, adapter init, connect
validation, Pub/Sub callback routing (message / membership / card / error),
outbound send with typing patch-in-place and chunking, attachment send paths,
SSRF guard on attachment download, supervisor reconnect, and authorization
(including the user_id_alt email match for GOOGLE_CHAT_ALLOWED_USERS).

Note: the Google libraries may not be installed in the test environment.
We shim the imports at module load so collection doesn't fail.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config


# ---------------------------------------------------------------------------
# Mock the google-* packages if they are not installed
# ---------------------------------------------------------------------------

class _FakeHttpError(Exception):
    """Stand-in for googleapiclient.errors.HttpError with .resp.status."""

    def __init__(self, status=500, content=b"", reason=""):
        self.resp = MagicMock()
        self.resp.status = status
        self.content = content
        self.reason = reason
        super().__init__(f"HTTP {status}: {reason or 'error'}")


def _ensure_google_mocks():
    """Install mock google-* modules so GoogleChatAdapter can be imported."""
    if "google.cloud.pubsub_v1" in sys.modules and hasattr(
        sys.modules["google.cloud.pubsub_v1"], "__file__"
    ):
        return  # Real libraries installed, use them.

    # --- google.cloud.pubsub_v1 ---
    google = MagicMock()
    google_cloud = MagicMock()
    pubsub_v1 = MagicMock()
    pubsub_v1.SubscriberClient = MagicMock
    pubsub_v1.types.FlowControl = MagicMock

    # --- google.api_core.exceptions ---
    gax = MagicMock()
    gax.NotFound = type("NotFound", (Exception,), {})
    gax.PermissionDenied = type("PermissionDenied", (Exception,), {})
    gax.Unauthenticated = type("Unauthenticated", (Exception,), {})

    # --- google.oauth2.service_account ---
    oauth2 = MagicMock()
    oauth2.Credentials.from_service_account_info = MagicMock(return_value=MagicMock())
    oauth2.Credentials.from_service_account_file = MagicMock(return_value=MagicMock())

    # --- google_auth_httplib2 + httplib2 ---
    httplib2 = MagicMock()
    httplib2.Http = MagicMock()
    google_auth_httplib2 = MagicMock()
    google_auth_httplib2.AuthorizedHttp = MagicMock()

    # --- googleapiclient ---
    gapi = MagicMock()
    gapi_discovery = MagicMock()
    gapi_discovery.build = MagicMock()
    gapi_errors = MagicMock()
    gapi_errors.HttpError = _FakeHttpError
    gapi_http = MagicMock()
    gapi_http.MediaFileUpload = MagicMock

    modules = {
        "google": google,
        "google.cloud": google_cloud,
        "google.cloud.pubsub_v1": pubsub_v1,
        "google.api_core": MagicMock(exceptions=gax),
        "google.api_core.exceptions": gax,
        "google.oauth2": MagicMock(service_account=oauth2),
        "google.oauth2.service_account": oauth2,
        "google_auth_httplib2": google_auth_httplib2,
        "httplib2": httplib2,
        "googleapiclient": gapi,
        "googleapiclient.discovery": gapi_discovery,
        "googleapiclient.errors": gapi_errors,
        "googleapiclient.http": gapi_http,
    }
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)


_ensure_google_mocks()


# Patch the availability flag before importing, so the adapter doesn't bail
# out at the "missing deps" gate during construction.
import gateway.platforms.google_chat as _gc_mod  # noqa: E402

_gc_mod.GOOGLE_CHAT_AVAILABLE = True

from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome  # noqa: E402
from gateway.platforms.google_chat import (  # noqa: E402
    GoogleChatAdapter,
    _is_google_owned_host,
    _mime_for_message_type,
    _redact_sensitive,
    check_google_chat_requirements,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _base_config(**extra):
    cfg = PlatformConfig(enabled=True)
    cfg.extra.update({
        "project_id": "test-project",
        "subscription_name": "projects/test-project/subscriptions/test-sub",
        "service_account_json": "/tmp/fake-sa.json",
    })
    cfg.extra.update(extra)
    return cfg


@pytest.fixture()
def adapter():
    """Build an adapter with its loop captured and Chat client mocked."""
    a = GoogleChatAdapter(_base_config())
    a._loop = asyncio.get_event_loop_policy().new_event_loop()
    a._chat_api = MagicMock()
    a._subscriber = MagicMock()
    a._credentials = MagicMock()
    a._project_id = "test-project"
    a._subscription_path = "projects/test-project/subscriptions/test-sub"
    a._new_authed_http = MagicMock(return_value=MagicMock())
    a.handle_message = AsyncMock()
    yield a
    try:
        a._loop.close()
    except Exception:
        pass


def _make_pubsub_message(data: dict, *, attributes=None):
    """Build a Mock Pub/Sub Message with ack/nack trackers."""
    msg = MagicMock()
    msg.data = json.dumps(data).encode("utf-8")
    msg.attributes = attributes or {}
    msg.ack = MagicMock()
    msg.nack = MagicMock()
    return msg


def _make_chat_envelope(text="hello", sender_email="u@example.com", sender_type="HUMAN",
                       msg_name=None, thread_name=None, attachments=None,
                       slash_command=None):
    """Build a realistic Google Chat CloudEvents-style envelope body."""
    msg = {
        "name": msg_name or "spaces/S/messages/M.M",
        "sender": {
            "name": "users/12345",
            "email": sender_email,
            "displayName": "User Name",
            "type": sender_type,
        },
        "text": text,
        "argumentText": text,
        "thread": {"name": thread_name or "spaces/S/threads/T"},
        "space": {"name": "spaces/S", "spaceType": "DIRECT_MESSAGE"},
    }
    if attachments is not None:
        msg["attachment"] = attachments
    if slash_command is not None:
        msg["slashCommand"] = slash_command

    return {
        "chat": {
            "messagePayload": {
                "space": msg["space"],
                "message": msg,
            }
        }
    }


# ===========================================================================
# Platform registration + requirements
# ===========================================================================


class TestPlatformRegistration:
    def test_enum_value(self):
        assert Platform.GOOGLE_CHAT.value == "google_chat"

    def test_requirements_check_returns_true_when_available(self):
        # The shim flag is True in this test module.
        assert check_google_chat_requirements() is True


# ===========================================================================
# Env-var config loading
# ===========================================================================


class TestEnvConfigLoading:
    _ENV_VARS = (
        "GOOGLE_CHAT_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CHAT_SUBSCRIPTION_NAME",
        "GOOGLE_CHAT_SUBSCRIPTION",
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CHAT_HOME_CHANNEL",
        "GOOGLE_CHAT_HOME_CHANNEL_NAME",
    )

    def _clean_env(self, monkeypatch):
        for v in self._ENV_VARS:
            monkeypatch.delenv(v, raising=False)

    def test_project_id_primary(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "my-proj")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME",
                           "projects/my-proj/subscriptions/my-sub")
        cfg = load_gateway_config()
        gc = cfg.platforms[Platform.GOOGLE_CHAT]
        assert gc.enabled is True
        assert gc.extra["project_id"] == "my-proj"

    def test_project_id_falls_back_to_google_cloud_project(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fallback-proj")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION",
                           "projects/fallback-proj/subscriptions/s")
        cfg = load_gateway_config()
        gc = cfg.platforms[Platform.GOOGLE_CHAT]
        assert gc.extra["project_id"] == "fallback-proj"

    def test_subscription_accepts_legacy_alias(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION", "projects/p/subscriptions/s")
        cfg = load_gateway_config()
        gc = cfg.platforms[Platform.GOOGLE_CHAT]
        assert gc.extra["subscription_name"] == "projects/p/subscriptions/s"

    def test_sa_path_falls_back_to_google_application_credentials(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME",
                           "projects/p/subscriptions/s")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/opt/sa.json")
        cfg = load_gateway_config()
        gc = cfg.platforms[Platform.GOOGLE_CHAT]
        assert gc.extra["service_account_json"] == "/opt/sa.json"

    def test_missing_subscription_does_not_enable(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        # No subscription.
        cfg = load_gateway_config()
        assert Platform.GOOGLE_CHAT not in cfg.platforms

    def test_missing_project_does_not_enable(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME",
                           "projects/p/subscriptions/s")
        cfg = load_gateway_config()
        assert Platform.GOOGLE_CHAT not in cfg.platforms

    def test_home_channel_populated(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME",
                           "projects/p/subscriptions/s")
        monkeypatch.setenv("GOOGLE_CHAT_HOME_CHANNEL", "spaces/HOME")
        cfg = load_gateway_config()
        gc = cfg.platforms[Platform.GOOGLE_CHAT]
        assert gc.home_channel is not None
        assert gc.home_channel.chat_id == "spaces/HOME"

    def test_connected_platforms_recognises_via_extras(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME",
                           "projects/p/subscriptions/s")
        cfg = load_gateway_config()
        assert Platform.GOOGLE_CHAT in cfg.get_connected_platforms()


# ===========================================================================
# Pure helpers
# ===========================================================================


class TestHelpers:
    def test_mime_image_maps_to_photo(self):
        assert _mime_for_message_type("image/png") == MessageType.PHOTO

    def test_mime_audio_maps_to_audio(self):
        assert _mime_for_message_type("audio/ogg") == MessageType.AUDIO

    def test_mime_video_maps_to_video(self):
        assert _mime_for_message_type("video/mp4") == MessageType.VIDEO

    def test_mime_other_maps_to_document(self):
        assert _mime_for_message_type("application/pdf") == MessageType.DOCUMENT

    def test_mime_empty_maps_to_document(self):
        assert _mime_for_message_type("") == MessageType.DOCUMENT


class TestRedactSensitive:
    def test_redacts_subscription_path(self):
        out = _redact_sensitive("error on projects/proj-a/subscriptions/sub-b please")
        assert "proj-a" not in out
        assert "sub-b" not in out
        assert "please" in out  # surrounding text preserved

    def test_redacts_topic_path(self):
        out = _redact_sensitive("publisher on projects/p/topics/t")
        assert "projects/p/topics/t" not in out
        assert "<redacted>" in out

    def test_redacts_service_account_email(self):
        out = _redact_sensitive("bot@my-project-123.iam.gserviceaccount.com is the principal")
        assert "bot" not in out
        assert "my-project-123" not in out
        assert "principal" in out

    def test_empty_text_passes_through(self):
        assert _redact_sensitive("") == ""
        assert _redact_sensitive(None) is None


class TestGoogleOwnedHost:
    @pytest.mark.parametrize("url", [
        "https://chat.googleapis.com/v1/x",
        "https://www.googleapis.com/upload/chat/v1/x",
        "https://drive.google.com/file/d/abc",
        "https://lh3.googleusercontent.com/photo.jpg",
    ])
    def test_accepts_google_hosts(self, url):
        assert _is_google_owned_host(url) is True

    @pytest.mark.parametrize("url", [
        "https://evil.com/foo",
        "https://169.254.169.254/latest/meta-data/",
        "https://metadata.internal/computeMetadata/v1/",
        "https://chat.google.com.attacker.example/",  # subdomain hijack
        "http://chat.googleapis.com/",  # http is rejected
        "ftp://drive.google.com/x",  # non-https rejected
        "not a url",
    ])
    def test_rejects_non_google_or_insecure(self, url):
        assert _is_google_owned_host(url) is False


# ===========================================================================
# Config validation (inside connect())
# ===========================================================================


class TestValidateConfig:
    def test_missing_project_raises(self):
        a = GoogleChatAdapter(PlatformConfig(enabled=True))
        with pytest.raises(ValueError, match="PROJECT"):
            a._validate_config()

    def test_missing_subscription_raises(self):
        cfg = PlatformConfig(enabled=True)
        cfg.extra["project_id"] = "p"
        a = GoogleChatAdapter(cfg)
        with pytest.raises(ValueError, match="SUBSCRIPTION"):
            a._validate_config()

    def test_subscription_format_rejected(self):
        cfg = _base_config(subscription_name="not-a-valid-path")
        a = GoogleChatAdapter(cfg)
        with pytest.raises(ValueError, match="projects/"):
            a._validate_config()

    def test_subscription_project_mismatch_rejected(self):
        cfg = _base_config(
            subscription_name="projects/other-proj/subscriptions/s",
            project_id="my-proj",
        )
        a = GoogleChatAdapter(cfg)
        with pytest.raises(ValueError, match="does not match"):
            a._validate_config()

    def test_validate_config_happy(self):
        a = GoogleChatAdapter(_base_config())
        project, sub = a._validate_config()
        assert project == "test-project"
        assert sub == "projects/test-project/subscriptions/test-sub"


# ===========================================================================
# _chunk_text
# ===========================================================================


class TestChunkText:
    def test_empty_returns_empty_list(self, adapter):
        assert adapter._chunk_text("") == []

    def test_short_returns_single_chunk(self, adapter):
        assert adapter._chunk_text("hola") == ["hola"]

    def test_long_splits_into_multiple(self, adapter):
        text = "a" * 10000
        chunks = adapter._chunk_text(text)
        assert len(chunks) >= 2
        assert all(len(c) <= 4000 for c in chunks)
        assert "".join(chunks) == text

    def test_splits_on_newline_near_boundary(self, adapter):
        # Build a ~5000-char string with a newline near the 4000 cut.
        text = "a" * 3800 + "\n" + "b" * 1500
        chunks = adapter._chunk_text(text)
        assert len(chunks) == 2
        # First chunk ends at the newline (3800 a's, no trailing b's)
        assert chunks[0].endswith("a")
        assert "\n" not in chunks[0][-5:]  # the split already ate the newline


# ===========================================================================
# _on_pubsub_message — event routing
# ===========================================================================


class TestOnPubsubMessage:
    """Pub/Sub callback routing. The callback runs in a thread and dispatches
    to the asyncio loop; here we assert ack/nack behaviour and that
    handle_message is scheduled only for MESSAGE events."""

    def test_shutting_down_nacks(self, adapter):
        adapter._shutting_down = True
        msg = _make_pubsub_message({"whatever": 1})
        adapter._on_pubsub_message(msg)
        msg.nack.assert_called_once()
        msg.ack.assert_not_called()

    def test_malformed_json_acks_without_dispatch(self, adapter):
        msg = MagicMock()
        msg.data = b"not valid json {"
        msg.attributes = {}
        msg.ack = MagicMock()
        msg.nack = MagicMock()
        adapter._on_pubsub_message(msg)
        msg.ack.assert_called_once()
        msg.nack.assert_not_called()

    def test_membership_created_caches_bot_user_id(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._bot_user_id = None
        envelope = {
            "chat": {
                "membershipPayload": {
                    "space": {"name": "spaces/S"},
                    "membership": {"member": {"name": "users/BOT_ID", "type": "BOT"}},
                }
            }
        }
        msg = _make_pubsub_message(
            envelope,
            attributes={"ce-type": "google.workspace.chat.membership.v1.created"},
        )
        adapter._on_pubsub_message(msg)
        assert adapter._bot_user_id == "users/BOT_ID"
        msg.ack.assert_called_once()

    def test_membership_deleted_acks_no_dispatch(self, adapter):
        envelope = {
            "chat": {
                "membershipPayload": {
                    "space": {"name": "spaces/S"},
                    "membership": {"member": {"name": "users/BOT_ID", "type": "BOT"}},
                }
            }
        }
        msg = _make_pubsub_message(
            envelope,
            attributes={"ce-type": "google.workspace.chat.membership.v1.deleted"},
        )
        adapter._on_pubsub_message(msg)
        msg.ack.assert_called_once()

    def test_bot_sender_is_filtered(self, adapter):
        env = _make_chat_envelope(sender_type="BOT")
        msg = _make_pubsub_message(env)
        with patch.object(adapter, "_submit_on_loop") as submit:
            adapter._on_pubsub_message(msg)
            submit.assert_not_called()
        msg.ack.assert_called_once()

    def test_duplicate_message_dropped(self, adapter):
        env = _make_chat_envelope(msg_name="spaces/S/messages/DUP.DUP")
        # Prime dedup
        adapter._dedup.is_duplicate("spaces/S/messages/DUP.DUP")
        msg = _make_pubsub_message(env)
        with patch.object(adapter, "_submit_on_loop") as submit:
            adapter._on_pubsub_message(msg)
            submit.assert_not_called()
        msg.ack.assert_called_once()

    def test_text_message_submits_to_loop(self, adapter):
        env = _make_chat_envelope(text="hola")
        msg = _make_pubsub_message(env)
        with patch.object(adapter, "_submit_on_loop") as submit:
            adapter._on_pubsub_message(msg)
            submit.assert_called_once()
        msg.ack.assert_called_once()

    def test_callback_exception_does_not_escape(self, adapter):
        env = _make_chat_envelope(text="hola")
        msg = _make_pubsub_message(env)
        with patch.object(
            adapter, "_submit_on_loop", side_effect=RuntimeError("boom")
        ):
            # Must not re-raise (would trigger Pub/Sub infinite redelivery).
            adapter._on_pubsub_message(msg)
        msg.ack.assert_called_once()


# ===========================================================================
# _build_message_event — payload parsing
# ===========================================================================


class TestBuildMessageEvent:
    @pytest.mark.asyncio
    async def test_dm_drops_thread_id_from_source_for_session_continuity(self, adapter):
        """Google Chat DMs spawn a new thread per top-level user message.
        We deliberately DROP thread_id from the source for DMs so the
        session_key (which includes thread_id when present) stays stable
        across top-level messages and the agent retains conversation
        memory. The thread is still cached for outbound reply placement."""
        env = _make_chat_envelope(text="hola", thread_name="spaces/S/threads/T1")
        msg = env["chat"]["messagePayload"]["message"]
        event = await adapter._build_message_event(msg, env)
        assert event is not None
        assert event.text == "hola"
        assert event.message_type == MessageType.TEXT
        assert event.source.chat_id == "spaces/S"
        # DM = thread_id NOT propagated to the source.
        assert event.source.thread_id is None
        assert event.source.user_id_alt == "u@example.com"
        # But the thread IS cached so outbound replies stay connected.
        assert adapter._last_inbound_thread["spaces/S"] == "spaces/S/threads/T1"

    @pytest.mark.asyncio
    async def test_group_keeps_thread_id_on_source(self, adapter):
        """In group spaces, threads are real conversational containers —
        keep thread_id on the source so different threads get isolated
        sessions (Telegram forum / Discord thread parity)."""
        env = _make_chat_envelope(text="ping", thread_name="spaces/G/threads/T1")
        # Force chat_type=group by making space type a regular SPACE.
        env["chat"]["messagePayload"]["space"]["spaceType"] = "SPACE"
        env["chat"]["messagePayload"]["message"]["space"]["spaceType"] = "SPACE"
        msg = env["chat"]["messagePayload"]["message"]
        event = await adapter._build_message_event(msg, env)
        assert event.source.chat_type == "group"
        assert event.source.thread_id == "spaces/G/threads/T1"

    @pytest.mark.asyncio
    async def test_slash_command_yields_command_type(self, adapter):
        env = _make_chat_envelope(
            text="foo bar",
            slash_command={"commandId": "42"},
        )
        msg = env["chat"]["messagePayload"]["message"]
        event = await adapter._build_message_event(msg, env)
        assert event.message_type == MessageType.COMMAND
        assert event.text.startswith("/cmd_42")

    @pytest.mark.asyncio
    async def test_attachment_image_triggers_download(self, adapter):
        attachments = [{
            "name": "att/img.png",
            "contentType": "image/png",
            "downloadUri": "https://chat.googleapis.com/media/x",
        }]
        env = _make_chat_envelope(text="", attachments=attachments)
        msg = env["chat"]["messagePayload"]["message"]
        with patch.object(
            adapter, "_download_attachment",
            new=AsyncMock(return_value=("/cache/img.png", "image/png")),
        ):
            event = await adapter._build_message_event(msg, env)
        assert event.media_urls == ["/cache/img.png"]
        assert event.media_types == ["image/png"]
        # With no text, the message type should reflect the first attachment.
        assert event.message_type == MessageType.PHOTO


# ===========================================================================
# send() — text, patch-in-place, chunking, error handling
# ===========================================================================


class TestSend:
    @pytest.mark.asyncio
    async def test_text_send_creates_message(self, adapter):
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m/1",
                                        "error": None})()
        )
        result = await adapter.send("spaces/S", "hola")
        adapter._create_message.assert_called()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_create_message_passes_messageReplyOption_when_thread_set(self, adapter):
        """Critical Google Chat API quirk: when messages.create is called
        with body.thread.name set BUT WITHOUT messageReplyOption query
        param, Google SILENTLY ignores the thread and creates a new
        thread. From official docs: 'Default. Starts a new thread.
        Using this option ignores any thread ID or threadKey that's
        included.'

        This test pins down the messageReplyOption=
        REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD parameter so a future
        refactor doesn't silently regress threading. (The user-visible
        symptom of regression: bot replies land at top-level instead of
        inside the user's thread.)"""
        # Capture the kwargs handed to .create() — this is what hits
        # Google's API. The mock chain is: spaces() -> messages() ->
        # create(**kwargs) -> .execute(...).
        create_call = MagicMock()
        create_call.return_value.execute = MagicMock(
            return_value={"name": "spaces/S/messages/M"}
        )
        adapter._chat_api.spaces.return_value.messages.return_value.create = create_call

        body = {
            "text": "respuesta",
            "thread": {"name": "spaces/S/threads/USER_THREAD"},
        }
        await adapter._create_message("spaces/S", body)
        kwargs = create_call.call_args.kwargs
        assert kwargs.get("parent") == "spaces/S"
        assert kwargs.get("body") == body
        assert kwargs.get("messageReplyOption") == "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

    @pytest.mark.asyncio
    async def test_create_message_omits_messageReplyOption_when_no_thread(self, adapter):
        """No thread.name in body → no messageReplyOption needed.
        Sending it would imply a thread intent we don't have."""
        create_call = MagicMock()
        create_call.return_value.execute = MagicMock(
            return_value={"name": "spaces/S/messages/M"}
        )
        adapter._chat_api.spaces.return_value.messages.return_value.create = create_call

        await adapter._create_message("spaces/S", {"text": "hola"})
        kwargs = create_call.call_args.kwargs
        assert "messageReplyOption" not in kwargs

    @pytest.mark.asyncio
    async def test_with_typing_card_patches_instead_of_creating(self, adapter):
        adapter._typing_messages["spaces/S"] = "spaces/S/messages/THINK"
        adapter._patch_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/THINK",
                                        "error": None})()
        )
        adapter._create_message = AsyncMock()
        result = await adapter.send(
            "spaces/S", "hola",
            metadata={"thread_id": "spaces/S/threads/T"},
        )
        adapter._patch_message.assert_awaited_once()
        adapter._create_message.assert_not_called()
        assert result.success is True
        # After patch, the typing slot holds the consumed sentinel so the
        # base class's _keep_typing loop cannot post a fresh marker that
        # the cleanup pass would later delete and tombstone.
        from gateway.platforms.google_chat import _TYPING_CONSUMED_SENTINEL
        assert adapter._typing_messages["spaces/S"] == _TYPING_CONSUMED_SENTINEL

    @pytest.mark.asyncio
    async def test_long_text_splits_and_sends_multiple(self, adapter):
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        long_text = "x" * 9000
        await adapter.send("spaces/S", long_text)
        assert adapter._create_message.await_count >= 2

    @pytest.mark.asyncio
    async def test_403_sets_fatal_error(self, adapter):
        exc = _FakeHttpError(status=403, reason="Forbidden")
        adapter._create_message = AsyncMock(side_effect=exc)
        result = await adapter.send("spaces/S", "hola")
        assert result.success is False
        assert adapter.has_fatal_error is True

    @pytest.mark.asyncio
    async def test_404_returns_target_not_found(self, adapter):
        exc = _FakeHttpError(status=404, reason="Not Found")
        adapter._create_message = AsyncMock(side_effect=exc)
        result = await adapter.send("spaces/S", "hola")
        assert result.success is False
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_429_increments_rate_limit_counter_and_raises(self, adapter):
        exc = _FakeHttpError(status=429, reason="Too Many Requests")
        adapter._create_message = AsyncMock(side_effect=exc)
        with pytest.raises(_FakeHttpError):
            await adapter.send("spaces/S", "hola")
        assert adapter._rate_limit_hits.get("spaces/S") == 1


# ===========================================================================
# send_typing / stop_typing
# ===========================================================================


class TestTypingLifecycle:
    @pytest.mark.asyncio
    async def test_send_typing_posts_and_tracks(self, adapter):
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/THINK",
                                        "error": None})()
        )
        await adapter.send_typing("spaces/S")
        adapter._create_message.assert_awaited_once()
        assert adapter._typing_messages["spaces/S"] == "spaces/S/messages/THINK"

    @pytest.mark.asyncio
    async def test_send_typing_skips_when_already_tracking(self, adapter):
        adapter._typing_messages["spaces/S"] = "spaces/S/messages/EXIST"
        adapter._create_message = AsyncMock()
        await adapter.send_typing("spaces/S")
        adapter._create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_typing_inherits_inbound_thread(self, adapter):
        """The typing card must be created in the same thread as the
        user's message, otherwise send() will patch a top-level card and
        the bot's whole reply ends up outside the user's thread (Chat
        messages.patch cannot change thread — it's immutable). Regression
        test for the 'reply lands at top-level instead of in my thread'
        UX bug."""
        adapter._last_inbound_thread["spaces/S"] = "spaces/S/threads/USER_THREAD"
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/THINK",
                                        "error": None})()
        )
        await adapter.send_typing("spaces/S")
        # Verify the body sent to _create_message included the thread.
        sent_body = adapter._create_message.call_args.args[1]
        assert sent_body.get("thread") == {"name": "spaces/S/threads/USER_THREAD"}

    @pytest.mark.asyncio
    async def test_send_typing_no_thread_when_cache_empty(self, adapter):
        """If no inbound thread has been seen yet, typing card creates
        without thread (Chat will assign a default). Defensive — first
        bot push without prior user message."""
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/THINK",
                                        "error": None})()
        )
        await adapter.send_typing("spaces/S")
        sent_body = adapter._create_message.call_args.args[1]
        assert "thread" not in sent_body

    @pytest.mark.asyncio
    async def test_send_typing_concurrent_calls_create_only_one_card(self, adapter):
        """When _keep_typing fires send_typing twice in flight (the
        first call slow, the second arriving before the first stores
        its msg_id), only ONE create should hit the API. Without this
        guard the second call would create a duplicate card → orphan
        'Hermes is thinking…' stuck in chat. Race fix via
        _typing_card_inflight Event.
        """
        call_count = 0
        first_call_started = asyncio.Event()
        release_first_call = asyncio.Event()

        async def _slow_create(chat_id, body):
            nonlocal call_count
            call_count += 1
            first_call_started.set()
            await release_first_call.wait()
            return type("R", (), {"success": True,
                                  "message_id": f"spaces/S/messages/CARD_{call_count}",
                                  "error": None})()

        adapter._create_message = _slow_create

        # Fire two send_typing tasks concurrently (mimics _keep_typing
        # firing while a previous tick is still in-flight).
        t1 = asyncio.create_task(adapter.send_typing("spaces/S"))
        await first_call_started.wait()
        t2 = asyncio.create_task(adapter.send_typing("spaces/S"))
        # Give t2 a moment to bail out via the in-flight check.
        await asyncio.sleep(0.05)
        # Release the first call to complete.
        release_first_call.set()
        await asyncio.gather(t1, t2)

        assert call_count == 1
        assert adapter._typing_messages["spaces/S"] == "spaces/S/messages/CARD_1"

    @pytest.mark.asyncio
    async def test_send_typing_survives_caller_cancellation(self, adapter):
        """base.py's _keep_typing wraps send_typing in
        asyncio.wait_for(timeout=1.5). When the create-API call takes
        longer than 1.5s, wait_for cancels the awaiter — but the create
        itself MUST complete and the msg_id MUST land in the slot,
        otherwise the next tick spawns a SECOND card (orphan).

        This test simulates that: cancel the awaiter while the create
        is in flight. The shielded background task should still
        populate the slot.
        """
        first_call_started = asyncio.Event()
        release_first_call = asyncio.Event()

        async def _slow_create(chat_id, body):
            first_call_started.set()
            await release_first_call.wait()
            return type("R", (), {"success": True,
                                  "message_id": "spaces/S/messages/CARD_X",
                                  "error": None})()

        adapter._create_message = _slow_create

        task = asyncio.create_task(adapter.send_typing("spaces/S"))
        await first_call_started.wait()
        # Simulate wait_for timeout cancelling the awaiter.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The shielded background create is still running. Release it.
        release_first_call.set()
        # Give the background task time to complete + record.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if "spaces/S" in adapter._typing_messages:
                break
        # Slot SHOULD be populated despite the cancellation.
        assert adapter._typing_messages.get("spaces/S") == "spaces/S/messages/CARD_X"

    @pytest.mark.asyncio
    async def test_orphan_typing_cards_reaped_on_completion(self, adapter):
        """If a background send_typing task created a card AFTER send()
        already populated the slot (race), the orphan id is tracked in
        _orphan_typing_messages. on_processing_complete must patch each
        orphan to a benign marker so users don't see stuck
        'Hermes is thinking…' messages."""
        from gateway.platforms.google_chat import _TYPING_CONSUMED_SENTINEL
        adapter._orphan_typing_messages["spaces/S"] = [
            "spaces/S/messages/ORPHAN1",
            "spaces/S/messages/ORPHAN2",
        ]
        adapter._typing_messages["spaces/S"] = _TYPING_CONSUMED_SENTINEL
        adapter._patch_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "x",
                                        "error": None})()
        )
        event = MagicMock()
        event.source = MagicMock()
        event.source.chat_id = "spaces/S"
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        # Both orphans patched (typing_messages cleared too).
        assert adapter._patch_message.await_count == 2
        patched_ids = [
            call.args[0] for call in adapter._patch_message.call_args_list
        ]
        assert "spaces/S/messages/ORPHAN1" in patched_ids
        assert "spaces/S/messages/ORPHAN2" in patched_ids
        assert "spaces/S" not in adapter._orphan_typing_messages

    @pytest.mark.asyncio
    async def test_stop_typing_is_noop_for_live_card(self, adapter):
        """Anti-tombstone: stop_typing leaves a real msg_id in place so
        send() can patch it. Deleting would create a "Message deleted by
        its author" tombstone."""
        adapter._typing_messages["spaces/S"] = "spaces/S/messages/THINK"
        delete_mock = MagicMock()
        delete_mock.return_value.execute = MagicMock(return_value={})
        adapter._chat_api.spaces.return_value.messages.return_value.delete = delete_mock

        await adapter.stop_typing("spaces/S")
        # Slot retained, no API delete fired.
        assert adapter._typing_messages["spaces/S"] == "spaces/S/messages/THINK"
        delete_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_typing_pops_sentinel(self, adapter):
        """After send() patches the typing card, the slot holds the
        sentinel; stop_typing pops it so the next turn starts fresh."""
        from gateway.platforms.google_chat import _TYPING_CONSUMED_SENTINEL
        adapter._typing_messages["spaces/S"] = _TYPING_CONSUMED_SENTINEL
        await adapter.stop_typing("spaces/S")
        assert "spaces/S" not in adapter._typing_messages

    @pytest.mark.asyncio
    async def test_stop_typing_noop_when_nothing_tracked(self, adapter):
        delete_mock = MagicMock()
        adapter._chat_api.spaces.return_value.messages.return_value.delete = delete_mock
        await adapter.stop_typing("spaces/S")
        delete_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_processing_complete_pops_sentinel_on_success(self, adapter):
        """SUCCESS path: send() set the sentinel; cleanup just pops it."""
        from gateway.platforms.google_chat import _TYPING_CONSUMED_SENTINEL
        adapter._typing_messages["spaces/S"] = _TYPING_CONSUMED_SENTINEL
        adapter._patch_message = AsyncMock()
        event = MagicMock()
        event.source = MagicMock()
        event.source.chat_id = "spaces/S"
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert "spaces/S" not in adapter._typing_messages
        adapter._patch_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_processing_complete_patches_stranded_card(self, adapter):
        """CANCELLED path: send() never ran. Patch the typing card with a
        benign final state instead of deleting (no tombstone)."""
        adapter._typing_messages["spaces/S"] = "spaces/S/messages/THINK"
        adapter._patch_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/THINK",
                                        "error": None})()
        )
        event = MagicMock()
        event.source = MagicMock()
        event.source.chat_id = "spaces/S"
        await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)
        adapter._patch_message.assert_awaited_once()
        # Patched with a final-state label, not deleted.
        args, kwargs = adapter._patch_message.call_args
        assert "interrupted" in args[1]["text"].lower()
        assert "spaces/S" not in adapter._typing_messages


# ===========================================================================
# edit_message / delete_message — required by gateway tool-progress + streaming
# ===========================================================================


class TestEditMessage:
    @pytest.mark.asyncio
    async def test_edit_message_patches_via_messages_patch(self, adapter):
        adapter._patch_message = AsyncMock(
            return_value=type("R", (), {"success": True,
                                        "message_id": "spaces/S/messages/M",
                                        "error": None})()
        )
        result = await adapter.edit_message(
            "spaces/S", "spaces/S/messages/M", "edited content",
        )
        assert result.success is True
        adapter._patch_message.assert_awaited_once_with(
            "spaces/S/messages/M", {"text": "edited content"},
        )

    @pytest.mark.asyncio
    async def test_edit_message_truncates_overlong_text(self, adapter):
        adapter._patch_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        long_text = "x" * 9000
        await adapter.edit_message("spaces/S", "spaces/S/messages/M", long_text)
        sent = adapter._patch_message.call_args[0][1]["text"]
        # Truncated to MAX_MESSAGE_LENGTH (4000) with ellipsis.
        assert len(sent) <= 4000

    @pytest.mark.asyncio
    async def test_edit_message_missing_id_returns_failure(self, adapter):
        result = await adapter.edit_message("spaces/S", "", "x")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_edit_message_429_increments_rate_limit_counter(self, adapter):
        exc = _FakeHttpError(status=429, reason="Too Many Requests")
        adapter._patch_message = AsyncMock(side_effect=exc)
        result = await adapter.edit_message(
            "spaces/S", "spaces/S/messages/M", "content",
        )
        assert result.success is False
        assert adapter._rate_limit_hits.get("spaces/S") == 1

    @pytest.mark.asyncio
    async def test_edit_message_overrides_base_so_progress_pipeline_runs(self, adapter):
        """The gateway tool-progress flow at gateway/run.py:10199 gates on
        ``type(adapter).edit_message is BasePlatformAdapter.edit_message``.
        If our subclass doesn't override edit_message, no tool progress is
        ever shown to the user — so this test guards against a future
        accidental removal."""
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.platforms.google_chat import GoogleChatAdapter
        assert GoogleChatAdapter.edit_message is not BasePlatformAdapter.edit_message


class TestDeleteMessage:
    @pytest.mark.asyncio
    async def test_delete_message_calls_api(self, adapter):
        delete_mock = MagicMock()
        delete_mock.return_value.execute = MagicMock(return_value={})
        adapter._chat_api.spaces.return_value.messages.return_value.delete = delete_mock
        result = await adapter.delete_message("spaces/S", "spaces/S/messages/M")
        assert result is True
        delete_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_message_swallows_404(self, adapter):
        exc = _FakeHttpError(status=404, reason="Not Found")
        delete_mock = MagicMock()
        delete_mock.return_value.execute = MagicMock(side_effect=exc)
        adapter._chat_api.spaces.return_value.messages.return_value.delete = delete_mock
        assert await adapter.delete_message("spaces/S", "spaces/S/messages/M") is False

    @pytest.mark.asyncio
    async def test_delete_message_missing_id_returns_false(self, adapter):
        assert await adapter.delete_message("spaces/S", "") is False


# ===========================================================================
# Native attachment delivery via user OAuth
#
# Google Chat's media.upload endpoint hard-rejects bot/SA auth, so the
# adapter calls it through a SEPARATE user-authed Chat API client built
# from a refresh token the user grants once via /setup-files.
# These tests cover:
#   - _send_file falls back to text notice when no user creds present
#   - _send_file does the two-step upload + create-with-attachment when
#     user creds ARE present
#   - the /setup-files slash command intercepts before the agent
#   - 401/403 from media.upload triggers a clean fallback (token revoked)
# ===========================================================================


class TestNativeAttachmentDelivery:
    @pytest.mark.asyncio
    async def test_send_file_posts_setup_notice_when_no_user_oauth(self, adapter, tmp_path):
        """Without user creds, _send_file posts a clear setup notice and
        returns success=False so callers know delivery did not land."""
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-fake")
        adapter._user_chat_api = None
        adapter._user_credentials = None
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m/notice",
                                        "error": None})()
        )

        result = await adapter._send_file(
            "spaces/S", str(f), caption="Aquí va el PDF",
            mime_hint="application/pdf",
        )
        assert result.success is False
        adapter._create_message.assert_awaited()
        sent_body = adapter._create_message.call_args.args[1]
        assert "/setup-files" in sent_body["text"]
        assert "report.pdf" in sent_body["text"]

    @pytest.mark.asyncio
    async def test_send_file_two_step_native_upload_when_user_oauth_ready(self, adapter, tmp_path):
        """With user creds, _send_file calls media.upload then
        messages.create with the attachmentDataRef — both via the
        user-authed Chat client."""
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-fake")

        upload_call = MagicMock()
        upload_call.return_value.execute = MagicMock(
            return_value={"attachmentDataRef": {"resourceName": "ref-abc"}}
        )
        create_call = MagicMock()
        create_call.return_value.execute = MagicMock(
            return_value={"name": "spaces/S/messages/MID"}
        )
        adapter._user_chat_api = MagicMock()
        adapter._user_chat_api.media.return_value.upload = upload_call
        adapter._user_chat_api.spaces.return_value.messages.return_value.create = create_call
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)

        result = await adapter._send_file(
            "spaces/S", str(f), caption="caption",
            mime_hint="application/pdf",
            thread_id="spaces/S/threads/T",
        )

        assert result.success is True
        upload_call.assert_called_once()
        create_call.assert_called_once()
        # Verify the messages.create body referenced the attachment ref.
        body_passed = create_call.call_args.kwargs["body"]
        assert body_passed["attachment"][0]["attachmentDataRef"] == {
            "resourceName": "ref-abc"
        }

    @pytest.mark.asyncio
    async def test_send_file_falls_back_to_notice_on_401(self, adapter, tmp_path):
        """A 401 from media.upload (token revoked / scope missing) should
        clear in-memory creds and post the setup notice."""
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-fake")
        upload_call = MagicMock()
        upload_call.return_value.execute = MagicMock(
            side_effect=_FakeHttpError(status=401, reason="Unauthorized")
        )
        adapter._user_chat_api = MagicMock()
        adapter._user_chat_api.media.return_value.upload = upload_call
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )

        result = await adapter._send_file(
            "spaces/S", str(f), caption=None,
            mime_hint="application/pdf",
        )
        assert result.success is False
        # In-memory creds cleared so subsequent uploads short-circuit.
        assert adapter._user_chat_api is None
        assert adapter._user_credentials is None
        # User saw a setup notice.
        adapter._create_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_send_file_returns_error_on_unrelated_http_error(self, adapter, tmp_path):
        """Non-auth HTTP errors propagate as SendResult.error without
        clearing user creds (transient failures shouldn't disable the
        feature)."""
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-fake")
        upload_call = MagicMock()
        upload_call.return_value.execute = MagicMock(
            side_effect=_FakeHttpError(status=500, reason="Server error")
        )
        adapter._user_chat_api = MagicMock()
        adapter._user_chat_api.media.return_value.upload = upload_call
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)

        result = await adapter._send_file(
            "spaces/S", str(f), caption=None,
            mime_hint="application/pdf",
        )
        assert result.success is False
        assert "500" in (result.error or "")
        # Creds NOT cleared on transient failure.
        assert adapter._user_chat_api is not None


class TestSetupFilesSlashCommand:
    @pytest.mark.asyncio
    async def test_slash_command_intercepted_before_agent(self, adapter):
        """/setup-files is bot-side admin, not agent input. The dispatch
        path must short-circuit and not call handle_message."""
        adapter._handle_setup_files_command = AsyncMock(return_value=True)
        adapter._build_message_event = AsyncMock(
            return_value=MessageEvent(
                text="/setup-files",
                message_type=MessageType.TEXT,
                source=adapter.build_source(
                    chat_id="spaces/S",
                    chat_name="DM",
                    chat_type="dm",
                    user_id="users/1",
                    user_name="Ramón",
                    thread_id="spaces/S/threads/T",
                ),
                raw_message={},
                message_id="spaces/S/messages/M",
            )
        )
        await adapter._dispatch_message({}, {})
        adapter._handle_setup_files_command.assert_awaited_once()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_arg_status_when_unconfigured(self, adapter, tmp_path, monkeypatch):
        """Without client_secret AND without token, status reply tells the
        user how to provide credentials on the host."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        handled = await adapter._handle_setup_files_command(
            chat_id="spaces/S",
            thread_id="spaces/S/threads/T",
            raw_text="/setup-files",
        )
        assert handled is True
        sent = adapter._create_message.call_args.args[1]["text"]
        assert "client_secret.json" in sent or "Create credentials" in sent

    @pytest.mark.asyncio
    async def test_revoke_clears_in_memory_creds(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._user_chat_api = MagicMock()
        adapter._user_credentials = MagicMock(valid=True)
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        await adapter._handle_setup_files_command(
            chat_id="spaces/S",
            thread_id=None,
            raw_text="/setup-files revoke",
        )
        assert adapter._user_chat_api is None
        assert adapter._user_credentials is None


class TestUserOAuthHelper:
    def test_load_user_credentials_returns_none_when_no_token(self, tmp_path, monkeypatch):
        """Missing token file is the expected no-op case (user hasn't
        run /setup-files yet). Must NOT raise."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.google_chat_user_oauth import load_user_credentials
        assert load_user_credentials() is None

    def test_load_user_credentials_returns_none_on_corrupt_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "google_chat_user_token.json").write_text("not json")
        from gateway.platforms.google_chat_user_oauth import load_user_credentials
        assert load_user_credentials() is None

    def test_scopes_are_minimal(self):
        """The OAuth flow should request ONLY chat.messages.create — no
        Drive, no broader Chat scopes. Defends against scope creep."""
        from gateway.platforms.google_chat_user_oauth import SCOPES
        assert SCOPES == ["https://www.googleapis.com/auth/chat.messages.create"]


# ===========================================================================
# Inbound attachment download SSRF guard
# ===========================================================================


class TestAttachmentSSRFGuard:
    @pytest.mark.asyncio
    async def test_drive_picker_only_skipped_when_no_resource_name(self, adapter):
        """Pure Drive-picker shares (source=DRIVE_FILE, no resourceName)
        cannot be downloaded with bot SA — skip silently."""
        attachment = {
            "source": "DRIVE_FILE",
            "contentType": "application/pdf",
            "downloadUri": "https://drive.google.com/file/d/abc",
        }
        path, mime = await adapter._download_attachment(attachment)
        assert path is None
        assert mime == "application/pdf"

    @pytest.mark.asyncio
    async def test_drive_file_with_resource_name_uses_bot_path(self, adapter, tmp_path, monkeypatch):
        """Drag-and-drop chat uploads ALSO carry source=DRIVE_FILE but
        come with attachmentDataRef.resourceName — bot media.download_media
        works against those. Regression test for the original bug where
        we skipped them all (left users with 'I don't see any PDF')."""
        attachment = {
            "source": "DRIVE_FILE",
            "contentType": "application/pdf",
            "name": "spaces/S/messages/M/attachments/A",
            "attachmentDataRef": {
                "resourceName": "spaces/S/messages/M/attachments/A",
            },
        }

        # Patch the inner _fetch_media path by hijacking asyncio.to_thread
        # — return some bytes directly, no need to walk the full
        # google-api-client mock chain.
        async def _fake_to_thread(fn, *args, **kwargs):
            return b"%PDF-fake"

        monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
        from gateway.platforms import google_chat as gc_mod
        monkeypatch.setattr(
            gc_mod, "cache_document_from_bytes",
            lambda data, ext=None, filename=None: str(tmp_path / "out.pdf"),
            raising=False,
        )

        path, mime = await adapter._download_attachment(attachment)
        assert path == str(tmp_path / "out.pdf")
        assert mime == "application/pdf"

    @pytest.mark.asyncio
    async def test_rejects_non_google_host(self, adapter):
        attachment = {
            "contentType": "image/png",
            "downloadUri": "https://evil.com/steal",
        }
        path, mime = await adapter._download_attachment(attachment)
        assert path is None
        assert mime == "image/png"

    @pytest.mark.asyncio
    async def test_rejects_metadata_endpoint(self, adapter):
        attachment = {
            "contentType": "image/png",
            "downloadUri": "https://169.254.169.254/computeMetadata/v1/",
        }
        path, mime = await adapter._download_attachment(attachment)
        assert path is None


# ===========================================================================
# Outbound thread routing (anti-top-level fallback in DMs)
# ===========================================================================


class TestOutboundThreadRouting:
    def test_resolve_uses_metadata_thread_id(self, adapter):
        result = adapter._resolve_thread_id(
            reply_to=None,
            metadata={"thread_id": "spaces/X/threads/EXPLICIT"},
            chat_id="spaces/X",
        )
        assert result == "spaces/X/threads/EXPLICIT"

    def test_resolve_falls_back_to_cached_thread_for_dm(self, adapter):
        """In DMs the source.thread_id is None, so the metadata passed
        to send() lacks a thread. Without the cache fallback, replies
        would land at top-level (visually disconnected from the user's
        thread)."""
        adapter._last_inbound_thread["spaces/X"] = "spaces/X/threads/CACHED"
        result = adapter._resolve_thread_id(
            reply_to=None,
            metadata=None,
            chat_id="spaces/X",
        )
        assert result == "spaces/X/threads/CACHED"

    def test_resolve_metadata_overrides_cache(self, adapter):
        """Explicit metadata (e.g. agent replying to a specific event)
        wins over the cached thread."""
        adapter._last_inbound_thread["spaces/X"] = "spaces/X/threads/CACHED"
        result = adapter._resolve_thread_id(
            reply_to=None,
            metadata={"thread_id": "spaces/X/threads/EXPLICIT"},
            chat_id="spaces/X",
        )
        assert result == "spaces/X/threads/EXPLICIT"

    def test_resolve_returns_none_when_no_inputs(self, adapter):
        result = adapter._resolve_thread_id(
            reply_to=None, metadata=None, chat_id="spaces/UNKNOWN",
        )
        assert result is None


# ===========================================================================
# Send file delegation (voice/video/animation route through send_document)
# ===========================================================================


class TestMediaDelegation:
    @pytest.mark.asyncio
    async def test_send_voice_delegates_to_document_with_audio_mime(self, adapter, tmp_path):
        f = tmp_path / "voice.ogg"
        f.write_bytes(b"audio-bytes")
        adapter._send_file = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        await adapter.send_voice("spaces/S", str(f))
        _, kwargs = adapter._send_file.await_args
        assert kwargs.get("mime_hint") == "audio/ogg"

    @pytest.mark.asyncio
    async def test_send_video_delegates_with_video_mime(self, adapter, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"video-bytes")
        adapter._send_file = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        await adapter.send_video("spaces/S", str(f))
        _, kwargs = adapter._send_file.await_args
        assert kwargs.get("mime_hint") == "video/mp4"

    @pytest.mark.asyncio
    async def test_send_animation_delegates_to_image(self, adapter):
        """Google Chat has no native animation type; the adapter falls back
        to send_image (which posts the URL inline). Animations and images
        share the same render path on Chat so we just delegate."""
        adapter.send_image = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        await adapter.send_animation(
            "spaces/S", "https://example.com/dance.gif", caption="hop"
        )
        adapter.send_image.assert_awaited_once()
        args, kwargs = adapter.send_image.await_args
        assert args[1] == "https://example.com/dance.gif"
        assert kwargs.get("caption") == "hop"

    @pytest.mark.asyncio
    async def test_send_file_missing_path_returns_error(self, adapter):
        result = await adapter._send_file("spaces/S", "/no/such/file.pdf",
                                          None, mime_hint="application/pdf")
        assert result.success is False
        assert "not found" in (result.error or "").lower()


# ===========================================================================
# Supervisor reconnect (backoff + fatal)
# ===========================================================================


class TestSupervisorReconnect:
    @pytest.mark.asyncio
    async def test_fatal_after_max_retries(self, adapter, monkeypatch):
        """Simulate 10+ failing subscribe() calls and assert fatal error set."""
        # Stub out sleep so the test doesn't actually wait minutes.
        async def _instant(*args, **kwargs):
            return None
        monkeypatch.setattr(
            "gateway.platforms.google_chat.asyncio.sleep", _instant
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("stream died")
        adapter._subscriber.subscribe = _fail

        # Keep the test fast — run supervisor until it exhausts retries.
        await adapter._run_supervisor()
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "pubsub_reconnect_exhausted"


# ===========================================================================
# Authorization: email-path check via user_id_alt
# ===========================================================================


class TestAuthorizationEmailMatch:
    def test_allowlist_matches_via_user_id_alt(self, monkeypatch):
        """Regression test: GOOGLE_CHAT_ALLOWED_USERS with email values must
        match against source.user_id_alt (email) not just source.user_id
        (which for Google Chat is 'users/{id}'). Without this, the setup
        wizard's email prompt produces a silent auth-denial."""
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "alice@example.com")
        cfg = GatewayConfig()
        runner = GatewayRunner(cfg)
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved = MagicMock(return_value=False)

        source = SessionSource(
            platform=Platform.GOOGLE_CHAT,
            chat_id="spaces/S",
            chat_type="dm",
            user_id="users/12345",
            user_name="Alice",
            user_id_alt="alice@example.com",
        )
        assert runner._is_user_authorized(source) is True

    def test_allowlist_denies_wrong_email(self, monkeypatch):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "alice@example.com")
        cfg = GatewayConfig()
        runner = GatewayRunner(cfg)
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved = MagicMock(return_value=False)

        source = SessionSource(
            platform=Platform.GOOGLE_CHAT,
            chat_id="spaces/S",
            chat_type="dm",
            user_id="users/99999",
            user_name="Bob",
            user_id_alt="bob@example.com",
        )
        assert runner._is_user_authorized(source) is False


# ===========================================================================
# Cron scheduler registry (regression guard from /review)
# ===========================================================================


class TestCronSchedulerRegistry:
    def test_google_chat_in_known_delivery_platforms(self):
        from cron.scheduler import _KNOWN_DELIVERY_PLATFORMS

        assert "google_chat" in _KNOWN_DELIVERY_PLATFORMS

    def test_google_chat_in_home_target_env_vars(self):
        from cron.scheduler import _HOME_TARGET_ENV_VARS

        assert _HOME_TARGET_ENV_VARS.get("google_chat") == "GOOGLE_CHAT_HOME_CHANNEL"
