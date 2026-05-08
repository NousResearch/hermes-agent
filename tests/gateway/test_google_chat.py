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
import types
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
#
# Note on imports: Teams' test suite uses
# ``tests.gateway._plugin_adapter_loader.load_plugin_adapter`` to load
# its adapter under a unique ``plugin_adapter_<name>`` module name. That
# helper assumes the plugin is a single ``adapter.py`` file with no
# companion modules — it does not set ``__package__`` on the loaded
# module, so any relative import (e.g. our adapter's ``from .oauth import``)
# raises ``ImportError: attempted relative import with no known parent
# package``.
#
# Our google_chat plugin has a companion ``oauth.py`` module (the
# OAuth helper for native attachment delivery), so we need a real package
# context. The fully-qualified package import below resolves correctly
# because ``plugins/__init__.py`` and ``plugins/platforms/__init__.py``
# exist as regular packages on disk. The conftest anti-pattern guard
# (which targets bare ``import adapter`` / ``from adapter import …`` and
# ``sys.path.insert`` into ``plugins/platforms/``) does not flag this
# fully-qualified form.
import plugins.platforms.google_chat.adapter as _gc_mod  # noqa: E402

_gc_mod.GOOGLE_CHAT_AVAILABLE = True

from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome  # noqa: E402
from plugins.platforms.google_chat.adapter import (  # noqa: E402
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
def adapter(tmp_path):
    """Build an adapter with its loop captured and Chat client mocked.

    Redirects the persistent thread-count store to a tmp file so tests
    don't pollute (or read state from) the developer's real
    ~/.hermes/google_chat_thread_counts.json.
    """
    from plugins.platforms.google_chat.adapter import _ThreadCountStore
    a = GoogleChatAdapter(_base_config())
    a._loop = asyncio.get_event_loop_policy().new_event_loop()
    a._chat_api = MagicMock()
    a._subscriber = MagicMock()
    a._credentials = MagicMock()
    a._project_id = "test-project"
    a._subscription_path = "projects/test-project/subscriptions/test-sub"
    a._new_authed_http = MagicMock(return_value=MagicMock())
    a.handle_message = AsyncMock()
    # Replace the production store (which would write to ~/.hermes/...)
    # with a tmp-path one so tests can roundtrip without side effects.
    a._thread_count_store = _ThreadCountStore(
        tmp_path / "google_chat_thread_counts.json"
    )
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


class TestExtractMessagePayload:
    """Three Pub/Sub envelope formats are accepted.

    The Workspace Add-ons format (current default) was already exercised
    by the rest of TestOnPubsubMessage; these tests pin the contract for
    the two alternative formats so the multi-format helper does not
    regress when operators have non-standard Chat app configurations.

    Patterns adapted from PR #14965 by @ArnarValur.
    """

    def test_native_chat_api_format_extracts_msg_and_space(self):
        """Format 2: top-level ``message`` + ``space`` + ``type=MESSAGE``.

        Used by Chat apps configured WITHOUT the Workspace Add-ons
        wrapper — events arrive directly from the Chat API publisher.
        """
        envelope = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/S/messages/M.M",
                "sender": {
                    "name": "users/12345",
                    "email": "alice@example.com",
                    "displayName": "Alice",
                    "type": "HUMAN",
                },
                "text": "hello",
                "argumentText": "hello",
                "thread": {"name": "spaces/S/threads/T"},
            },
            "space": {"name": "spaces/S", "spaceType": "DIRECT_MESSAGE"},
        }
        result = GoogleChatAdapter._extract_message_payload(envelope, ce_type="")
        assert result is not None
        msg, space, fmt = result
        assert fmt == "native_chat_api"
        assert msg.get("name") == "spaces/S/messages/M.M"
        assert msg.get("sender", {}).get("email") == "alice@example.com"
        assert space.get("name") == "spaces/S"
        assert space.get("spaceType") == "DIRECT_MESSAGE"

    def test_native_chat_api_format_drops_non_message_events(self):
        """Format 2 with ``type != MESSAGE`` returns None — caller acks."""
        envelope = {
            "type": "ADDED_TO_SPACE",
            "message": {"name": "spaces/S/messages/M"},
            "space": {"name": "spaces/S"},
        }
        assert GoogleChatAdapter._extract_message_payload(envelope) is None

    def test_relay_flat_format_synthesizes_chat_api_shape(self):
        """Format 3: flat fields from a custom Cloud Run relay.

        Some self-hosted setups put a relay in front of Pub/Sub to keep
        GCP credentials off the Hermes host. The relay flattens Chat
        events into top-level ``sender_email`` / ``text`` / ``space_name``
        / etc. The helper synthesizes a Chat-API-shaped ``message`` dict
        so downstream code (``_dispatch_message`` →
        ``_build_message_event``) consumes it without branching.
        """
        envelope = {
            "event_type": "MESSAGE",
            "sender_email": "bob@example.com",
            "sender_display_name": "Bob",
            "text": "ping",
            "space_name": "spaces/RELAY",
            "thread_name": "spaces/RELAY/threads/T1",
            "message_name": "spaces/RELAY/messages/M.M",
        }
        result = GoogleChatAdapter._extract_message_payload(envelope)
        assert result is not None
        msg, space, fmt = result
        assert fmt == "relay_flat"
        # Synthesized to look like the canonical Chat API shape so
        # _build_message_event reads it the same way as format 1/2.
        assert msg["text"] == "ping"
        assert msg["argumentText"] == "ping"
        assert msg["sender"]["email"] == "bob@example.com"
        assert msg["sender"]["displayName"] == "Bob"
        assert msg["sender"]["type"] == "HUMAN"
        # Resource name is unknown for relay events; helper synthesizes
        # a deterministic surrogate so dedup keys stay stable across
        # at-least-once redelivery.
        assert msg["sender"]["name"].startswith("users/relay-")
        assert msg["thread"]["name"] == "spaces/RELAY/threads/T1"
        assert msg["name"] == "spaces/RELAY/messages/M.M"
        assert space["name"] == "spaces/RELAY"

    def test_unrecognized_envelope_returns_none(self):
        """Random JSON with no known shape returns None (caller acks)."""
        envelope = {"foo": "bar", "baz": 123}
        assert GoogleChatAdapter._extract_message_payload(envelope) is None


# ===========================================================================
# _build_message_event — payload parsing
# ===========================================================================


class TestBuildMessageEvent:
    @pytest.mark.asyncio
    async def test_dm_first_message_in_thread_is_main_flow(self, adapter):
        """Google Chat DMs spawn a fresh thread per top-level user
        message in the input box. The FIRST message in any new thread
        is treated as 'main flow' — thread_id is NOT propagated to the
        source so all top-level messages share one DM session and the
        agent retains continuity. The thread is still cached for
        outbound reply placement."""
        env = _make_chat_envelope(text="hola", thread_name="spaces/S/threads/T1")
        msg = env["chat"]["messagePayload"]["message"]
        event = await adapter._build_message_event(msg, env)
        assert event is not None
        assert event.text == "hola"
        assert event.source.chat_id == "spaces/S"
        # First message in this thread → main-flow → no thread_id on source.
        assert event.source.thread_id is None
        # Identity convention (post-#14965 absorption): the sender's email
        # is the canonical ``user_id``; the Chat resource name moves to
        # ``user_id_alt`` for traceability and Chat-API operations.
        assert event.source.user_id == "u@example.com"
        assert event.source.user_id_alt == "users/12345"
        # Cache MUST be empty for main-flow so outbound bot reply lands
        # at top-level (Chat creates a separate thread for it). If we
        # cached the user's auto-thread name and replied with thread.name
        # set, Chat would show the pair as an expandable thread under
        # the user's message instead of two adjacent top-level cards.
        assert "spaces/S" not in adapter._last_inbound_thread
        # Counter populated for next-time decision (persisted store).
        assert adapter._thread_count_store.get(
            "spaces/S", "spaces/S/threads/T1"
        ) == 1

    @pytest.mark.asyncio
    async def test_dm_second_message_in_same_thread_is_side_thread(self, adapter):
        """If we've SEEN a thread before (count > 0), the user explicitly
        re-engaged it (clicked 'Reply in thread' on a prior message).
        Isolate to its own session so old top-level chatter doesn't
        leak in.

        Without this isolation the bug Ramón reported reappears: he
        opens a new thread, says 'Hola!', asks 'dime los mensajes
        anteriores' and the bot answers with messages from OTHER
        threads — because all DM threads were sharing one session."""
        env1 = _make_chat_envelope(text="primera vez", thread_name="spaces/S/threads/T1")
        msg1 = env1["chat"]["messagePayload"]["message"]
        event1 = await adapter._build_message_event(msg1, env1)
        assert event1.source.thread_id is None  # first time = main flow

        env2 = _make_chat_envelope(text="segunda vez", thread_name="spaces/S/threads/T1")
        msg2 = env2["chat"]["messagePayload"]["message"]
        event2 = await adapter._build_message_event(msg2, env2)
        # Second time same thread = user re-engaged → isolated session.
        assert event2.source.thread_id == "spaces/S/threads/T1"

    @pytest.mark.asyncio
    async def test_dm_side_thread_caches_thread_for_outbound(self, adapter):
        """When a thread is identified as side-thread, the cache MUST
        be populated so the bot's reply lands inside it. Without this
        the bot would respond at top-level and the user's threaded
        question would look unanswered."""
        # First message → main flow (cache stays clear).
        env1 = _make_chat_envelope(text="primera", thread_name="spaces/S/threads/SIDE")
        await adapter._build_message_event(
            env1["chat"]["messagePayload"]["message"], env1
        )
        assert "spaces/S" not in adapter._last_inbound_thread

        # Second message in same thread → side thread → cache populated.
        env2 = _make_chat_envelope(text="segunda", thread_name="spaces/S/threads/SIDE")
        await adapter._build_message_event(
            env2["chat"]["messagePayload"]["message"], env2
        )
        assert adapter._last_inbound_thread["spaces/S"] == "spaces/S/threads/SIDE"

    @pytest.mark.asyncio
    async def test_dm_main_flow_after_side_thread_clears_cache(self, adapter):
        """User was in a side thread, then returns to top-level (input
        box). Main-flow cache must be CLEARED so the bot reply doesn't
        accidentally land in the abandoned side thread."""
        # Two messages in T_side → side thread, cache populated.
        for _ in range(2):
            env = _make_chat_envelope(text="x", thread_name="spaces/S/threads/T_side")
            await adapter._build_message_event(
                env["chat"]["messagePayload"]["message"], env
            )
        assert adapter._last_inbound_thread["spaces/S"] == "spaces/S/threads/T_side"

        # User types in input box: NEW thread T_new (count goes 0→1, main flow).
        env_main = _make_chat_envelope(text="back to top", thread_name="spaces/S/threads/T_new")
        await adapter._build_message_event(
            env_main["chat"]["messagePayload"]["message"], env_main
        )
        # Cache cleared so outbound reply lands top-level.
        assert "spaces/S" not in adapter._last_inbound_thread

    @pytest.mark.asyncio
    async def test_dm_different_top_level_threads_share_session(self, adapter):
        """Three separate top-level user messages → three different
        thread.names from Chat. None should appear on source.thread_id
        so they all share one DM session."""
        for tid in ("T_a", "T_b", "T_c"):
            env = _make_chat_envelope(text=f"msg in {tid}",
                                      thread_name=f"spaces/S/threads/{tid}")
            msg = env["chat"]["messagePayload"]["message"]
            event = await adapter._build_message_event(msg, env)
            assert event.source.thread_id is None, (
                f"thread {tid} (count=1) should be main-flow, got isolated"
            )

    @pytest.mark.asyncio
    async def test_group_keeps_thread_id_on_source(self, adapter):
        """In group spaces, threads are real conversational containers —
        keep thread_id on the source from the FIRST message so different
        threads get isolated sessions (Telegram forum / Discord thread
        parity)."""
        env = _make_chat_envelope(text="ping", thread_name="spaces/G/threads/T1")
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
        from plugins.platforms.google_chat.adapter import _TYPING_CONSUMED_SENTINEL
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
        from plugins.platforms.google_chat.adapter import _TYPING_CONSUMED_SENTINEL
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
        from plugins.platforms.google_chat.adapter import _TYPING_CONSUMED_SENTINEL
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
        from plugins.platforms.google_chat.adapter import _TYPING_CONSUMED_SENTINEL
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
        from plugins.platforms.google_chat.adapter import GoogleChatAdapter
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
        from plugins.platforms.google_chat.oauth import load_user_credentials
        assert load_user_credentials() is None

    def test_load_user_credentials_returns_none_on_corrupt_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "google_chat_user_token.json").write_text("not json")
        from plugins.platforms.google_chat.oauth import load_user_credentials
        assert load_user_credentials() is None

    def test_scopes_are_minimal(self):
        """The OAuth flow should request ONLY chat.messages.create — no
        Drive, no broader Chat scopes. Defends against scope creep."""
        from plugins.platforms.google_chat.oauth import SCOPES
        assert SCOPES == ["https://www.googleapis.com/auth/chat.messages.create"]

    def test_sanitize_email_lowercases_and_replaces_unsafe_chars(self):
        """Path components must be filesystem-safe across users.
        ``a@B.com`` and ``A@b.com`` must collapse to the same key, and
        path-traversal characters must NOT escape into the filename."""
        from plugins.platforms.google_chat.oauth import _sanitize_email
        assert _sanitize_email("Ramon@NTTData.com") == "ramon@nttdata.com"
        assert _sanitize_email("user+tag@x.io") == "user_tag@x.io"
        # Slashes are stripped (path separator); dots inside names are
        # preserved for the .com / .json suffix UX. The resulting filename
        # is harmless when joined onto a directory.
        assert _sanitize_email("../etc/passwd") == ".._etc_passwd"
        assert _sanitize_email("") == "_unknown_"

    def test_per_user_token_path_isolated_from_legacy(self, tmp_path, monkeypatch):
        """Per-user files live under a dedicated subdirectory so the
        legacy single-user JSON stays addressable on disk."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from plugins.platforms.google_chat.oauth import (
            _token_path, _legacy_token_path,
        )
        per_user = _token_path("alice@example.com")
        legacy = _legacy_token_path()
        assert per_user.parent.name == "google_chat_user_tokens"
        assert per_user != legacy
        assert per_user.name == "alice@example.com.json"

    def test_load_user_credentials_per_email_returns_none_when_missing(
        self, tmp_path, monkeypatch
    ):
        """A user who has not authorized has no token file; load returns
        ``None`` and never throws — same contract as the legacy path."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from plugins.platforms.google_chat.oauth import load_user_credentials
        assert load_user_credentials("nobody@example.com") is None

    def test_list_authorized_emails_lists_per_user_files(
        self, tmp_path, monkeypatch
    ):
        """``list_authorized_emails`` enumerates the per-user dir; the
        legacy file is intentionally excluded (its owner is unknown)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        users_dir = tmp_path / "google_chat_user_tokens"
        users_dir.mkdir(parents=True)
        (users_dir / "alice@example.com.json").write_text("{}")
        (users_dir / "bob@example.com.json").write_text("{}")
        # Legacy file should NOT appear in the list.
        (tmp_path / "google_chat_user_token.json").write_text("{}")

        from plugins.platforms.google_chat.oauth import list_authorized_emails
        assert list_authorized_emails() == [
            "alice@example.com", "bob@example.com",
        ]

    def test_list_authorized_emails_empty_when_dir_missing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from plugins.platforms.google_chat.oauth import list_authorized_emails
        assert list_authorized_emails() == []

    def test_pending_auth_path_is_per_user_when_email_given(
        self, tmp_path, monkeypatch
    ):
        """Two users running /setup-files start in parallel must not
        clobber each other's PKCE verifier — the pending state file
        is namespaced by email."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from plugins.platforms.google_chat.oauth import _pending_auth_path
        a = _pending_auth_path("alice@example.com")
        b = _pending_auth_path("bob@example.com")
        legacy = _pending_auth_path(None)
        assert a != b
        assert a != legacy
        assert "google_chat_user_oauth_pending" in str(a.parent)


class TestPerUserAttachmentRouting:
    """The bot must use the *requesting user's* OAuth token when sending
    an attachment, not the first user who happened to have one stored.
    Backward compat: when no per-user token exists, fall back to a legacy
    single-user token; only when both are missing does the user see the
    setup-instructions notice."""

    @pytest.mark.asyncio
    async def test_build_message_event_caches_sender_email(self, adapter):
        """The asker's email is captured per chat_id at inbound time so
        a later outbound attachment can pick the right per-user token."""
        envelope = _make_chat_envelope(
            text="hi", sender_email="Alice@Example.com",
        )
        msg = envelope["chat"]["messagePayload"]["message"]
        await adapter._build_message_event(msg, envelope["chat"]["messagePayload"])
        # Lower-cased to match the on-disk sanitized key.
        assert adapter._last_sender_by_chat["spaces/S"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_send_file_uses_per_user_token_when_sender_known(
        self, adapter, tmp_path, monkeypatch
    ):
        """sender_email maps to a per-user file → that user's API client
        is built and used for the upload, NOT the legacy fallback."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        users_dir = tmp_path / "google_chat_user_tokens"
        users_dir.mkdir(parents=True)
        (users_dir / "alice@example.com.json").write_text(json.dumps({
            "type": "authorized_user",
            "client_id": "cid", "client_secret": "csec",
            "refresh_token": "rtok", "token": "atok",
        }))
        adapter._last_sender_by_chat["spaces/S"] = "alice@example.com"

        per_user_api = MagicMock()
        per_user_api.media.return_value.upload.return_value.execute.return_value = {
            "attachmentDataRef": {"resourceName": "ref-alice"}
        }
        per_user_api.spaces.return_value.messages.return_value.create.return_value.execute.return_value = {
            "name": "spaces/S/messages/MID",
            "thread": {"name": "spaces/S/threads/T"},
        }
        # Force legacy path NOT to be picked even if per-user breaks.
        adapter._user_chat_api = MagicMock()
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)

        from plugins.platforms.google_chat import oauth as helper
        with patch.object(
            helper, "load_user_credentials",
            return_value=MagicMock(valid=True),
        ), patch.object(
            helper, "build_user_chat_service", return_value=per_user_api,
        ):
            f = tmp_path / "doc.pdf"
            f.write_bytes(b"%PDF")
            result = await adapter._send_file(
                "spaces/S", str(f), caption=None,
                mime_hint="application/pdf",
            )

        assert result.success is True
        # Per-user client was used; legacy was untouched.
        per_user_api.media.return_value.upload.assert_called_once()
        adapter._user_chat_api.media.assert_not_called()
        # Cache populated for next call.
        assert "alice@example.com" in adapter._user_chat_api_by_email

    @pytest.mark.asyncio
    async def test_send_file_falls_back_to_legacy_when_per_user_missing(
        self, adapter, tmp_path, monkeypatch
    ):
        """sender known but no per-user token → legacy creds fill in.
        This is the migration window: legacy keeps working until each
        user runs /setup-files."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._last_sender_by_chat["spaces/S"] = "newuser@example.com"

        legacy_api = MagicMock()
        legacy_api.media.return_value.upload.return_value.execute.return_value = {
            "attachmentDataRef": {"resourceName": "ref-legacy"}
        }
        legacy_api.spaces.return_value.messages.return_value.create.return_value.execute.return_value = {
            "name": "spaces/S/messages/MID",
            "thread": {"name": "spaces/S/threads/T"},
        }
        adapter._user_chat_api = legacy_api
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        result = await adapter._send_file(
            "spaces/S", str(f), caption=None,
            mime_hint="application/pdf",
        )

        assert result.success is True
        legacy_api.media.return_value.upload.assert_called_once()
        # Cache untouched — the per-user slot stays empty so the next
        # /setup-files for newuser will write into a clean state.
        assert "newuser@example.com" not in adapter._user_chat_api_by_email

    @pytest.mark.asyncio
    async def test_send_file_no_creds_anywhere_posts_setup_notice(
        self, adapter, tmp_path
    ):
        """Sender unknown AND no legacy fallback → setup-instructions
        notice. Same shape as the existing single-user path; the test
        confirms the multi-user routing didn't accidentally bypass it."""
        adapter._last_sender_by_chat["spaces/S"] = "ghost@example.com"
        adapter._user_chat_api = None
        adapter._user_credentials = None
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )

        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF")
        from plugins.platforms.google_chat import oauth as helper
        with patch.object(helper, "load_user_credentials", return_value=None):
            result = await adapter._send_file(
                "spaces/S", str(f), caption=None,
                mime_hint="application/pdf",
            )

        assert result.success is False
        sent = adapter._create_message.call_args.args[1]["text"]
        assert "/setup-files" in sent

    @pytest.mark.asyncio
    async def test_send_file_per_user_401_evicts_only_that_user(
        self, adapter, tmp_path, monkeypatch
    ):
        """A 401 from one user's token must NOT clobber another user's
        cache nor the legacy slot. The eviction is scoped."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._last_sender_by_chat["spaces/S"] = "alice@example.com"

        alice_api = MagicMock()
        alice_api.media.return_value.upload.return_value.execute.side_effect = (
            _FakeHttpError(status=401, reason="Unauthorized")
        )
        bob_api = MagicMock()
        adapter._user_chat_api_by_email["alice@example.com"] = alice_api
        adapter._user_creds_by_email["alice@example.com"] = MagicMock(valid=True)
        adapter._user_chat_api_by_email["bob@example.com"] = bob_api
        adapter._user_creds_by_email["bob@example.com"] = MagicMock(valid=True)
        # Legacy untouched.
        adapter._user_chat_api = MagicMock()
        adapter._user_credentials = MagicMock(valid=True)
        adapter._consume_typing_card_with_text = AsyncMock(return_value=None)
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )

        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF")
        result = await adapter._send_file(
            "spaces/S", str(f), caption=None,
            mime_hint="application/pdf",
        )

        assert result.success is False
        # Alice evicted, Bob and legacy preserved.
        assert "alice@example.com" not in adapter._user_chat_api_by_email
        assert "bob@example.com" in adapter._user_chat_api_by_email
        assert adapter._user_chat_api is not None
        assert adapter._user_credentials is not None

    @pytest.mark.asyncio
    async def test_setup_files_writes_to_per_user_path(
        self, adapter, tmp_path, monkeypatch
    ):
        """``/setup-files <code>`` from sender alice writes to alice's
        token slot; bob's slot stays untouched."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )
        from plugins.platforms.google_chat import oauth as helper
        # Stub the costly bits; we're verifying routing, not OAuth I/O.
        alice_creds = MagicMock(valid=True)
        with patch.object(helper, "exchange_auth_code") as ex, \
             patch.object(helper, "load_user_credentials", return_value=alice_creds), \
             patch.object(helper, "build_user_chat_service",
                          return_value=MagicMock()):
            await adapter._handle_setup_files_command(
                chat_id="spaces/S",
                thread_id=None,
                raw_text="/setup-files PASTED_CODE",
                sender_email="alice@example.com",
            )

        # Helper was invoked with the sender email, so the token lands in
        # the per-user path (not the legacy file).
        assert ex.call_args.args[0] == "PASTED_CODE"
        assert ex.call_args.args[1] == "alice@example.com"
        # Adapter cache populated for alice only.
        assert "alice@example.com" in adapter._user_chat_api_by_email
        assert "bob@example.com" not in adapter._user_chat_api_by_email

    @pytest.mark.asyncio
    async def test_setup_files_revoke_drops_only_that_user(
        self, adapter, tmp_path, monkeypatch
    ):
        """Per-user revoke clears alice's slot; bob and the legacy
        fallback both keep working. Alice's choice to revoke must not
        knock out unrelated users."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter._user_chat_api_by_email["alice@example.com"] = MagicMock()
        adapter._user_creds_by_email["alice@example.com"] = MagicMock()
        adapter._user_chat_api_by_email["bob@example.com"] = MagicMock()
        adapter._user_creds_by_email["bob@example.com"] = MagicMock()
        legacy_api = MagicMock()
        legacy_creds = MagicMock()
        adapter._user_chat_api = legacy_api
        adapter._user_credentials = legacy_creds
        adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m",
                                        "error": None})()
        )

        from plugins.platforms.google_chat import oauth as helper
        with patch.object(helper, "revoke") as rev:
            await adapter._handle_setup_files_command(
                chat_id="spaces/S",
                thread_id=None,
                raw_text="/setup-files revoke",
                sender_email="alice@example.com",
            )

        # Helper called with alice's email
        assert rev.call_args.args[0] == "alice@example.com"
        assert "alice@example.com" not in adapter._user_chat_api_by_email
        assert "bob@example.com" in adapter._user_chat_api_by_email
        # Legacy fallback survives an unrelated user's revoke.
        assert adapter._user_chat_api is legacy_api
        assert adapter._user_credentials is legacy_creds


# ===========================================================================
# Persistent thread-count store (restart-safe side-thread heuristic)
# ===========================================================================


class TestThreadCountStore:
    def test_missing_file_returns_zero_counts(self, tmp_path):
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        store = _ThreadCountStore(tmp_path / "nonexistent.json")
        store.load()
        assert store.get("spaces/X", "spaces/X/threads/T") == 0

    def test_corrupt_json_treated_as_empty(self, tmp_path):
        """A garbage file shouldn't crash the adapter — log warn, treat
        as fresh, move on. The next incr() will overwrite."""
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        path = tmp_path / "counts.json"
        path.write_text("not valid json {")
        store = _ThreadCountStore(path)
        store.load()
        assert store.get("spaces/X", "spaces/X/threads/T") == 0
        # Next write should overwrite cleanly.
        prev = store.incr("spaces/X", "spaces/X/threads/T")
        assert prev == 0
        # File now has valid JSON.
        import json
        data = json.loads(path.read_text())
        assert data == {"spaces/X": {"spaces/X/threads/T": 1}}

    def test_incr_returns_pre_increment_value(self, tmp_path):
        """The PRE-increment count is the heuristic input — it answers
        'have we seen this thread BEFORE this message?'. Off-by-one in
        either direction would break the main-flow vs side-thread call."""
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        store = _ThreadCountStore(tmp_path / "counts.json")
        store.load()
        assert store.incr("spaces/X", "spaces/X/threads/T") == 0
        assert store.incr("spaces/X", "spaces/X/threads/T") == 1
        assert store.incr("spaces/X", "spaces/X/threads/T") == 2
        assert store.get("spaces/X", "spaces/X/threads/T") == 3

    def test_round_trip_persists_across_load(self, tmp_path):
        """Two store instances on the same file behave like a single
        store split across a process boundary. This is the exact
        restart-safety property the store exists to provide."""
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        path = tmp_path / "counts.json"

        store_a = _ThreadCountStore(path)
        store_a.load()
        store_a.incr("spaces/X", "spaces/X/threads/T")
        store_a.incr("spaces/X", "spaces/X/threads/T")
        store_a.incr("spaces/Y", "spaces/Y/threads/U")

        # Simulate gateway restart: fresh store instance, same file.
        store_b = _ThreadCountStore(path)
        store_b.load()
        assert store_b.get("spaces/X", "spaces/X/threads/T") == 2
        assert store_b.get("spaces/Y", "spaces/Y/threads/U") == 1
        # Next incr in store_b returns the persisted prev count.
        assert store_b.incr("spaces/X", "spaces/X/threads/T") == 2

    def test_invalid_shape_dropped_silently(self, tmp_path):
        """If someone hand-edits the file with weird shapes, drop the
        bad entries but keep the valid ones."""
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        import json
        path = tmp_path / "counts.json"
        path.write_text(json.dumps({
            "spaces/OK": {"spaces/OK/threads/T": 3},
            "spaces/BAD_VALUE": "not a dict",
            "spaces/BAD_COUNT": {"spaces/BAD_COUNT/threads/T": "five"},
        }))
        store = _ThreadCountStore(path)
        store.load()
        assert store.get("spaces/OK", "spaces/OK/threads/T") == 3
        assert store.get("spaces/BAD_VALUE", "any") == 0
        assert store.get("spaces/BAD_COUNT", "spaces/BAD_COUNT/threads/T") == 0

    @pytest.mark.asyncio
    async def test_outbound_thread_tracked_for_user_reply_in_bot_thread(self, adapter):
        """The bug Ramón hit on the live mac-mini: when the bot replies
        in a fresh thread (Chat-created for the bot's outbound message),
        a future user 'Reply in thread' on that bot message should be
        recognized as a SIDE THREAD (not main flow). For that, the
        outbound thread must be in the count store BEFORE the user's
        reply arrives.

        Regression pin: counting only inbound left bot-created threads
        invisible. User 'Reply in thread' on the bot's response was
        misclassified as main-flow because prev_count was 0."""
        # Stub _create_message's underlying create call — we want to
        # exercise the real _create_message body so the count-tracking
        # branch actually fires.
        create_call = MagicMock()
        create_call.return_value.execute = MagicMock(
            return_value={
                "name": "spaces/S/messages/BOT_REPLY",
                "thread": {"name": "spaces/S/threads/BOT_THREAD"},
            }
        )
        adapter._chat_api.spaces.return_value.messages.return_value.create = create_call

        # Bot sends a top-level reply (no thread.name in body — main flow).
        await adapter._create_message("spaces/S", {"text": "hola"})

        # Outbound thread must now be in the store with count >= 1.
        assert adapter._thread_count_store.get(
            "spaces/S", "spaces/S/threads/BOT_THREAD"
        ) == 1

        # Now user clicks "Reply in thread" on the bot's message →
        # inbound arrives in spaces/S/threads/BOT_THREAD.
        env = _make_chat_envelope(
            text="follow-up", thread_name="spaces/S/threads/BOT_THREAD"
        )
        msg = env["chat"]["messagePayload"]["message"]
        event = await adapter._build_message_event(msg, env)

        # MUST be classified as side thread (isolated session +
        # outbound stays in the thread).
        assert event.source.thread_id == "spaces/S/threads/BOT_THREAD"
        assert adapter._last_inbound_thread["spaces/S"] == "spaces/S/threads/BOT_THREAD"

    @pytest.mark.asyncio
    async def test_side_thread_detection_survives_restart(self, adapter, tmp_path):
        """End-to-end regression for the bug Ramón hit across 4
        iterations: gateway restart must NOT demote an active side
        thread back to main flow.

        Flow:
          1. User has an existing thread (count >= 1 from prior turn).
          2. Gateway restarts (fresh adapter instance with same store path).
          3. User sends another message in that thread.
          4. Adapter must STILL classify it as side thread (isolated
             session + outbound thread) — otherwise main-flow context
             leaks in.
        """
        # Turn 1: simulate prior engagement of T_existing.
        env1 = _make_chat_envelope(text="first", thread_name="spaces/S/threads/T_existing")
        await adapter._build_message_event(env1["chat"]["messagePayload"]["message"], env1)
        env2 = _make_chat_envelope(text="second", thread_name="spaces/S/threads/T_existing")
        await adapter._build_message_event(env2["chat"]["messagePayload"]["message"], env2)
        # After two turns, this is a known side-thread. The store on disk
        # has count >= 2.
        assert adapter._thread_count_store.get(
            "spaces/S", "spaces/S/threads/T_existing"
        ) == 2

        # Simulate restart: build a fresh adapter pointing at the SAME
        # persistence file the previous one used.
        from plugins.platforms.google_chat.adapter import (
            GoogleChatAdapter, _ThreadCountStore,
        )
        store_path = adapter._thread_count_store._path
        fresh = GoogleChatAdapter(_base_config())
        fresh._chat_api = MagicMock()
        fresh._credentials = MagicMock()
        fresh._new_authed_http = MagicMock(return_value=MagicMock())
        fresh.handle_message = AsyncMock()
        fresh._thread_count_store = _ThreadCountStore(store_path)
        fresh._thread_count_store.load()

        # Turn 3 (post-restart, same thread).
        env3 = _make_chat_envelope(text="third", thread_name="spaces/S/threads/T_existing")
        event3 = await fresh._build_message_event(
            env3["chat"]["messagePayload"]["message"], env3
        )
        # MUST be classified as side thread (isolated session).
        assert event3.source.thread_id == "spaces/S/threads/T_existing"
        # Outbound cache populated for in-thread reply.
        assert fresh._last_inbound_thread["spaces/S"] == "spaces/S/threads/T_existing"


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
        from plugins.platforms.google_chat import adapter as gc_mod
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
# Outbound retry (transient API failure handling)
# ===========================================================================


class TestOutboundRetry:
    """Outbound message creation retries on transient failures.

    Without retry, a single 503/429 from Google's Chat REST API drops the
    user-visible reply. The retry wrapper handles 429/5xx/timeout/connection
    errors with exponential backoff + jitter; permanent errors (auth,
    client errors) bubble up on the first attempt.

    Pattern lifted from PR #14965 by @ArnarValur.
    """

    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self, adapter, monkeypatch):
        """A 503 from messages.create triggers backoff + retry.

        On the second attempt the call succeeds, so the user sees the
        reply with no visible failure. The wrapper's sleep is patched
        out so the test runs instantly.
        """
        from plugins.platforms.google_chat import adapter as gc_mod
        async def _no_sleep(*_a, **_kw):
            return None
        monkeypatch.setattr(gc_mod.asyncio, "sleep", _no_sleep)

        # First attempt 503, second attempt OK.
        execute = MagicMock()
        execute.execute.side_effect = [
            _FakeHttpError(status=503, reason="Service unavailable"),
            {"name": "spaces/S/messages/M", "thread": {"name": "spaces/S/threads/T"}},
        ]
        adapter._chat_api.spaces.return_value.messages.return_value.create.return_value = execute

        result = await adapter._create_message("spaces/S", {"text": "hi"})

        assert result.success is True
        assert result.message_id == "spaces/S/messages/M"
        # Two execute() calls — initial + one retry.
        assert execute.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, adapter, monkeypatch):
        """Three consecutive 503s exhaust the retry budget; the call raises."""
        from plugins.platforms.google_chat import adapter as gc_mod
        async def _no_sleep(*_a, **_kw):
            return None
        monkeypatch.setattr(gc_mod.asyncio, "sleep", _no_sleep)

        execute = MagicMock()
        execute.execute.side_effect = _FakeHttpError(status=503, reason="Down")
        adapter._chat_api.spaces.return_value.messages.return_value.create.return_value = execute

        with pytest.raises(_FakeHttpError):
            await adapter._create_message("spaces/S", {"text": "hi"})
        # _RETRY_MAX_ATTEMPTS = 3 → 3 calls total.
        assert execute.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_400(self, adapter, monkeypatch):
        """A 400 (client error) is permanent — no retry, fails immediately."""
        from plugins.platforms.google_chat import adapter as gc_mod
        async def _no_sleep(*_a, **_kw):
            return None
        monkeypatch.setattr(gc_mod.asyncio, "sleep", _no_sleep)

        execute = MagicMock()
        execute.execute.side_effect = _FakeHttpError(status=400, reason="Bad request")
        adapter._chat_api.spaces.return_value.messages.return_value.create.return_value = execute

        with pytest.raises(_FakeHttpError):
            await adapter._create_message("spaces/S", {"text": "hi"})
        # Only one attempt — 400 is not retryable.
        assert execute.execute.call_count == 1

    def test_is_retryable_error_classifier(self):
        """Spot-check the retryable-error taxonomy."""
        from plugins.platforms.google_chat.adapter import _is_retryable_error

        # Retryable: 429, 5xx, timeout-flavored exceptions
        assert _is_retryable_error(_FakeHttpError(status=429, reason="rate"))
        assert _is_retryable_error(_FakeHttpError(status=500, reason="oops"))
        assert _is_retryable_error(_FakeHttpError(status=502, reason="bad gw"))
        assert _is_retryable_error(_FakeHttpError(status=503, reason="down"))
        assert _is_retryable_error(_FakeHttpError(status=504, reason="gw timeout"))
        assert _is_retryable_error(TimeoutError("connection timed out"))
        assert _is_retryable_error(ConnectionResetError("connection reset"))
        # NOT retryable: client errors, auth, programmer errors
        assert not _is_retryable_error(_FakeHttpError(status=400, reason="bad"))
        assert not _is_retryable_error(_FakeHttpError(status=401, reason="auth"))
        assert not _is_retryable_error(_FakeHttpError(status=403, reason="forbidden"))
        assert not _is_retryable_error(_FakeHttpError(status=404, reason="not found"))
        assert not _is_retryable_error(ValueError("typed wrong thing"))


class TestFormatMessage:
    """Markdown→Chat dialect conversion + invisible Unicode stripping.

    `format_message` runs on EVERY outbound message, so the regex
    behavior is the safety surface. Tests cover happy paths, code-block
    protection, edge cases the LLM emits in practice (URLs with parens,
    unmatched syntax, mixed bold+italic), and the Unicode strip's
    interaction with composite emoji.

    Pattern lifted from PR #14965 by @ArnarValur.
    """

    def test_bold_double_asterisk_to_single(self):
        """**bold** → *bold* (Chat's bold syntax uses single asterisks)."""
        out = GoogleChatAdapter.format_message("hello **world**")
        assert out == "hello *world*"

    def test_bold_italic_combo_to_chat_dialect(self):
        """***x*** → *_x_* (bold-italic compound)."""
        out = GoogleChatAdapter.format_message("***fancy*** word")
        assert out == "*_fancy_* word"

    def test_markdown_link_to_chat_anglebracket(self):
        """[text](url) → <url|text> (Slack-style anglebracket links)."""
        out = GoogleChatAdapter.format_message("see [docs](https://example.com)")
        assert out == "see <https://example.com|docs>"

    def test_header_to_bold_at_line_start_only(self):
        """# Title → *Title* but only at line-start; mid-line `#` untouched."""
        out = GoogleChatAdapter.format_message("# Heading\nbody with # mid-line hash")
        assert out == "*Heading*\nbody with # mid-line hash"

    def test_fenced_code_block_protected(self):
        """**asterisks** inside a fenced code block do NOT convert.

        Without protection, the regex would mangle code samples emitted
        by the LLM (e.g. Python or shell with literal `**` operators).
        """
        src = "before\n```python\nx = 2 ** 10\n```\nafter"
        out = GoogleChatAdapter.format_message(src)
        # Code block content survives verbatim.
        assert "```python\nx = 2 ** 10\n```" in out
        # Surrounding text untouched (no asterisks to convert).
        assert out.startswith("before")
        assert out.endswith("after")

    def test_inline_code_protected(self):
        """`**text**` inside inline backticks does NOT convert."""
        out = GoogleChatAdapter.format_message("see `**literal**` for syntax")
        assert "`**literal**`" in out

    def test_url_with_parens_in_path(self):
        """`[txt](https://x.com/foo(bar))` — pin the documented limitation.

        The regex captures the URL up to the FIRST closing paren, so
        URLs with parens in the path get truncated. This pins the
        behavior so any future regex change is intentional. Real
        Wikipedia / docs URLs with parens (e.g. ``Halting_(disambiguation)``)
        are an edge case; the LLM rarely emits them and operators can
        URL-encode if needed.
        """
        out = GoogleChatAdapter.format_message("[wiki](https://x.com/foo(bar))")
        # URL captured up to first ')'; trailing paren left as text.
        assert "<https://x.com/foo(bar|wiki>" in out

    def test_mixed_bold_italic_orderings(self):
        """**bold** _italic_ in the same line — both surface conversions."""
        # Italic stays as `_italic_` (Chat's italic dialect matches our
        # input form, no transform needed).
        out = GoogleChatAdapter.format_message("**bold** and _italic_ together")
        assert "*bold*" in out
        assert "_italic_" in out

    def test_strips_zwj_and_variation_selector(self):
        """ZWJ (U+200D) + Variation Selector 16 (U+FE0F) get stripped.

        These appear in composite emoji like 👨‍👩‍👧 (family) — Chat's
        restricted font can't render them and shows tofu. Stripping
        means the underlying base emoji renders cleanly even if the
        composite breaks; better than tofu boxes.
        """
        # Family emoji: man + ZWJ + woman + ZWJ + girl.
        src = "hello \U0001f468‍\U0001f469‍\U0001f467 world"
        out = GoogleChatAdapter.format_message(src)
        assert "‍" not in out  # ZWJ gone
        # Base codepoints survive (man, woman, girl).
        assert "\U0001f468" in out
        assert "\U0001f469" in out
        assert "\U0001f467" in out

    def test_strips_bom_and_bidi_marks(self):
        """BOM, LTR/RTL marks stripped — they break Chat's font rendering."""
        src = "﻿ hello ‎ world ‏"
        out = GoogleChatAdapter.format_message(src)
        assert "﻿" not in out
        assert "‎" not in out
        assert "‏" not in out
        assert "hello" in out and "world" in out

    def test_empty_and_none_safe(self):
        """Empty / None pass through without raising.

        The double-space collapser runs on every non-empty input — that's
        intentional cleanup after Unicode stripping. So pure-whitespace
        input collapses to a single space; documented as expected.
        """
        assert GoogleChatAdapter.format_message("") == ""
        assert GoogleChatAdapter.format_message(None) is None
        # Multi-space input collapses to single space (the cleanup step
        # runs unconditionally; cheap correctness over rare preservation).
        assert GoogleChatAdapter.format_message("   ") == " "

    def test_unmatched_asterisks_left_alone(self):
        """A lone `**` with no closing pair is not transformed.

        Defensive: the regex requires a closing `**`. Unmatched syntax
        from a partial LLM stream stays visible as-is rather than
        consuming the rest of the message.
        """
        out = GoogleChatAdapter.format_message("rate is ** TBD")
        assert "**" in out  # not converted


class TestADCFallback:
    """When no SA JSON is configured, fall back to Application Default Credentials.

    Critical for Cloud Run / GCE / GKE deploys where workload identity
    means key files are unnecessary and a security risk to manage.
    Pattern lifted from PR #14965.
    """

    def test_load_credentials_uses_adc_when_no_sa_path(self, adapter, monkeypatch):
        """No SA path → google.auth.default() is called."""
        adapter.config.extra.pop("service_account_json", None)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", raising=False)

        adc_creds = MagicMock(name="adc_credentials")
        fake_default = MagicMock(return_value=(adc_creds, "fake-project"))
        # ``google`` is mocked at module load via _ensure_google_mocks; patch
        # the attribute path the adapter uses (``google.auth.default``).
        google_pkg = sys.modules.get("google") or types.SimpleNamespace()
        fake_auth_module = types.SimpleNamespace(default=fake_default)
        monkeypatch.setattr(google_pkg, "auth", fake_auth_module, raising=False)
        monkeypatch.setitem(sys.modules, "google", google_pkg)
        monkeypatch.setitem(sys.modules, "google.auth", fake_auth_module)

        result = adapter._load_sa_credentials()

        assert result is adc_creds
        fake_default.assert_called_once()

    def test_load_credentials_raises_when_no_sa_and_adc_unavailable(
        self, adapter, monkeypatch
    ):
        """ADC failure surfaces a useful error pointing at the two fixes."""
        adapter.config.extra.pop("service_account_json", None)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", raising=False)

        def _boom(*_a, **_kw):
            raise Exception("no credentials")
        google_pkg = sys.modules.get("google") or types.SimpleNamespace()
        fake_auth_module = types.SimpleNamespace(default=_boom)
        monkeypatch.setattr(google_pkg, "auth", fake_auth_module, raising=False)
        monkeypatch.setitem(sys.modules, "google", google_pkg)
        monkeypatch.setitem(sys.modules, "google.auth", fake_auth_module)

        with pytest.raises(ValueError) as ei:
            adapter._load_sa_credentials()
        msg = str(ei.value).lower()
        assert "default credentials" in msg or "adc" in msg
        assert "google_chat_service_account_json" in msg


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
            "plugins.platforms.google_chat.adapter.asyncio.sleep", _instant
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
    """`GOOGLE_CHAT_ALLOWED_USERS=email` matches naturally without a bridge.

    Post-#14965 absorption: the adapter sets ``source.user_id =
    sender_email`` directly, so the generic allowlist match in
    ``_is_user_authorized`` finds it without any platform-specific
    code path. Pinning here so the bridge can never silently come
    back without a test failing.
    """

    def test_allowlist_matches_when_user_id_is_email(self, monkeypatch):
        """Email allowlist match — the canonical case.

        The adapter assigns ``user_id = sender_email`` so the generic
        check_ids path picks it up. No platform-specific bridge needed.
        """
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
            user_id="alice@example.com",       # post-swap: email is canonical
            user_name="Alice",
            user_id_alt="users/12345",         # resource name moves to alt
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
            user_id="bob@example.com",
            user_name="Bob",
            user_id_alt="users/99999",
        )
        assert runner._is_user_authorized(source) is False

    def test_allowlist_falls_back_to_resource_name_when_no_email(
        self, monkeypatch
    ):
        """If sender has no email, ``user_id`` falls back to the resource
        name. Operators who allowlist by ``users/{id}`` still match.
        """
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "users/77777")
        cfg = GatewayConfig()
        runner = GatewayRunner(cfg)
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved = MagicMock(return_value=False)

        source = SessionSource(
            platform=Platform.GOOGLE_CHAT,
            chat_id="spaces/S",
            chat_type="dm",
            user_id="users/77777",  # no email available — resource name wins
            user_name="System",
            user_id_alt=None,
        )
        assert runner._is_user_authorized(source) is True


# ===========================================================================
# Cron scheduler registry (regression guard from /review)
#
# After the generic-plugin-interface migration, Google Chat no longer lives in
# the hardcoded ``_KNOWN_DELIVERY_PLATFORMS`` / ``_HOME_TARGET_ENV_VARS`` sets
# in ``cron/scheduler.py``.  It earns cron delivery via
# ``PlatformEntry.cron_deliver_env_var``, which the scheduler consults through
# ``_is_known_delivery_platform`` and ``_resolve_home_env_var``.  The tests
# below check that public resolver behavior, not the hardcoded sets.
# ===========================================================================


class TestCronSchedulerRegistry:
    def _ensure_registered(self):
        """Force the plugin system to register the Google Chat adapter.

        The adapter's ``register(ctx)`` is only invoked during plugin
        discovery; module-level import alone does not register it.  We call
        discover + manually invoke the register hook so the resolver sees
        ``cron_deliver_env_var``.
        """
        from gateway.platform_registry import platform_registry
        if platform_registry.get("google_chat") is not None:
            return
        # Discover first so the plugin is loaded at all.
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
        except Exception:
            pass
        if platform_registry.get("google_chat") is not None:
            return
        # Fallback: construct a minimal ctx and call register directly.
        from plugins.platforms.google_chat.adapter import register as _register
        class _Ctx:
            class _M:
                name = "google_chat-platform"
            manifest = _M()
            _manager = type("_Mgr", (), {"_plugin_platform_names": set()})()
            def register_platform(self, **kwargs):
                from gateway.platform_registry import PlatformEntry
                entry = PlatformEntry(source="plugin", **kwargs)
                platform_registry.register(entry)
        _register(_Ctx())

    def test_google_chat_is_known_delivery_platform(self):
        self._ensure_registered()
        from cron.scheduler import _is_known_delivery_platform

        assert _is_known_delivery_platform("google_chat") is True

    def test_google_chat_home_env_var_resolves(self):
        self._ensure_registered()
        from cron.scheduler import _resolve_home_env_var

        assert _resolve_home_env_var("google_chat") == "GOOGLE_CHAT_HOME_CHANNEL"
