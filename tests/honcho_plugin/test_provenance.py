"""Behavioral contract for first-class Honcho message provenance (BC-61)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    DeliveryState,
    HonchoSession,
    HonchoSessionManager,
)


AGADOR_DEFAULTS = {
    "schema": "agador.provenance/v1",
    "authority": "user",
    "evidence_refs": [],
    "effective_at": None,
    "confidence": None,
    "verification_state": "unverified",
    "review_state": "captured",
    "sensitivity": "private",
    "retention_class": "semantic",
    "derived_from": [],
    "supersedes": [],
    "superseded_by": [],
    "deletion_request_id": None,
    "deleted_at": None,
    "target_artifacts": [],
    "extensions": {"agador": {"policy": "synthetic"}},
}

INIT_CONTEXT = {
    "hermes_home": "/profiles/work",
    "platform": "telegram",
    "agent_context": "primary",
    "agent_identity": "work",
    "agent_workspace": "hermes",
    "user_id": "platform-user",
    "user_id_alt": "stable-user",
    "user_name": "Synthetic User",
    "chat_id": "chat-42",
    "chat_name": "Synthetic Room",
    "chat_type": "group",
    "thread_id": "thread-9",
    "gateway_session_key": "agent:work:telegram:group:chat-42:thread-9",
    "session_title": "Synthetic provenance",
}


@pytest.fixture(autouse=True)
def _stable_machine_identity(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.honcho.session._resolve_machine_identity",
        lambda _context: "runtime-node-7",
    )


def _manager(*, static=None, **context):
    cfg = HonchoClientConfig(
        ai_peer="hermes-coding",
        peer_name="stable-human",
        message_metadata=static if static is not None else AGADOR_DEFAULTS,
        write_frequency="turn",
    )
    return HonchoSessionManager(
        honcho=MagicMock(),
        config=cfg,
        runtime_user_peer_name="stable-human",
        provenance_context={**INIT_CONTEXT, **context},
        source_session_id="source-session-123",
    )


def _cached_session():
    return HonchoSession(
        key="resolved-session",
        user_peer_id="stable-human",
        assistant_peer_id="hermes-coding",
        honcho_session_id="resolved-session",
    )


def _wire_flush(mgr, session, *, fail_first=False):
    user_peer = MagicMock()
    assistant_peer = MagicMock()
    created = []

    def make(role):
        def factory(content, *, metadata, created_at):
            msg = SimpleNamespace(content=content, metadata=metadata, created_at=created_at, role=role)
            created.append(msg)
            return msg
        return factory

    user_peer.message.side_effect = make("user")
    assistant_peer.message.side_effect = make("assistant")
    mgr._get_or_create_peer = MagicMock(side_effect=lambda peer: user_peer if peer == "stable-human" else assistant_peer)
    sdk_session = MagicMock()
    if fail_first:
        sdk_session.add_messages.side_effect = [RuntimeError("synthetic outage"), None]
    mgr._sessions_cache[session.honcho_session_id] = sdk_session
    return created, sdk_session


def test_exact_normative_metadata_and_sdk_created_at_for_both_roles():
    mgr = _manager()
    session = _cached_session()
    mgr.add_source_message(session, "user", "hello")
    mgr.add_source_message(session, "assistant", "hi")
    created, sdk_session = _wire_flush(mgr, session)

    assert mgr._flush_session(session).state is DeliveryState.DELIVERED
    assert len(created) == 2
    user, assistant = created
    for item in created:
        md = item.metadata
        assert md["schema"] == "agador.provenance/v1"
        assert md["record_kind"] == "source_event"
        assert md["human_peer"] == "stable-human"
        assert md["ai_peer"] == "hermes-coding"
        assert md["interface"] == "telegram"
        assert md["machine"] == "runtime-node-7"
        assert md["runtime"] == "hermes:work:primary"
        assert md["session_id"] == "source-session-123"
        assert md["channel_id"] == "chat-42"
        assert md["source_record_id"]
        assert md["observed_at"].endswith("Z")
        assert md["confidence"] is None
        assert md["review_state"] == "captured"
        assert md["retention_class"] == "semantic"
        assert md["deletion_request_id"] is None
        assert md["extensions"]["agador"] == {"policy": "synthetic"}
        assert md["extensions"]["hermes"]["thread_id"] == "thread-9"
        assert md["extensions"]["hermes"]["agent_workspace"] == "hermes"
        assert "hermes_home" not in md["extensions"]["hermes"]
        assert md["extensions"]["hermes"]["chat_type"] == "group"
        assert md["extensions"]["hermes"]["gateway_session_key"] == INIT_CONTEXT["gateway_session_key"]
        assert item.created_at == datetime.fromisoformat(md["observed_at"].replace("Z", "+00:00"))
        assert item.created_at.tzinfo == timezone.utc
    assert user.metadata["source_kind"] == "user_statement"
    assert assistant.metadata["source_kind"] == "assistant_statement"
    assert user.metadata["authority"] == "stable-human"
    assert assistant.metadata["authority"] == "hermes-coding"
    assert user.metadata["event_id"] != assistant.metadata["event_id"]
    sdk_session.add_messages.assert_called_once()


def test_failed_flush_retry_preserves_event_ids_metadata_and_created_at():
    mgr = _manager()
    session = _cached_session()
    mgr.add_source_message(session, "user", "retry me")
    created, _ = _wire_flush(mgr, session, fail_first=True)

    assert mgr._flush_session(session).state is DeliveryState.FAILED
    assert mgr._flush_session(session).state is DeliveryState.DELIVERED
    assert len(created) == 2
    assert created[0].metadata == created[1].metadata
    assert created[0].created_at == created[1].created_at


def test_chunked_role_side_has_one_source_record_and_unique_stable_events():
    mgr = _manager()
    session = _cached_session()
    source_record_id = mgr.new_source_record_id()
    mgr.add_source_message(session, "user", "part 1", source_record_id=source_record_id, chunk_index=0, chunk_count=2)
    mgr.add_source_message(session, "user", "part 2", source_record_id=source_record_id, chunk_index=1, chunk_count=2)

    first, second = (m["metadata"] for m in session.messages)
    assert first["source_record_id"] == second["source_record_id"] == source_record_id
    assert first["event_id"] != second["event_id"]
    assert [first["extensions"]["hermes"]["chunk_index"], second["extensions"]["hermes"]["chunk_index"]] == [0, 1]
    assert first["extensions"]["hermes"]["chunk_count"] == second["extensions"]["hermes"]["chunk_count"] == 2


def test_static_defaults_merge_but_dynamic_fields_are_protected():
    static = {
        **AGADOR_DEFAULTS,
        "event_id": "attacker-event",
        "record_kind": "promoted_artifact",
        "source_kind": "imported_record",
        "interface": "discord",
        "machine": "wrong-machine",
        "runtime": "wrong-runtime",
        "session_id": "wrong-session",
        "channel_id": "wrong-channel",
        "source_record_id": "wrong-record",
        "observed_at": "2000-01-01T00:00:00Z",
        "human_peer": "wrong-human",
        "ai_peer": "wrong-ai",
        "extensions": {"custom": {"kept": True}, "hermes": {"thread_id": "wrong-thread", "extra": "kept"}},
    }
    mgr = _manager(static=static)
    session = _cached_session()
    mgr.add_source_message(session, "assistant", "safe")
    md = session.messages[0]["metadata"]

    assert md["event_id"] != "attacker-event"
    assert md["record_kind"] == "source_event"
    assert md["source_kind"] == "assistant_statement"
    assert (md["interface"], md["machine"], md["runtime"], md["session_id"]) == (
        "telegram", "runtime-node-7", "hermes:work:primary", "source-session-123"
    )
    assert (md["human_peer"], md["ai_peer"]) == ("stable-human", "hermes-coding")
    assert md["channel_id"] == "chat-42"
    assert md["source_record_id"] != "wrong-record"
    assert md["observed_at"] != "2000-01-01T00:00:00Z"
    assert md["extensions"]["custom"] == {"kept": True}
    assert md["extensions"]["hermes"]["extra"] == "kept"
    assert md["extensions"]["hermes"]["thread_id"] == "thread-9"


def test_default_config_emits_coherent_provider_neutral_envelope():
    mgr = _manager(static={})
    session = _cached_session()
    mgr.add_source_message(session, "user", "default metadata")

    md = session.messages[0]["metadata"]
    assert md["schema"] == "hermes.provenance/v1"
    assert md["record_kind"] == "source_event"
    assert md["source_kind"] == "user_statement"
    assert md["authority"] == "stable-human"
    assert md["verification_state"] == "unverified"
    assert md["review_state"] == "captured"
    assert md["sensitivity"] == "private"
    assert md["retention_class"] == "semantic"
    assert md["derived_from"] == []
    assert md["extensions"]["hermes"]["chunk_count"] == 1


@pytest.mark.parametrize("strategy", ["per-session", "per-directory", "per-repo", "global"])
def test_original_source_session_is_preserved_for_every_session_strategy(strategy):
    cfg = HonchoClientConfig(session_strategy=strategy, message_metadata=AGADOR_DEFAULTS)
    provider = HonchoMemoryProvider()
    provider._config = cfg
    provider._session_key = f"resolved-{strategy}"
    manager = _manager()
    manager._write_frequency = "session"
    manager._cache[provider._session_key] = _cached_session()
    provider._manager = manager
    provider._cron_skipped = False
    provider._session_initialized = True
    provider.sync_turn("user", "assistant", session_id="source-session-123")
    provider._sync_thread.join(timeout=1)

    session = manager._cache.get(provider._session_key)
    # Avoid remote setup in this behavioral test: if not cached, the implementation
    # has not yet provided a testable local path and the assertion is intentionally red.
    assert session is not None
    assert {m["metadata"]["session_id"] for m in session.messages} == {"source-session-123"}


def test_current_sync_session_overrides_initialized_source_session_after_switch():
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(message_metadata=AGADOR_DEFAULTS)
    provider._session_key = "resolved-session"
    manager = _manager()
    manager._write_frequency = "session"
    manager._cache[provider._session_key] = _cached_session()
    provider._manager = manager
    provider._cron_skipped = False
    provider._session_initialized = True

    provider.sync_turn("after switch", "current response", session_id="source-session-B")
    assert provider._sync_thread is not None
    provider._sync_thread.join(timeout=1)

    assert {
        message["metadata"]["session_id"]
        for message in manager._cache[provider._session_key].messages
    } == {"source-session-B"}


def test_runtime_aliases_preserve_one_human_peer_across_interfaces():
    cfg = HonchoClientConfig(
        ai_peer="hermes-coding",
        peer_name="stable-human",
        user_peer_aliases={
            "telegram-42": "stable-human",
            "discord-99": "stable-human",
        },
        message_metadata=AGADOR_DEFAULTS,
        write_frequency="session",
    )
    peers = []
    for interface, runtime_id in (("telegram", "telegram-42"), ("discord", "discord-99")):
        manager = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name=runtime_id,
            provenance_context={**INIT_CONTEXT, "platform": interface},
            source_session_id=f"{interface}-session",
        )
        manager._get_or_create_peer = MagicMock(return_value=MagicMock())
        manager._get_or_create_honcho_session = MagicMock(return_value=(MagicMock(), []))
        session = manager.get_or_create(f"{interface}:chat")
        manager.add_source_message(session, "user", "same human")
        peers.append(session.messages[0]["metadata"]["human_peer"])

    assert peers == ["stable-human", "stable-human"]


def test_metadata_secrets_are_force_redacted_before_sdk_boundary(monkeypatch):
    secret = "sk-" + "proj-" + ("abc123XYZ" * 5)
    secret_key = "ghp_" + ("9Az" * 16)
    static = {
        **AGADOR_DEFAULTS,
        "authority": f"api_key={secret}",
        "extensions": {"custom": {"token": secret, secret_key: "key-carried-secret"}},
    }
    mgr = _manager(static=static, chat_name=f"token={secret}")
    session = _cached_session()
    mgr.add_source_message(session, "user", "conversation content is not changed")
    created, _ = _wire_flush(mgr, session)

    assert mgr._flush_session(session).state is DeliveryState.DELIVERED
    serialized = repr(created[0].metadata)
    assert secret not in serialized
    assert secret_key not in serialized
    assert "key-carried-secret" in serialized
    assert "redacted" in serialized
    assert created[0].content == "conversation content is not changed"


@pytest.mark.parametrize(
    ("metadata", "plain_secret"),
    [
        ({"password": "hunter2"}, "hunter2"),
        ({"api_key": "ordinary-secret-value"}, "ordinary-secret-value"),
        ({"token": "short-secret"}, "short-secret"),
        ({"nested": {"client_secret": "abc123"}}, "abc123"),
        ({"clientSecret": "camel-client-secret"}, "camel-client-secret"),
        ({"accessToken": "camel-access-token"}, "camel-access-token"),
        ({"refreshToken": "camel-refresh-token"}, "camel-refresh-token"),
        ({"authToken": "camel-auth-token"}, "camel-auth-token"),
        ({"authorizationHeader": "Basic ordinary-value"}, "Basic ordinary-value"),
        ({"passwordValue": "camel-password"}, "camel-password"),
        ({"APIKeyValue": "acronym-api-key"}, "acronym-api-key"),
        ({"privateAPIKey": "nested-acronym-key"}, "nested-acronym-key"),
        ({"PRIVATEKEY": "uppercase-private-key"}, "uppercase-private-key"),
    ],
)
def test_credential_named_metadata_redacts_ordinary_values(metadata, plain_secret):
    mgr = _manager(static={**AGADOR_DEFAULTS, "extensions": metadata})
    session = _cached_session()
    mgr.add_source_message(session, "user", "safe content")

    serialized = repr(session.messages[0]["metadata"])
    assert plain_secret not in serialized
    assert "redacted-metadata" in serialized


def test_credential_reference_metadata_remains_usable_without_secret_values():
    mgr = _manager(
        static={
            **AGADOR_DEFAULTS,
            "extensions": {
                "apiKeyPath": "infisical://project/path",
                "secretReference": "infisical:n8n/prod/homeassistant",
                "tokenCount": 12,
            },
        }
    )
    session = _cached_session()
    mgr.add_source_message(session, "user", "safe content")

    extensions = session.messages[0]["metadata"]["extensions"]
    assert extensions["apiKeyPath"] == "infisical://project/path"
    assert extensions["secretReference"] == "infisical:n8n/prod/homeassistant"
    assert extensions["tokenCount"] == 12


def test_non_json_metadata_values_fail_closed_before_sdk_serialization():
    mgr = _manager(static={**AGADOR_DEFAULTS, "extensions": {"custom": object()}})
    session = _cached_session()
    mgr.add_source_message(session, "user", "safe content")

    assert session.messages[0]["metadata"]["extensions"]["custom"] == "«redacted-metadata»"


def test_loaded_synced_messages_reconstruct_sdk_metadata():
    mgr = _manager()
    existing = SimpleNamespace(
        peer_id="stable-human",
        content="loaded",
        created_at=datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
        metadata={"schema": "agador.provenance/v1", "event_id": "existing-event"},
    )
    mgr._get_or_create_peer = MagicMock(return_value=MagicMock())
    mgr._get_or_create_honcho_session = MagicMock(return_value=(MagicMock(), [existing]))

    loaded = mgr.get_or_create("loaded-session")
    assert loaded.messages[0]["metadata"] == existing.metadata
    assert loaded.messages[0]["created_at"] == existing.created_at
    assert loaded.messages[0]["_synced"] is True
