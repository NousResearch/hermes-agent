"""Tests for disabled-by-default A2A interop primitives."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_interop.a2a import (
    A2AAuthVerifier,
    A2APeerPolicyStore,
    A2ATranscriptStore,
    A2AVerificationError,
    DEFAULT_A2A_POLICY,
    load_a2a_config,
)


def test_a2a_config_is_disabled_by_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_a2a_config()

    assert config.enabled is False
    assert config.peers == {}
    assert config.transcript_dir == tmp_path / "a2a" / "transcripts"


def test_a2a_relative_transcript_dir_resolves_under_hermes_home(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a2a:\n  transcript_dir: relative/transcripts\n", encoding="utf-8")

    config = load_a2a_config(config_path=config_path, hermes_home=tmp_path)

    assert config.transcript_dir == tmp_path / "relative" / "transcripts"


def test_peer_policy_rejects_unsafe_modes_and_preserves_mandatory_blocks(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
a2a:
  enabled: true
  peers:
    peer:
      auth:
        bearer_token_env: PEER_TOKEN
        hmac_secret_env: PEER_HMAC
      dangerous_command_mode: allow_everything
      blocked_roots: []
      max_tokens_per_day: -1
""".strip(),
        encoding="utf-8",
    )
    store = A2APeerPolicyStore(config_path=config_path, hermes_home=tmp_path)

    with pytest.raises(ValueError, match="dangerous_command_mode"):
        store.get_peer("peer")


def test_redaction_covers_private_keys_cookies_and_key_value_secrets(tmp_path):
    store = A2ATranscriptStore(root=tmp_path / "a2a" / "transcripts")
    body = "password=hunter2 Cookie: sessionid=abc123 -----BEGIN PRIVATE KEY----- abc -----END PRIVATE KEY-----"

    store.append_message(peer_id="peer", conversation_id="c", direction="inbound", body=body)
    message = store.get_conversation("peer", "c")[0]

    assert "hunter2" not in message["body"]
    assert "sessionid=abc123" not in message["body"]
    assert "PRIVATE KEY" not in message["body"]
    assert message["redaction_count"] >= 3


def test_auth_verifier_rejects_oversized_nonce(monkeypatch):
    monkeypatch.setenv("OPENCLAW_A2A_TOKEN", "test-token")
    monkeypatch.setenv("OPENCLAW_A2A_HMAC", "test-secret")
    now = int(time.time())
    body = b"{}"
    nonce = "n" * 300
    verifier = A2AAuthVerifier(nonce_store=set(), now=lambda: now)
    signature = verifier.sign_body(body, "test-secret", timestamp=now, nonce=nonce)

    with pytest.raises(A2AVerificationError, match="nonce"):
        verifier.verify(
            body=body,
            headers={
                "Authorization": "Bearer test-token",
                "X-Hermes-A2A-Timestamp": str(now),
                "X-Hermes-A2A-Nonce": nonce,
                "X-Hermes-A2A-Signature": signature,
            },
            bearer_token_env="OPENCLAW_A2A_TOKEN",
            hmac_secret_env="OPENCLAW_A2A_HMAC",
        )


def test_a2a_config_rejects_quoted_boolean_enabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('a2a:\n  enabled: "false"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="enabled"):
        load_a2a_config(config_path=config_path, hermes_home=tmp_path)


def test_auth_verifier_rejects_dot_nonce_to_avoid_signature_boundary_ambiguity(monkeypatch):
    monkeypatch.setenv("OPENCLAW_A2A_TOKEN", "test-token")
    monkeypatch.setenv("OPENCLAW_A2A_HMAC", "test-secret")
    now = int(time.time())
    body = b"{}"
    nonce = "nonce.with.dot"
    verifier = A2AAuthVerifier(nonce_store=set(), now=lambda: now)
    signature = verifier.sign_body(body, "test-secret", timestamp=now, nonce=nonce)

    with pytest.raises(A2AVerificationError, match="nonce"):
        verifier.verify(
            body=body,
            headers={
                "Authorization": "Bearer test-token",
                "X-Hermes-A2A-Timestamp": str(now),
                "X-Hermes-A2A-Nonce": nonce,
                "X-Hermes-A2A-Signature": signature,
            },
            bearer_token_env="OPENCLAW_A2A_TOKEN",
            hmac_secret_env="OPENCLAW_A2A_HMAC",
        )


def test_peer_policy_loader_requires_enabled_flag_and_applies_readonly_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
a2a:
  enabled: true
  peers:
    openclaw-local:
      display_name: OpenClaw Local
      owner: William
      auth:
        bearer_token_env: OPENCLAW_A2A_TOKEN
        hmac_secret_env: OPENCLAW_A2A_HMAC
      allowed_ingress_routes: [webhook]
      allowed_outbound_routes: [local]
""".strip(),
        encoding="utf-8",
    )

    store = A2APeerPolicyStore(config_path=config_path, hermes_home=tmp_path)
    policy = store.get_peer("openclaw-local")

    assert policy.peer_id == "openclaw-local"
    assert policy.enabled is True
    assert policy.read_roots == DEFAULT_A2A_POLICY.read_roots == []
    assert policy.write_roots == []
    assert policy.allowed_toolsets == []
    assert policy.external_send_allowed is False
    assert policy.dangerous_command_mode == "deny"
    assert "~/.credentials" in policy.blocked_roots
    assert policy.max_tool_calls_per_message == 5


def test_peer_policy_loader_rejects_enabled_peer_without_auth(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
a2a:
  enabled: true
  peers:
    unsafe-peer:
      display_name: Unsafe
      owner: Nobody
      auth: {}
""".strip(),
        encoding="utf-8",
    )

    store = A2APeerPolicyStore(config_path=config_path, hermes_home=tmp_path)

    with pytest.raises(ValueError, match="auth"):
        store.get_peer("unsafe-peer")


def test_peer_policy_loader_can_validate_all_configured_peers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
a2a:
  enabled: true
  peers:
    safe-peer:
      display_name: Safe
      owner: William
      auth:
        bearer_token_env: SAFE_A2A_TOKEN
        hmac_secret_env: SAFE_A2A_HMAC
    unsafe-peer:
      display_name: Unsafe
      owner: Nobody
      auth: {}
""".strip(),
        encoding="utf-8",
    )

    store = A2APeerPolicyStore(config_path=config_path, hermes_home=tmp_path)

    with pytest.raises(ValueError, match="unsafe-peer"):
        store.list_peers(validate=True)


def test_peer_policy_rejects_quoted_external_send_boolean(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
a2a:
  enabled: true
  peers:
    peer:
      auth:
        bearer_token_env: PEER_TOKEN
        hmac_secret_env: PEER_HMAC
      external_send_allowed: "false"
""".strip(),
        encoding="utf-8",
    )
    store = A2APeerPolicyStore(config_path=config_path, hermes_home=tmp_path)

    with pytest.raises(ValueError, match="external_send_allowed"):
        store.get_peer("peer")


def test_transcript_store_is_separate_from_state_db_and_redacts_credentials(tmp_path):
    store = A2ATranscriptStore(root=tmp_path / "a2a" / "transcripts")

    event = store.append_message(
        peer_id="openclaw-local",
        conversation_id="thread-1",
        direction="inbound",
        body="Deploy with token sk-live-secret and file ~/.credentials/hermes-request/vercel.env",
        metadata={"route": "webhook"},
    )
    messages = store.get_conversation("openclaw-local", "thread-1")

    assert store.root == tmp_path / "a2a" / "transcripts"
    assert "state.db" not in str(event.path)
    assert len(messages) == 1
    assert messages[0]["body"] == "Deploy with token <redacted> and file <credential-path-redacted>"
    assert messages[0]["redaction_count"] == 2
    assert messages[0]["metadata"]["route"] == "webhook"


def test_transcript_store_redacts_nested_metadata_credentials(tmp_path):
    store = A2ATranscriptStore(root=tmp_path / "a2a" / "transcripts")

    store.append_message(
        peer_id="openclaw-local",
        conversation_id="thread-1",
        direction="outbound",
        body="ok",
        metadata={
            "token": "Bearer secret-token-value",
            "paths": ["~/.credentials/hermes-request/vercel.env"],
            "Bearer metadata-key-secret": "~/.credentials/hermes-request/vercel.env",
            "nested": {"safe": "value"},
        },
    )

    message = store.get_conversation("openclaw-local", "thread-1")[0]
    assert message["metadata"]["token"] == "<redacted>"
    assert message["metadata"]["paths"] == ["<credential-path-redacted>"]
    assert message["metadata"]["<redacted>"] == "<credential-path-redacted>"
    assert message["metadata"]["nested"] == {"safe": "value"}
    assert message["redaction_count"] == 4


def test_auth_verifier_accepts_valid_bearer_hmac_timestamp_and_nonce(monkeypatch):
    monkeypatch.setenv("OPENCLAW_A2A_TOKEN", "test-token")
    monkeypatch.setenv("OPENCLAW_A2A_HMAC", "test-secret")
    now = int(time.time())
    body = b'{"message":"hello"}'
    verifier = A2AAuthVerifier(nonce_store=set(), now=lambda: now)
    signature = verifier.sign_body(body, "test-secret", timestamp=now, nonce="nonce-1")

    result = verifier.verify(
        body=body,
        headers={
            "Authorization": "Bearer test-token",
            "X-Hermes-A2A-Timestamp": str(now),
            "X-Hermes-A2A-Nonce": "nonce-1",
            "X-Hermes-A2A-Signature": signature,
        },
        bearer_token_env="OPENCLAW_A2A_TOKEN",
        hmac_secret_env="OPENCLAW_A2A_HMAC",
    )

    assert result.ok is True
    assert result.nonce == "nonce-1"


def test_auth_verifier_rejects_replay_and_bad_signature(monkeypatch):
    monkeypatch.setenv("OPENCLAW_A2A_TOKEN", "test-token")
    monkeypatch.setenv("OPENCLAW_A2A_HMAC", "test-secret")
    now = int(time.time())
    body = b'{"message":"hello"}'
    verifier = A2AAuthVerifier(nonce_store=set(), now=lambda: now)
    signature = verifier.sign_body(body, "test-secret", timestamp=now, nonce="nonce-1")
    headers = {
        "Authorization": "Bearer test-token",
        "X-Hermes-A2A-Timestamp": str(now),
        "X-Hermes-A2A-Nonce": "nonce-1",
        "X-Hermes-A2A-Signature": signature,
    }

    verifier.verify(
        body=body,
        headers=headers,
        bearer_token_env="OPENCLAW_A2A_TOKEN",
        hmac_secret_env="OPENCLAW_A2A_HMAC",
    )
    with pytest.raises(A2AVerificationError, match="replay"):
        verifier.verify(
            body=body,
            headers=headers,
            bearer_token_env="OPENCLAW_A2A_TOKEN",
            hmac_secret_env="OPENCLAW_A2A_HMAC",
        )

    with pytest.raises(A2AVerificationError, match="signature"):
        A2AAuthVerifier(nonce_store=set(), now=lambda: now).verify(
            body=body,
            headers={**headers, "X-Hermes-A2A-Nonce": "nonce-2", "X-Hermes-A2A-Signature": "bad"},
            bearer_token_env="OPENCLAW_A2A_TOKEN",
            hmac_secret_env="OPENCLAW_A2A_HMAC",
        )


def test_auth_verifier_rejects_stale_timestamp(monkeypatch):
    monkeypatch.setenv("OPENCLAW_A2A_TOKEN", "test-token")
    monkeypatch.setenv("OPENCLAW_A2A_HMAC", "test-secret")
    now = 1_700_000_000
    body = b"{}"
    verifier = A2AAuthVerifier(nonce_store=set(), now=lambda: now)
    stale = now - 999
    signature = verifier.sign_body(body, "test-secret", timestamp=stale, nonce="nonce-stale")

    with pytest.raises(A2AVerificationError, match="timestamp"):
        verifier.verify(
            body=body,
            headers={
                "Authorization": "Bearer test-token",
                "X-Hermes-A2A-Timestamp": str(stale),
                "X-Hermes-A2A-Nonce": "nonce-stale",
                "X-Hermes-A2A-Signature": signature,
            },
            bearer_token_env="OPENCLAW_A2A_TOKEN",
            hmac_secret_env="OPENCLAW_A2A_HMAC",
        )
