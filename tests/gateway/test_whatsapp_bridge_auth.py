"""Security regressions for authenticated WhatsApp bridge IPC."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_bridge_token_is_persistent_on_disk(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import _load_or_create_bridge_token

    first = _load_or_create_bridge_token(tmp_path)
    second = _load_or_create_bridge_token(tmp_path)

    assert first == second
    assert len(first) >= 32
    assert (tmp_path / ".bridge-token").read_text(encoding="utf-8").strip() == first


def test_concurrent_bridge_token_creation_reuses_exclusive_winner(
    tmp_path: Path,
):
    from plugins.platforms.whatsapp import adapter

    with ThreadPoolExecutor(max_workers=8) as executor:
        tokens = list(
            executor.map(adapter._load_or_create_bridge_token, [tmp_path] * 8)
        )

    persisted = (tmp_path / ".bridge-token").read_text(encoding="utf-8").strip()
    assert tokens == [persisted] * len(tokens)


def test_malformed_persisted_bridge_token_is_replaced(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import _load_or_create_bridge_token

    token_path = tmp_path / ".bridge-token"
    token_path.write_text("truncated", encoding="utf-8")

    token = _load_or_create_bridge_token(tmp_path)

    assert len(token) >= 32
    assert token_path.read_text(encoding="utf-8") == token


def test_health_authentication_requires_challenge_bound_proof():
    from plugins.platforms.whatsapp.adapter import (
        _bridge_auth_proof,
        _health_response_is_authenticated,
    )

    token = "test-token-with-enough-entropy"
    challenge = "fresh-challenge"
    valid = {
        "status": "connected",
        "scriptHash": "public-script-hash",
        "authProof": _bridge_auth_proof(token, challenge),
    }

    assert _health_response_is_authenticated(valid, token, challenge)
    assert not _health_response_is_authenticated(
        {**valid, "authProof": _bridge_auth_proof(token, "replayed-challenge")},
        token,
        challenge,
    )
    assert not _health_response_is_authenticated(
        {"status": "connected", "scriptHash": "public-script-hash"},
        token,
        challenge,
    )


def test_bridge_headers_include_bearer_token_and_optional_challenge():
    from plugins.platforms.whatsapp.adapter import _bridge_auth_headers

    assert _bridge_auth_headers("secret") == {"Authorization": "Bearer secret"}
    assert _bridge_auth_headers("secret", "nonce") == {
        "Authorization": "Bearer secret",
        "X-Hermes-Bridge-Challenge": "nonce",
    }


def test_bridge_source_hash_changes_when_auth_helper_changes(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import _bridge_source_hash

    bridge = tmp_path / "bridge.js"
    auth = tmp_path / "bridge_auth.js"
    bridge.write_text("// bridge\n")
    auth.write_text("// auth v1\n")
    first = _bridge_source_hash(bridge)

    auth.write_text("// auth v2\n")

    assert _bridge_source_hash(bridge) != first
