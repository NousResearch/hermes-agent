"""Security regressions for authenticated WhatsApp bridge IPC."""

from pathlib import Path


def test_bridge_token_is_persistent_on_disk(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import _load_or_create_bridge_token

    first = _load_or_create_bridge_token(tmp_path)
    second = _load_or_create_bridge_token(tmp_path)

    assert first == second
    assert len(first) >= 32
    assert (tmp_path / ".bridge-token").read_text(encoding="utf-8").strip() == first


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
