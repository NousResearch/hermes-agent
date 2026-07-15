from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from gateway.api_verifier_credentials import (
    build_api_approval_scrypt_verifier,
    build_api_bearer_verifier,
    parse_api_approval_scrypt_verifier,
    parse_api_bearer_verifier,
)
from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter
from gateway.systemd_credentials import (
    GATEWAY_API_APPROVAL_VERIFIER_CREDENTIAL,
    GATEWAY_API_BEARER_VERIFIER_CREDENTIAL,
)


BEARER = "production-bearer-for-tests-0123456789abcdef"
PASSKEY = "production-owner-passkey-tests-0123456789abcdef"


def _adapter(monkeypatch: pytest.MonkeyPatch) -> APIServerAdapter:
    bearer = build_api_bearer_verifier(BEARER).decode("ascii")
    approval = build_api_approval_scrypt_verifier(
        PASSKEY,
        salt=b"s" * 32,
    ).decode("ascii")
    monkeypatch.setattr(
        api_server,
        "_load_systemd_api_bearer_verifier_credential",
        lambda name: bearer,
    )
    monkeypatch.setattr(
        api_server,
        "_load_systemd_api_approval_verifier_credential",
        lambda name: approval,
    )
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "key_verifier_credential": (
                    GATEWAY_API_BEARER_VERIFIER_CREDENTIAL
                ),
                "approval_verifier_credential": (
                    GATEWAY_API_APPROVAL_VERIFIER_CREDENTIAL
                ),
            },
        )
    )


def _request(*, bearer: str = BEARER, peer: str = "127.0.0.1") -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {bearer}"}
    request.secure = False
    request.remote = peer
    request.method = "POST"
    request.path_qs = "/v1/approvals/example/response"
    request.transport.get_extra_info.return_value = (peer, 12345)
    return request


def _authority(*, passkey: str = PASSKEY, nonce: str = "3" * 32) -> dict:
    return {
        "schema": api_server.API_APPROVAL_PASSKEY_AUTHORITY_SCHEMA,
        "nonce": nonce,
        "issued_at_unix": 1_000,
        "expires_at_unix": 1_100,
        "capability_epoch_sha256": "2" * 64,
        "passkey": passkey,
    }


def test_gateway_retains_only_public_verifiers_and_digest_disclosure_is_useless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    try:
        assert adapter._api_key == ""
        assert adapter._approval_passkey == ""
        assert adapter._api_bearer_verifier == parse_api_bearer_verifier(
            build_api_bearer_verifier(BEARER)
        )
        assert adapter._approval_passkey_verifier == (
            parse_api_approval_scrypt_verifier(
                build_api_approval_scrypt_verifier(PASSKEY, salt=b"s" * 32)
            )
        )
        assert adapter._check_auth(_request()) is None
        disclosed = adapter._api_bearer_verifier.sha256_hex
        rejected = adapter._check_auth(_request(bearer=disclosed))
        assert rejected is not None
        assert rejected.status == 401
        assert PASSKEY not in repr(adapter.__dict__)
        assert BEARER not in repr(adapter.__dict__)
    finally:
        adapter._response_store.close()


def test_verifier_positive_approval_is_exact_short_lived_and_replay_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(api_server.time, "time", lambda: 1_050)
    authority = _authority()
    try:
        first = adapter._verify_and_consume_api_approval_authority(
            authority,
            session_id="session-one",
            approval_id="1" * 32,
            choice="once",
            capability_epoch_sha256="2" * 64,
            request=_request(),
        )
        assert first is None
        assert PASSKEY not in json.dumps(adapter._api_consumed_approval_nonces)
        replay = adapter._verify_and_consume_api_approval_authority(
            authority,
            session_id="session-one",
            approval_id="1" * 32,
            choice="once",
            capability_epoch_sha256="2" * 64,
            request=_request(),
        )
        assert replay is not None
        assert replay.status == 409
    finally:
        adapter._response_store.close()


def test_verifier_approval_rejects_verifier_disclosure_and_nonsecure_remote_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(api_server.time, "time", lambda: 1_050)
    try:
        disclosed = adapter._approval_passkey_verifier.verifier.hex()
        invalid = adapter._verify_and_consume_api_approval_authority(
            _authority(passkey=disclosed),
            session_id="session-one",
            approval_id="1" * 32,
            choice="once",
            capability_epoch_sha256="2" * 64,
            request=_request(),
        )
        assert invalid is not None
        assert invalid.status == 403

        remote = adapter._verify_and_consume_api_approval_authority(
            _authority(nonce="4" * 32),
            session_id="session-one",
            approval_id="1" * 32,
            choice="once",
            capability_epoch_sha256="2" * 64,
            request=_request(peer="203.0.113.7"),
        )
        assert remote is not None
        assert remote.status == 403
    finally:
        adapter._response_store.close()


def test_verifier_config_rejects_any_secret_bearing_parallel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_SERVER_KEY", BEARER)
    with pytest.raises(ValueError, match="cannot be combined"):
        APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key_verifier_credential": (
                        GATEWAY_API_BEARER_VERIFIER_CREDENTIAL
                    )
                },
            )
        )
