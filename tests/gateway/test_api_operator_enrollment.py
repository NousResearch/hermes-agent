"""Tests for one-time operator enrollment codes and token lifecycle."""

import json

import pytest

from gateway.api_operator_auth import OperatorCredentialStore
from gateway.api_operator_enrollment import (
    MAX_FAILED_ATTEMPTS,
    PAIRING_CODE_TTL_SECONDS,
    OperatorEnrollmentStore,
)


class FakeClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_enrollment_is_single_use_and_returns_token_once(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    grant = enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["chat:write", "profiles:read"],
    )

    preview = enrollments.inspect(grant.code, "https://hermes.example")
    issued = enrollments.exchange(grant.code, "https://hermes.example")

    assert preview.scopes == ("chat:write", "profiles:read")
    assert issued.token.startswith("hop_")
    assert enrollments.exchange(grant.code, "https://hermes.example") is None


def test_origin_mismatch_does_not_consume_code(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    grant = enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["profiles:read"],
    )

    assert enrollments.inspect(grant.code, "https://attacker.example") is None
    assert enrollments.exchange(grant.code, "https://attacker.example") is None
    assert enrollments.exchange(grant.code, "https://hermes.example").token.startswith("hop_")


def test_expired_code_is_rejected(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    grant = enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["chat:read"],
    )

    clock.advance(PAIRING_CODE_TTL_SECONDS + 1)

    assert enrollments.inspect(grant.code, "https://hermes.example") is None
    assert enrollments.exchange(grant.code, "https://hermes.example") is None


def test_malformed_code_is_rejected_without_raising(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["chat:read"],
    )

    assert enrollments.inspect("", "https://hermes.example") is None
    assert enrollments.inspect("not-a-real-code", "https://hermes.example") is None
    assert enrollments.inspect(None, "https://hermes.example") is None
    assert enrollments.exchange("\x00\x01garbage", "https://hermes.example") is None


def test_repeated_failed_attempts_trigger_lockout(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    grant = enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["chat:read"],
    )

    for _ in range(MAX_FAILED_ATTEMPTS):
        assert enrollments.exchange("wrong-code", "https://hermes.example") is None

    # Locked out now: even the correct code + origin is rejected until the
    # lockout window clears.
    assert enrollments.exchange(grant.code, "https://hermes.example") is None
    assert enrollments.inspect(grant.code, "https://hermes.example") is None


def test_plaintext_code_is_never_persisted(tmp_path):
    clock = FakeClock(1000.0)
    credentials = OperatorCredentialStore(tmp_path / "credentials.json")
    enrollments = OperatorEnrollmentStore(
        tmp_path / "enrollments.json", credentials, now=clock
    )
    grant = enrollments.create(
        label="Galaxy S24",
        origin="https://hermes.example",
        scopes=["chat:read"],
    )

    raw_text = (tmp_path / "enrollments.json").read_text(encoding="utf-8")
    assert grant.code not in raw_text

    stored = json.loads(raw_text)
    entry = next(iter(stored["enrollments"].values()))
    assert set(entry.keys()) == {
        "hash", "salt", "label", "origin", "scopes",
        "created_at", "expires_at", "consumed_at",
    }
