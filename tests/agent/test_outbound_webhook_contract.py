"""Strict Task 14 outbound configuration and signature contract."""

import hashlib
import hmac
import json
import logging

import pytest

from agent import outbound_webhooks as ow
from agent import secret_scope


def _cfg(entry):
    return {"hooks": {"outbound": [entry]}}


def _entry(**overrides):
    entry = {
        "url": "https://example.com/hook",
        "events": ["on_session_end"],
    }
    entry.update(overrides)
    return entry


def test_unknown_field_rejects_entire_target():
    assert ow.iter_configured_targets(_cfg(_entry(secrete="typo"))) == []


def test_inline_plaintext_secret_reports_migration_and_disables_target(caplog):
    caplog.set_level(logging.ERROR, logger=ow.__name__)

    assert ow.iter_configured_targets(
        _cfg(_entry(secret="super-secret-value"))
    ) == []

    message = caplog.text
    assert "unsupported inline plaintext" in message
    assert "secret_ref" in message
    assert "secret_env" in message
    assert "target disabled" in message
    assert "super-secret-value" not in message


def test_missing_explicit_secret_reference_fails_closed(monkeypatch, caplog):
    monkeypatch.delenv("MISSING_WR_SECRET", raising=False)
    caplog.set_level(logging.ERROR, logger=ow.__name__)

    assert ow.iter_configured_targets(
        _cfg(_entry(secret_ref="MISSING_WR_SECRET"))
    ) == []

    assert "target disabled" in caplog.text
    assert "will not send this webhook unsigned" in caplog.text


@pytest.mark.parametrize("value", ["", "   ", None, 7])
def test_malformed_explicit_secret_reference_never_downgrades_to_unsigned(
    value, caplog,
):
    caplog.set_level(logging.ERROR, logger=ow.__name__)

    assert ow.iter_configured_targets(
        _cfg(_entry(secret_ref=value))
    ) == []

    assert "must be a non-empty secret name" in caplog.text
    assert "will not send this webhook unsigned" in caplog.text


def test_multiple_reference_fields_are_rejected(monkeypatch, caplog):
    monkeypatch.setenv("A", "one")
    monkeypatch.setenv("B", "two")
    caplog.set_level(logging.ERROR, logger=ow.__name__)

    assert ow.iter_configured_targets(
        _cfg(_entry(secret_ref="A", secret_env="B"))
    ) == []

    assert "exactly one reference field" in caplog.text


def test_no_secret_fields_is_the_only_unsigned_configuration():
    target = ow.iter_configured_targets(_cfg(_entry()))[0]
    assert target.secret is None


def test_unscoped_reference_fails_closed_without_process_env_fallback(
    monkeypatch, caplog,
):
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("PROFILE_ONLY_SECRET", "must-not-leak")
    caplog.set_level(logging.ERROR, logger=ow.__name__)

    assert ow.iter_configured_targets(
        _cfg(_entry(secret_ref="PROFILE_ONLY_SECRET"))
    ) == []

    assert "active profile secret scope" in caplog.text
    assert "no process-environment fallback was attempted" in caplog.text


def test_unexpected_secret_resolver_failure_is_not_swallowed(monkeypatch):
    def explode(_name, _default):
        raise RuntimeError("resolver import/state failure")

    monkeypatch.setattr(ow, "get_secret", explode)

    with pytest.raises(RuntimeError, match="resolver import/state failure"):
        ow.iter_configured_targets(
            _cfg(_entry(secret_ref="WR_SECRET"))
        )


def test_v2_signature_binds_timestamp_and_body(monkeypatch):
    monkeypatch.setenv("WR_SECRET", "s3cret")
    target = ow.iter_configured_targets(
        _cfg(_entry(secret_ref="WR_SECRET"))
    )[0]
    body = json.dumps({"schema_version": 1, "x": 1}).encode()
    delivery = ow._build_delivery("on_session_end", target, body, "d1")
    headers = delivery["headers"]
    timestamp = headers["X-Hermes-Timestamp"]

    expected_v2 = hmac.new(
        b"s3cret", timestamp.encode("ascii") + b"." + body, hashlib.sha256
    ).hexdigest()
    expected_legacy = hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()

    assert headers["X-Hermes-Signature-V2"] == f"sha256={expected_v2}"
    assert headers["X-Hermes-Signature-256"] == f"sha256={expected_legacy}"
    assert headers["X-Hermes-Schema-Version"] == "1"


def test_receiver_contract_documents_complete_replay_checks():
    docs = ow.__doc__ or ""
    assert "300 seconds" in docs
    assert "hmac.compare_digest" in docs
    assert "X-Hermes-Delivery" in docs
    assert "still replayable" in docs
