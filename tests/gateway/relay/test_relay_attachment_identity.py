"""Supported relay identity regressions for candidate2ddd.

Provenance: test-only file from corrected worker commit
eca499865c612b4aff3e60c093c7b54c43680d2f. Only this file is carried into
the existing PR #101435 owner branch; the worker commit ancestry and any
unsupported attachment hypothesis are not integrated.

These tests exercise the existing Desktop-to-gateway relay boundary with
synthetic metadata only.  They never open media, contact a peer, or start a
real Hermes process.

Canonical contract note: the supported v2 relay identity is the message
payload plus the declared message/target identity (message ID, idempotency
key, target connection, target profile, and target handle).  Attachment
metadata, an attachment list, and ``content_digest`` are not fields in the
sender's ``RelayEnvelope`` and are ignored when an adversarial caller adds
them.  The earlier hypothesis that swapping those unsupported fields should
conflict is therefore refuted; this file records that they replay the same
semantic delivery exactly once.  The supported message and target identity
swap cases are the regression consumer for the relay owner branch/CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import tui_gateway.server as srv
from tools import bot_relay


MESSAGE = "inspect the synthetic relay artifact"
IDEMPOTENCY_KEY = "candidate2ddd:relay:message-1"
UNSUPPORTED_CONTENT_DIGEST_A = "sha256:" + ("a" * 64)
UNSUPPORTED_CONTENT_DIGEST_B = "sha256:" + ("b" * 64)
UNSUPPORTED_ATTACHMENT_METADATA_A = {
    "name": "plan-a.txt",
    "kind": "file",
    "size": 11,
    "sha256": "a" * 64,
}
UNSUPPORTED_ATTACHMENT_METADATA_B = {
    "name": "plan-b.txt",
    "kind": "file",
    "size": 17,
    "sha256": "b" * 64,
}
UNSUPPORTED_ATTACHMENTS_A = [
    {"name": "plan-a.txt", "kind": "file", "sha256": "a" * 64}
]
UNSUPPORTED_ATTACHMENTS_B = [
    {"name": "plan-b.txt", "kind": "file", "sha256": "b" * 64}
]


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give each case an isolated synthetic Hermes home and session table."""

    hermes_home = tmp_path / ".hermes"
    for profile in ("ops", "other"):
        (hermes_home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    srv._sessions.clear()
    yield hermes_home
    srv._sessions.clear()


def _relay_params(
    *,
    profile: str = "ops",
    target_connection: str = "connection-a",
    message_id: str = "a" * 32,
    message: str = MESSAGE,
    attachment_metadata: dict = UNSUPPORTED_ATTACHMENT_METADATA_A,
    attachments: list[dict] = UNSUPPORTED_ATTACHMENTS_A,
    content_digest: str = UNSUPPORTED_CONTENT_DIGEST_A,
) -> dict:
    """Build one valid v2 envelope with supported and ignored fields."""

    envelope = {
        "schema": "asm-hermes-a2a-envelope/v2",
        "message_id": message_id,
        "idempotency_key": IDEMPOTENCY_KEY,
        "type": "REQUEST",
        "from_agent": "sender",
        "to_agent": profile,
        "target_profile": profile,
        "target_connection": target_connection,
        "message": message,
        "scope": {"mutation": "none", "production": "none"},
        "expires_at": time.time() + 600,
        "authority_effect": "none",
        "attachment_metadata": attachment_metadata,
        "attachments": attachments,
        "content_digest": content_digest,
    }
    return {
        "profile": profile,
        "message": message,
        "message_id": message_id,
        "idempotency_key": IDEMPOTENCY_KEY,
        "envelope_schema": envelope["schema"],
        "envelope": envelope,
    }


def _deliver_twice(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_params: dict,
    second_params: dict,
) -> tuple[dict, list[tuple[tuple, dict]], dict]:
    """Complete one synthetic turn, then attempt a mutated replay."""

    calls: list[tuple[tuple, dict]] = []

    class Process:
        returncode = 0
        stdout = "synthetic relay response"
        stderr = ""

    def fake_run(*args, **kwargs):
        argv = args[0] if args else ()
        if isinstance(argv, (list, tuple)) and "--query-file" in argv:
            calls.append((args, kwargs))
        return Process()

    # The gateway handler imports subprocess itself; patching the module-level
    # function keeps the test hermetic while preserving the argv/receipt path.
    monkeypatch.setattr("subprocess.run", fake_run)
    first = srv._methods["bot_relay.deliver"](1, first_params)
    assert "error" not in first, first

    receipt_path = bot_relay.delivery_receipt_path(home, IDEMPOTENCY_KEY)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    second = srv._methods["bot_relay.deliver"](2, second_params)
    return second, calls, receipt


def _assert_idempotency_conflict(response: dict) -> None:
    assert response["error"]["data"]["reason"] == "idempotency_conflict", response


def test_enqueue_ignores_unsupported_attachment_metadata(home: Path) -> None:
    """Unsupported attachment-shaped metadata is not part of the relay envelope."""

    target = {
        "profile": "ops",
        "handle": "ops",
        "connection_id": "connection-a",
    }
    unsupported = {
        "attachment_metadata": UNSUPPORTED_ATTACHMENT_METADATA_A,
        "attachments": UNSUPPORTED_ATTACHMENTS_A,
        "content_digest": UNSUPPORTED_CONTENT_DIGEST_A,
    }
    envelope = bot_relay.enqueue_envelope(
        home,
        target=target,
        message=MESSAGE,
        sender_profile="sender",
        sender_handle="sender",
        metadata=unsupported,
    )
    outbox_path = bot_relay.relay_root(home) / bot_relay.OUTBOX_DIR / f"{envelope['id']}.json"
    serialized = json.loads(outbox_path.read_text(encoding="utf-8"))
    for field in unsupported:
        assert field not in serialized


def test_completed_receipt_records_supported_target_identity_only(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, receipt = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(),
        _relay_params(),
    )
    assert receipt["message_id"] == "a" * 32
    assert receipt["target_connection"] == "connection-a"
    assert receipt["target_profile"] == "ops"
    assert receipt["target_handle"] == "ops"
    assert "attachment_metadata" not in receipt
    assert "attachments" not in receipt
    assert "content_digest" not in receipt


def test_unsupported_attachment_metadata_and_digest_do_not_change_delivery_identity(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second, calls, _ = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(),
        _relay_params(
            attachment_metadata=UNSUPPORTED_ATTACHMENT_METADATA_B,
            attachments=UNSUPPORTED_ATTACHMENTS_B,
            content_digest=UNSUPPORTED_CONTENT_DIGEST_B,
        ),
    )
    assert second["result"]["replayed"] is True, second
    assert len(calls) == 1


def test_changed_message_conflicts_with_completed_target_receipt(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second, calls, _ = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(message="first supported message"),
        _relay_params(message="second supported message"),
    )
    _assert_idempotency_conflict(second)
    assert len(calls) == 1


def test_swapped_message_id_cannot_replay_target_receipt(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second, calls, _ = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(),
        _relay_params(message_id="b" * 32),
    )
    _assert_idempotency_conflict(second)
    assert len(calls) == 1


def test_swapped_target_connection_cannot_replay_target_receipt(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second, calls, _ = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(),
        _relay_params(target_connection="connection-b"),
    )
    _assert_idempotency_conflict(second)
    assert len(calls) == 1


def test_swapped_target_profile_conflicts_with_completed_target_receipt(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second, calls, _ = _deliver_twice(
        home,
        monkeypatch,
        _relay_params(),
        _relay_params(profile="other", target_connection="connection-a"),
    )
    _assert_idempotency_conflict(second)
    assert len(calls) == 1
