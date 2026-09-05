"""Hermetic cross-node Bot Relay receipt integration harness.

This test-owned surface composes the canonical in-memory ``StubConnector``
with the real candidate ``bot_relay`` receipt store and
``tui_gateway.methods_bot_relay`` delivery handler.  It never opens a socket,
sends a network message, restarts a live gateway, or edits the relay owner
surfaces.

The harness is intentionally stricter than sender-side ACKs and reply files:
the target receipt must be bound to the delivery fingerprint, survive a fake
process restart, suppress a duplicate exactly once, and reject stale or
out-of-order completion state.
"""

# Provenance: imported from jonah-ux/hermes-agent commit
# eec1fbe45cd4c7248473b770491aba38dff2c25a; original file SHA-256
# 4f472a93c93ab27d54f89fd7a1899d3b79e50c524365fdf3018bed857f26cc53.
# Only this verified test file is carried forward; its sibling branch history
# is intentionally not merged into the receipt-owner branch.

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import tui_gateway.server as server
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from hermes_state import SessionDB
from tests.gateway.relay.stub_connector import StubConnector
from tools import bot_mode_dm, bot_relay


def _descriptor(label: str = "Hermes fake gateway") -> CapabilityDescriptor:
    """Use the same capability shape as the canonical relay test fixture."""

    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label=label,
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        emoji="🤖",
        platform_hint="Hermes fake gateway",
        pii_safe=False,
    )


@dataclass
class FakeGatewayProcess:
    """A restartable fake gateway sharing only its durable test root."""

    home: Path
    connector: StubConnector
    db: SessionDB
    session_id: str
    generation: int = 1

    @classmethod
    def create(cls, home: Path) -> "FakeGatewayProcess":
        profile_home = home / "profiles" / "ops"
        profile_home.mkdir(parents=True)
        db = SessionDB(profile_home / "state.db")
        session_id = db.create_session(
            f"harness-bot-chat-{os.getpid()}-{time.time_ns()}",
            "gateway_botmode",
        )
        db.set_session_title(session_id, "Bot Chat")
        db.set_session_hidden(session_id, True)
        return cls(
            home=home,
            connector=StubConnector(_descriptor()),
            db=db,
            session_id=session_id,
        )

    def restart(self) -> "FakeGatewayProcess":
        """Start a fresh fake process against the same durable target home."""

        self.db.close()
        db = SessionDB(self.home / "profiles" / "ops" / "state.db")
        return type(self)(
            home=self.home,
            connector=StubConnector(_descriptor()),
            db=db,
            session_id=self.session_id,
            generation=self.generation + 1,
        )

    def persist_target_turn(self, params: dict[str, Any]) -> None:
        """Model the fake target process's durable transcript write."""

        self.db.append_message(
            self.session_id,
            "user",
            params["message"],
            platform_message_id=params["message_id"],
            observed=True,
        )
        self.db.append_message(
            self.session_id,
            "assistant",
            "target response",
            platform_message_id=f"reply:{params['message_id']}",
            observed=True,
        )

    def read_target_turn(self) -> list[dict[str, Any]]:
        """Read the target transcript through the canonical SessionDB path."""

        return self.db.get_messages(self.session_id)


class _CompletedProcess:
    returncode = 0
    stderr = ""

    def __init__(self, stdout: str = "target response") -> None:
        self.stdout = stdout


def _params(*, key: str = "harness:delivery:one", message_id: str = "a" * 32) -> dict[str, Any]:
    message = "synthetic cross-node receipt payload"
    envelope = {
        "schema": "asm-hermes-a2a-envelope/v2",
        "message_id": message_id,
        "idempotency_key": key,
        "type": "REQUEST",
        "from_agent": "sender",
        "to_agent": "ops",
        "target_connection": "spark02",
        "target_profile": "ops",
        "target_handle": "ops",
        "message": message,
        "scope": {"mutation": "none", "production": "none"},
        "expires_at": time.time() + 600,
        "authority_effect": "none",
    }
    return {
        "profile": "ops",
        "message": message,
        "message_id": message_id,
        "idempotency_key": key,
        "envelope_schema": "asm-hermes-a2a-envelope/v2",
        "envelope": envelope,
    }


def _deliver(
    gateway: FakeGatewayProcess,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[Any, ...]],
    params: dict[str, Any],
) -> dict[str, Any]:
    monkeypatch.setenv("HERMES_HOME", str(gateway.home))

    def _run(argv: Any, **_kwargs: Any) -> _CompletedProcess:
        calls.append(tuple(argv))
        gateway.persist_target_turn(params)
        return _CompletedProcess()

    monkeypatch.setattr("subprocess.run", _run)
    response = server._methods["bot_relay.deliver"](1, params)
    assert "error" not in response
    return response["result"]


def _only_receipt(gateway: FakeGatewayProcess) -> tuple[Path, dict[str, Any]]:
    files = sorted((gateway.home / "bot_relay" / "delivered").glob("*.json"))
    assert len(files) == 1
    path = files[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_sender_ack_without_target_receipt_is_not_terminal_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sender ACK and drained outbox remain pending without exact target proof."""

    sender = FakeGatewayProcess.create(tmp_path / "sender")
    target = FakeGatewayProcess.create(tmp_path / "target")
    await sender.connector.connect()
    ack = await sender.connector.send_outbound({"op": "send", "content": "synthetic"})
    assert ack["success"] is True

    target_roster = {
        "profile": "ops",
        "handle": "ops",
        "connection_id": "spark02",
        "connection_label": "Spark02",
    }
    bot_relay.write_remote_roster(sender.home, [target_roster])

    def _sender_ack(
        _command: str,
        _label: str,
        *,
        task_id: Any,
        agent: Any,
        delivery_committed: bool = False,
    ) -> str:
        del task_id, agent, delivery_committed
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", _sender_ack)
    dispatch_ack = bot_mode_dm._try_relay_delivery(
        sender.home,
        "ops",
        "synthetic",
        "default",
        task_id=None,
        agent=None,
        metadata={"idempotency_key": "harness:sender-only"},
    )
    assert dispatch_ack is not None
    assert json.loads(dispatch_ack)["status"] == "sent"

    claimed = bot_relay.claim_pending_envelopes(sender.home)
    assert len(claimed) == 1
    assert not list((sender.home / "bot_relay" / "outbox").glob("*.json"))

    submitted: list[dict[str, Any]] = []

    @contextmanager
    def _target_db(_session: dict[str, Any]):
        yield target.db

    monkeypatch.setenv("HERMES_HOME", str(target.home))
    monkeypatch.setattr(server, "_profile_home", lambda _name: target.home / "profiles" / "ops")
    monkeypatch.setattr(server, "_session_db", _target_db)
    monkeypatch.setattr(server, "LIVE_TARGET_PERSIST_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(server, "LIVE_TARGET_PERSIST_POLL_SECONDS", 0.001)
    monkeypatch.setitem(
        server._sessions,
        target.session_id,
        {
            "profile_home": str(target.home / "profiles" / "ops"),
            "session_key": target.session_id,
            "pending_title": "Bot Chat",
            "history": [],
        },
    )
    monkeypatch.setitem(
        server._methods,
        "prompt.submit",
        lambda rid, payload: submitted.append(payload)
        or server._ok(rid, {"status": "streaming"}),
    )

    params = _params(key="harness:sender-only-target")
    target_result = server._methods["bot_relay.deliver"](2, params)
    assert target_result["error"]["code"] == 4101
    assert target_result["error"]["data"]["reason"] == "target_receipt_pending"
    assert submitted == [{
        "session_id": target.session_id,
        "text": params["message"],
        "queued": True,
        "_relay_message_id": params["message_id"],
    }]
    receipt_path, receipt = _only_receipt(target)
    assert receipt["status"] == "started"
    assert not target.read_target_turn()
    fingerprint = bot_relay.delivery_fingerprint(
        params["envelope"],
        target_profile=params["profile"],
        message=params["message"],
        structured=True,
    )
    assert bot_relay.read_idempotent_delivery(
        target.home,
        params["idempotency_key"],
        message_id=params["message_id"],
        delivery_fingerprint=fingerprint,
        target_connection=params["envelope"]["target_connection"],
        target_profile=params["envelope"]["target_profile"],
        target_handle=params["envelope"]["target_handle"],
    ) == {"disposition": "pending", "reason": "target_receipt_pending"}
    assert receipt_path.exists()
    assert not list((sender.home / "bot_relay" / "delivered").glob("*.json"))


def test_valid_receipt_with_wrong_delivery_digest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / ".hermes"
    root.mkdir()
    fingerprint = "target=ops|message=synthetic"
    identity = {
        "target_connection": "spark02",
        "target_profile": "ops",
        "target_handle": "ops",
    }
    assert bot_relay.begin_idempotent_delivery(
        root, "harness:digest", "b" * 32, fingerprint, **identity
    )["disposition"] == "admitted"
    bot_relay.complete_idempotent_delivery(
        root, "harness:digest", "target response", **identity
    )

    receipt_path = bot_relay.delivery_receipt_path(root, "harness:digest")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["delivery_sha256"] = hashlib.sha256(b"wrong delivery").hexdigest()
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    readback = bot_relay.read_idempotent_delivery(
        root,
        "harness:digest",
        message_id="b" * 32,
        delivery_fingerprint=fingerprint,
        **identity,
    )
    assert readback == {
        "disposition": "mismatch",
        "reason": "target_receipt_mismatch",
    }
    verdict = bot_relay.begin_idempotent_delivery(
        root, "harness:digest", "b" * 32, fingerprint, **identity
    )
    assert verdict["disposition"] == "conflict"


def test_completed_receipt_readback_survives_fake_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = FakeGatewayProcess.create(tmp_path / "target")
    calls: list[tuple[Any, ...]] = []
    params = _params()

    first = _deliver(gateway, monkeypatch, calls, params)
    _, receipt = _only_receipt(gateway)
    assert receipt["status"] == "completed"
    assert receipt["idempotency_sha256"] == hashlib.sha256(
        params["idempotency_key"].encode("utf-8")
    ).hexdigest()
    assert receipt["message_id"] == params["message_id"]
    assert receipt["target_connection"] == params["envelope"]["target_connection"]
    assert receipt["target_profile"] == params["envelope"]["target_profile"]
    assert receipt["target_handle"] == params["envelope"]["target_handle"]
    assert first["target_receipt"] == receipt
    fingerprint = bot_relay.delivery_fingerprint(
        params["envelope"],
        target_profile=params["profile"],
        message=params["message"],
        structured=True,
    )
    receipt_readback = bot_relay.read_idempotent_delivery(
        gateway.home,
        params["idempotency_key"],
        message_id=params["message_id"],
        delivery_fingerprint=fingerprint,
        target_connection=params["envelope"]["target_connection"],
        target_profile=params["envelope"]["target_profile"],
        target_handle=params["envelope"]["target_handle"],
    )
    assert receipt_readback["disposition"] == "completed"
    assert [row["content"] for row in gateway.read_target_turn()] == [
        params["message"],
        "target response",
    ]

    restarted = gateway.restart()
    assert restarted.generation == gateway.generation + 1
    assert [row["content"] for row in restarted.read_target_turn()] == [
        params["message"],
        "target response",
    ]
    second = _deliver(restarted, monkeypatch, calls, params)
    assert second["replayed"] is True
    assert "already delivered" in second["reply"]
    assert second["target_receipt"] == receipt
    assert len(calls) == 1


def test_duplicate_replay_runs_target_turn_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = FakeGatewayProcess.create(tmp_path / "target")
    calls: list[tuple[Any, ...]] = []
    params = _params(key="harness:duplicate")

    first = _deliver(gateway, monkeypatch, calls, params)
    second = _deliver(gateway, monkeypatch, calls, params)
    assert first["reply"] == "target response"
    assert second["replayed"] is True
    assert second["target_receipt"] == first["target_receipt"]
    assert len(calls) == 1


def test_out_of_order_completion_does_not_overwrite_newer_receipt(tmp_path: Path) -> None:
    root = tmp_path / ".hermes"
    root.mkdir()
    key = "harness:ordering"
    fingerprint = "target=ops|message=ordering"
    identity = {
        "target_connection": "spark02",
        "target_profile": "ops",
        "target_handle": "ops",
    }
    admitted = bot_relay.begin_idempotent_delivery(
        root, key, "c" * 32, fingerprint, **identity
    )
    assert admitted["disposition"] == "admitted"

    bot_relay.complete_idempotent_delivery(root, key, "newer completion", **identity)
    bot_relay.complete_idempotent_delivery(root, key, "older completion", **identity)

    receipt_path = bot_relay.delivery_receipt_path(root, key)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["reply_sha256"] == hashlib.sha256(b"newer completion").hexdigest()


def test_stale_receipt_is_swept_before_a_fresh_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = FakeGatewayProcess.create(tmp_path / "target")
    calls: list[tuple[Any, ...]] = []
    params = _params(key="harness:stale")

    _deliver(gateway, monkeypatch, calls, params)
    receipt_path, _ = _only_receipt(gateway)
    stale_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    stale_payload["completed_at"] = "1970-01-01T00:00:00+00:00"
    receipt_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    stale_at = time.time() - bot_relay.DELIVERY_RECEIPT_RETENTION_SECONDS - 1
    os.utime(receipt_path, (stale_at, stale_at))

    replay = _deliver(gateway, monkeypatch, calls, params)
    assert replay["replayed"] is True
    assert len(calls) == 1

    swept = bot_relay._sweep_stale(bot_relay.relay_root(gateway.home), now=time.time())
    assert swept == 1
    assert not receipt_path.exists()

    fresh = _deliver(gateway, monkeypatch, calls, params)
    assert fresh.get("replayed") is not True
    assert len(calls) == 2
    _, fresh_receipt = _only_receipt(gateway)
    assert fresh_receipt["status"] == "completed"
