"""A2A inbound caller provenance: the auth facts recorded alongside an inbound task.

Invariants (not snapshots): the identity every existing decision keys on is unchanged, the
fingerprint is derived one-way from the presented credential, and neither the log line nor the
audit record ever contains that credential.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, security


def _ctx(shared: str = "", peers: dict[str, str] | None = None) -> security.A2ASecurityContext:
    """``peers`` maps token -> peer name (the order A2ASecurityContext stores in ``peer_tokens``)."""
    return security.A2ASecurityContext(
        bearer_token=shared, peer_tokens=tuple((peers or {}).items()), trusted_peers=frozenset(),
        allow_all_users=False, requested_host="127.0.0.1", push_secret="",
    )


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_classify_reports_credential_class_without_changing_identity():
    shared = "s" * 32
    peer_token = "p" * 40

    local = _ctx().classify(None, "1.2.3.4")
    assert local == {"identity": "ip:1.2.3.4", "auth": "localhost", "fingerprint": ""}

    by_shared = _ctx(shared=shared).classify(f"Bearer {shared}", "1.2.3.4")
    assert by_shared is not None
    assert by_shared["identity"] == "ip:1.2.3.4"  # unchanged shape: trust lists/rate limits still key on it
    assert by_shared["auth"] == "shared-token"
    assert by_shared["fingerprint"] == hashlib.sha256(shared.encode()).hexdigest()[:8]

    by_peer = _ctx(shared=shared, peers={peer_token: "codex-homepc"}).classify(f"Bearer {peer_token}", "1.2.3.4")
    assert by_peer is not None
    assert (by_peer["identity"], by_peer["auth"]) == ("codex-homepc", "peer-token")
    assert by_peer["fingerprint"] == hashlib.sha256(peer_token.encode()).hexdigest()[:8]

    ctx = _ctx(shared=shared, peers={peer_token: "codex-homepc"})
    assert ctx.classify("Bearer wrong-token-value", "1.2.3.4") is None
    # Observation must never diverge from the accept/reject decision it decorates.
    for header in (None, f"Bearer {shared}", f"Bearer {peer_token}", "Bearer wrong-token-value"):
        facts = ctx.classify(header, "1.2.3.4")
        assert ctx.authenticate(header, "1.2.3.4") == (facts["identity"] if facts else None)
    # A guessable credential is never fingerprinted (no offline oracle in the log).
    assert _ctx(shared="hunter2").classify("Bearer hunter2", "1.2.3.4")["fingerprint"] == ""


def test_audit_records_caller_provenance_and_never_the_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "z" * 48
    caller = {"auth": "shared-token", "fp": security.credential_fingerprint(token),
              "ua": "codex-lark-bridge/2.1", "src": "127.0.0.1:54321", "claimed": "codex-homepc"}

    security.audit("inbound", "ip:127.0.0.1", "task-1", "hello", caller=caller)

    written = (tmp_path / "a2a_audit.jsonl").read_text(encoding="utf-8")
    rec = json.loads(written.strip().splitlines()[-1])
    assert rec["peer"] == "ip:127.0.0.1"  # existing readers keep working
    assert rec["caller"] == caller
    assert caller["fp"] != token and len(caller["fp"]) == 8
    assert token not in written


@pytest.mark.integration
def test_live_request_records_who_called(monkeypatch, tmp_path, caplog):
    """Over real HTTP: a shared token degrades the peer to ``ip:<addr>``, so the UA / socket /
    credential fingerprint are the only way an operator can tell which agent sent the task."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "t" * 40
    monkeypatch.setenv("A2A_BEARER_TOKEN", token)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    port = _free_port()
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

    async def fake_handle_message(event):
        await adapter.send(event.source.chat_id, "ok", metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore[assignment]
    adapter._message_handler = object()  # non-None so dispatch proceeds

    def _post() -> dict:
        body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": "message/send",
                           "params": {"message": protocol.text_message(protocol.ROLE_USER, "hello")}}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                     "User-Agent": "codex-lark-bridge/2.1", "X-A2A-Identity": "codex-homepc"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    async def run():
        assert await adapter.connect() is True
        try:
            # A2A marshals the event onto the gateway loop via run_coroutine_threadsafe, so the
            # request must not block the loop thread itself.
            with caplog.at_level(logging.INFO):
                resp = await asyncio.to_thread(_post)
            assert resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        finally:
            await adapter.disconnect()

    asyncio.run(run())

    records = [json.loads(line) for line in (tmp_path / "a2a_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    caller = next(r["caller"] for r in records if r["direction"] == "inbound")
    assert caller["auth"] == "shared-token"
    assert caller["fp"] == hashlib.sha256(token.encode()).hexdigest()[:8]
    assert caller["ua"] == "codex-lark-bridge/2.1"
    assert caller["claimed"] == "codex-homepc"
    assert caller["src"].startswith("127.0.0.1:")
    assert token not in (tmp_path / "a2a_audit.jsonl").read_text(encoding="utf-8")
    assert "ua=codex-lark-bridge/2.1" in caplog.text and "auth=shared-token" in caplog.text
