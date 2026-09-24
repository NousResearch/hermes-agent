"""#120937: session chat on the gateway API server must never silently truncate a long message.

Real ``python -m gateway.run`` with the API-server platform enabled through the profile ``.env``
(``API_SERVER_ENABLED``/``API_SERVER_KEY``/host/port, as a user configures it), sandboxed
HOME/HERMES_HOME, the model routed to ``FakeLLMServer`` — which records every request, so the cell
reads exactly what the model was sent.

Contract for ``POST /api/sessions/{id}/chat``: a message either reaches the model whole (length
kept, tail canary present) or the API answers an explicit 4xx — never a 200 on a cut prompt.
Controls (enforced, separate test): a 60,000-char message through the same ``/chat`` and a
100,000-char ``input`` through ``/v1/runs`` both reach the model whole.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.e2e.core.dashboard._helpers import Sandbox, make_sandbox
from tests.e2e.core.dashboard._issue_helpers import GatewayApiServer, Issue120937, xfail_known

TURN_TIMEOUT = 90.0

KNOWN: dict[str, tuple[str, type]] = {
    "test_session_chat_100k_message_reaches_model_whole_or_is_rejected": (
        "#120937 POST /api/sessions/{id}/chat silently truncates a string message to 65,536 chars "
        "(200, no error, no flag)", Issue120937),
}


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Sandbox, GatewayApiServer]]:
    root = tmp_path_factory.mktemp("gw120937")
    sb = make_sandbox(root)
    gw = None
    try:
        gw = GatewayApiServer(sb, sb.hermes_home, root / "gateway.log")
        yield sb, gw
    finally:
        if gw is not None:
            gw.stop()
        sb.finish()


def _long_message(n: int) -> tuple[str, str, str]:
    """``(message, head, tail)``: exactly ``n`` chars, a unique head marker and a unique tail canary."""
    head, tail = f"HEAD-{secrets.token_hex(6)}::", f"::TAIL-CANARY-{secrets.token_hex(8)}"
    body = ("lorem ipsum dolor sit amet " * (n // 27 + 1))[: n - len(head) - len(tail)]
    msg = head + body + tail
    assert len(msg) == n
    return msg, head, tail


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _model_saw(sb: Sandbox, head: str, timeout: float = TURN_TIMEOUT) -> str:
    """The user message (containing ``head``) of the provider's recorded main request."""
    srv = sb.profiles["default"].srv
    assert srv is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for body in reversed(srv.main_requests()):
            for m in reversed(body.get("messages") or []):
                if m.get("role") == "user" and head in _text(m.get("content")):
                    return _text(m.get("content"))
        time.sleep(0.1)
    raise AssertionError(f"the model never received a user message carrying {head!r}")


def _new_session(gw: GatewayApiServer) -> str:
    sid = f"e2e120937-{secrets.token_hex(5)}"
    status, body = gw.call("POST", "/api/sessions", {"id": sid})
    assert status == 201, f"POST /api/sessions -> {status}: {body}\n{gw.log_tail()}"
    return sid


def _stored_user_len(gw: GatewayApiServer, sid: str, head: str) -> int | None:
    status, body = gw.call("GET", f"/api/sessions/{sid}/messages")
    if status != 200:
        return None
    for m in body.get("messages") or body.get("data") or []:
        if m.get("role") == "user" and head in _text(m.get("content")):
            return len(_text(m.get("content")))
    return None


def _chat(gw: GatewayApiServer, sid: str, message: str) -> tuple[int, Any]:
    return gw.call("POST", f"/api/sessions/{sid}/chat", {"message": message}, timeout=TURN_TIMEOUT)


def test_session_chat_60k_and_runs_100k_reach_model_whole(gateway: tuple[Sandbox, GatewayApiServer]) -> None:
    """Controls (enforced): under the session-chat cap, and on /v1/runs above it, nothing is cut."""
    sb, gw = gateway
    msg, head, tail = _long_message(60_000)
    sid = _new_session(gw)
    status, body = _chat(gw, sid, msg)
    assert status == 200, f"/chat 60k -> {status}: {str(body)[:500]}\n{gw.log_tail()}"
    seen = _model_saw(sb, head)
    assert len(seen) >= len(msg) and tail in seen, (
        f"/api/sessions/{{id}}/chat delivered {len(seen)} of {len(msg)} chars (tail canary "
        f"{'present' if tail in seen else 'MISSING'})")

    msg, head, tail = _long_message(100_000)
    status, body = gw.call("POST", "/v1/runs", {"input": msg}, timeout=TURN_TIMEOUT)
    assert status == 202, f"/v1/runs -> {status}: {str(body)[:500]}"
    seen = _model_saw(sb, head)
    assert len(seen) >= len(msg) and tail in seen, (
        f"/v1/runs delivered {len(seen)} of {len(msg)} chars (tail canary {'present' if tail in seen else 'MISSING'})")
    run_id = body["run_id"]
    deadline, run = time.monotonic() + TURN_TIMEOUT, {}
    while time.monotonic() < deadline:  # settle the run so the module teardown stops an idle gateway
        st, run = gw.call("GET", f"/v1/runs/{run_id}")
        if st == 200 and run.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)
    assert run.get("status") == "completed", f"/v1/runs/{run_id} -> {run}"


@xfail_known(KNOWN, "test_session_chat_100k_message_reaches_model_whole_or_is_rejected")
def test_session_chat_100k_message_reaches_model_whole_or_is_rejected(
        gateway: tuple[Sandbox, GatewayApiServer]) -> None:
    sb, gw = gateway
    msg, head, tail = _long_message(100_000)
    sid = _new_session(gw)
    status, body = _chat(gw, sid, msg)
    if 400 <= status < 500:  # an explicit rejection honours the contract
        assert isinstance(body, dict) and body.get("error"), f"{status} without an error body: {body}"
        return
    assert status == 200, f"/chat 100k -> {status}: {str(body)[:500]}\n{gw.log_tail()}"
    seen = _model_saw(sb, head)
    if len(seen) < len(msg) or tail not in seen:
        raise Issue120937(
            f"#120937 /api/sessions/{{id}}/chat answered 200 but the model received {len(seen)} of "
            f"{len(msg)} chars (tail canary {'present' if tail in seen else 'MISSING'}; stored user "
            f"message length {_stored_user_len(gw, sid, head)}; response keys {sorted(body)[:12]})")
