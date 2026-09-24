"""MCP over streamable HTTP against a REAL ``mcp`` 2.x server subprocess (``mcp_fixture_server.py``).

The server records every JSON-RPC body it receives (an ASGI wrapper in front of the SDK's
``streamable_http_app``), can answer ``tools/call`` with HTTP 401 from an on-disk budget, and can
be crashed mid-call and restarted on the same port. Assertions read the server's inbound log and
the tool results the model saw on the next provider request:

* a no-required-param tool receives ``arguments: {}`` over HTTP and no information-free
  ``params._meta`` (some hosted servers reject ``_meta: {}`` with HTTP 400, #120923);
* a 401 on ``tools/call`` takes the auth recovery path (the model is told the server needs
  authentication, #121285) and never poisons the connection: the next call succeeds;
* the server crashing mid-call fails THAT call as a tool error (the turn still completes, the
  call is not replayed), and after the server comes back the next turn in the same long-lived
  host session reaches the new server process.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.mcp_plugins._helpers import (
    FINAL,
    HttpMcpServer,
    build_home,
    call_tool,
    calls_received,
    http_server_cfg,
    inbound,
    provider,
    run_chat_q,
    script,
    tool_name,
    tool_results,
)
from tests.e2e.core.mcp_plugins._plugin_helpers import reap_tagged, tui_host
from tests.fakes.fake_llm_provider import Text

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="orphan sweep uses /proc"),
    pytest.mark.live_system_guard_bypass,  # reap_tagged signals only this run's tagged tree
]

SERVER = "web"

KNOWN: dict[str, str] = {
    "test_no_information_free_meta_is_sent_over_http":
        "#120923 empty params._meta sent on every request; some hosted MCP servers answer HTTP 400",
    "test_401_on_tools_call_is_reported_as_an_auth_failure":
        "#121285 mcp 2.x folds a tools/call 401 into a generic MCPError; auth recovery never runs",
}


@pytest.fixture(autouse=True)
def _known(request: pytest.FixtureRequest) -> None:
    reason = KNOWN.get(request.node.name)
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))


def _payload(result: str) -> dict[str, Any]:
    match = re.search(r"^\{.*\}$", result, re.M | re.S)
    assert match, f"no JSON payload in tool result: {result!r}"
    return json.loads(match.group(0))


def _one_turn(root: Path, calls: list[tuple[str, dict]], *, unauthorized_calls: int = 0) -> dict[str, Any]:
    fault = root / "401_budget"
    fault.write_text(str(unauthorized_calls), encoding="utf-8")
    with provider(script(*[(tool_name(SERVER, n), a) for n, a in calls])) as srv:
        eh = build_home(root, srv.base_url)
        http = HttpMcpServer(root, eh.tag, MCPE2E_CANARY=f"CANARY-{root.name}", MCPE2E_401_CALLS=str(fault)).start()
        eh.update_config(lambda c: c["mcp_servers"].__setitem__(SERVER, http_server_cfg(http.url)))
        try:
            proc = run_chat_q(eh, "Use the web tools, then report.")
        finally:
            http.stop()
            reap_tagged(eh)
        assert proc.returncode == 0 and FINAL in proc.stdout, (
            f"turn did not complete: exit {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        return {"log": http.log, "results": tool_results(srv), "canary": f"CANARY-{root.name}"}


# Call shape ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shape(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _one_turn(tmp_path_factory.mktemp("shape"), [("noargs_probe", {}), ("optional_obj_probe", {"parameters": {}})])


def test_no_required_param_call_over_http_sends_an_arguments_object(shape: dict[str, Any]) -> None:
    got = [(p["name"], p.get("arguments")) for n in ("noargs_probe", "optional_obj_probe")
           for p in calls_received(shape["log"], n)]
    assert got == [("noargs_probe", {}), ("optional_obj_probe", {"parameters": {}})], got
    assert all(shape["canary"] in r for r in shape["results"]), shape["results"]


def test_no_information_free_meta_is_sent_over_http(shape: dict[str, Any]) -> None:
    requests = [m for m in inbound(shape["log"]) if isinstance(m, dict) and "id" in m and "method" in m]
    assert requests, "the server logged no requests"
    empty = [m["method"] for m in requests if "_meta" in (m.get("params") or {}) and not m["params"]["_meta"]]
    assert not empty, f"requests carried an empty/null params._meta: {empty}"


# 401 on tools/call ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unauthorized(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _one_turn(tmp_path_factory.mktemp("unauth"), [("noargs_probe", {}), ("noargs_probe", {})],
                     unauthorized_calls=1)


def test_401_on_tools_call_is_reported_as_an_auth_failure(unauthorized: dict[str, Any]) -> None:
    assert any("injected_401_for" in m for m in inbound(unauthorized["log"]) if isinstance(m, dict)), (
        "fixture never answered 401 (vacuous)")
    first = _payload(unauthorized["results"][0])
    assert "error" in first, first
    assert re.search(r"auth|401|sign.?in|credential", json.dumps(first), re.I), (
        f"a 401 on tools/call reached the model without any sign it is an auth failure: {first}")


def test_the_call_after_a_401_reaches_the_server(unauthorized: dict[str, Any]) -> None:
    served = [m for m in inbound(unauthorized["log"])
              if isinstance(m, dict) and m.get("method") == "tools/call"]
    assert len(served) >= 2, f"the call after the 401 never reached the server: {served}"
    second = _payload(unauthorized["results"][-1])
    assert f"NOARGS:{unauthorized['canary']}" in json.dumps(second), (
        f"the connection stayed broken after one 401: {second}")


# Crash mid-call, then reconnect in the same long-lived session ------------------------------------

# Turn-2 plan after the server restarts, and which of its calls must return the canary.
# write-capable: the first call may fail as outcome-uncertain (stale session, at-most-once), the
# one after it must reach the new server. read-only: replay after session expiry is safe, so the
# FIRST call must already succeed.
RECONNECT_PLANS: dict[str, tuple[list[str], int, str]] = {
    "write-capable": (["noargs_probe", "noargs_probe"], -1, "NOARGS:CANARY-crash"),
    "read-only": (["ro_probe"], 0, "RO:CANARY-crash:"),
}
KNOWN["test_server_crash_mid_call_fails_that_call_and_the_next_turn_reconnects[read-only]"] = (
    "#121042 readOnlyHint unseen under mcp 2.x, so a read-only call is not replayed after session expiry")


def _planner(turn2: list[str]):
    """Turn 'CRASH-TURN' calls crash_probe once; the next turn runs ``turn2``; then answer."""

    def respond(record: dict[str, Any]):
        body = record["body"]
        msgs = body.get("messages") or []
        last_user = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
        done = sum(1 for m in msgs[last_user + 1:] if m.get("role") == "tool")
        plan = ["crash_probe"] if "CRASH-TURN" in json.dumps(msgs[last_user].get("content")) else turn2
        return Text(FINAL) if done >= len(plan) else call_tool(body, tool_name(SERVER, plan[done]), {})

    return respond


@pytest.mark.parametrize("kind", list(RECONNECT_PLANS))
def test_server_crash_mid_call_fails_that_call_and_the_next_turn_reconnects(tmp_path: Path, kind: str) -> None:
    turn2, must_succeed, canary = RECONNECT_PLANS[kind]
    with provider(_planner(turn2)) as srv:
        eh = build_home(tmp_path, srv.base_url)
        http = HttpMcpServer(tmp_path, eh.tag, MCPE2E_CANARY="CANARY-crash").start()
        eh.update_config(lambda c: c["mcp_servers"].__setitem__(SERVER, http_server_cfg(http.url)))
        try:
            with tui_host(eh) as host:
                sid = host.new_session()
                assert FINAL in host.turn(sid, "CRASH-TURN: use the web tool.")
                crashed = tool_results(srv)
                assert calls_received(http.log, "crash_probe"), f"crash_probe never reached the server: {crashed}"
                assert http.wait_exit(30) == 7, "fixture server did not crash in the call"
                assert len(crashed) == 1 and "error" in _payload(crashed[0]), crashed
                assert len(calls_received(http.log, "crash_probe")) == 1, "crashed call was replayed"
                assert host.proc.poll() is None, f"host died with the MCP server:\n{host.cap.stderr[-2000:]}"

                old_pid = http.proc.pid if http.proc else None
                http.start()  # same port: the configured URL is unchanged
                assert FINAL in host.turn(sid, "Use the web tool again.")
                after = tool_results(srv)[1:]
                assert len(after) == len(turn2), after
                assert canary in after[must_succeed], f"{kind}: next turn did not reach the restarted server: {after}"
                served_by = {m["pid"] for m in _raw(http.log) if m["msg"].get("method") == "tools/call"
                             and (m["msg"].get("params") or {}).get("name") == turn2[0]}
                assert served_by and old_pid not in served_by, served_by
        finally:
            http.stop()
            reap_tagged(eh)


def _raw(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
