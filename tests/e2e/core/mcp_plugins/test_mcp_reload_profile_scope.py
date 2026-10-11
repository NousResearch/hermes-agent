"""``/reload-mcp`` restarts only the CALLING profile's MCP servers (#58414), with REAL servers.

One ``tui_gateway`` host (the Ink TUI / Desktop backend) serves the launch profile and a named
profile ``work``, each configured with its own real stdio fixture server. Every server process logs
what it receives together with its pid, so a (re)start is a new pid sending ``initialize``:

* a reload from ``work`` starts a new ``work`` server and leaves the launch profile's server
  process alone; that same untouched process still answers the launch chat's next tool call;
* a reload from the launch profile leaves ``work``'s server alone, and the ``work`` chat's
  model-facing ``tools[]`` and system prompt (the cached prompt prefix) stay byte-identical;
* ``slash.exec /reload-mcp now`` (what an older Desktop sends) reaches the live session's pool, not
  the slash worker's own;
* a server of the calling profile that cannot start is named in ``failed``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import hermes_yaml as yaml
from tests.e2e.core._pm_dependencies import select_test_dependencies
from tests.e2e.core.mcp_plugins._helpers import (
    FINAL,
    REPO_ROOT,
    E2EHome,
    build_home as _build_home,
    calls_received,
    provider,
    script,
    stdio_server,
    tool_name,
    tool_results,
)
from tests.e2e.core.mcp_plugins._plugin_helpers import tui_host
from tests.e2e.core.parity._drive_rpc import READY_TIMEOUT
from tests.e2e.core.parity._helpers import wait_until
from tests.fakes.fake_llm_provider import write_hermes_home

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs only processes carrying this test's tag
]

LAUNCH, WORK = "alpha", "beta"
RELOAD_TIMEOUT = 180.0
CALL_RE = re.compile(r"CALL-ALPHA:([A-Za-z0-9]+)")


def _last_user(body: dict[str, Any]) -> str:
    users = [m for m in body.get("messages") or [] if m.get("role") == "user"]
    content = users[-1].get("content") if users else ""
    return content if isinstance(content, str) else json.dumps(content)


def _call_alpha_on_demand(record: dict[str, Any]):
    match = CALL_RE.search(_last_user(record["body"]))
    return script(*(((tool_name(LAUNCH, "ro_probe"), {"nonce": match.group(1)}),) if match else ()))(record)


def _home_with_profile(root: Path, base_url: str, *, extra_work: dict[str, dict] | None = None) -> E2EHome:
    """Launch profile with server ``alpha``, named profile ``work`` with server ``beta`` (+ ``extra_work``)."""
    eh = _build_home(root, base_url)
    select_test_dependencies(eh.hermes_home, REPO_ROOT)
    eh.extra_env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    eh.update_config(lambda cfg: cfg.__setitem__("mcp_servers", {LAUNCH: stdio_server(LAUNCH, root / "alpha.jsonl", eh.tag)}))
    work = eh.hermes_home / "profiles" / "work"
    work.mkdir(parents=True)
    write_hermes_home(work, base_url)
    select_test_dependencies(work, REPO_ROOT)
    cfg = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
    cfg.update({"mcp_servers": {WORK: stdio_server(WORK, root / "beta.jsonl", eh.tag), **(extra_work or {})},
                "mcp_discovery_timeout": 120, "mcp_single_query_discovery_timeout": 120, "updates": {"check": False}})
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return eh


def _server_pids(log: Path) -> list[int]:
    """Pids that opened a session (sent ``initialize``), in start order."""
    if not log.exists():
        return []
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [rec["pid"] for rec in records if isinstance(rec["msg"], dict) and rec["msg"].get("method") == "initialize"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _turn_request(bodies: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    """The request that opened the turn whose prompt carries ``marker``."""
    for body in bodies:
        msgs = body.get("messages") or []
        if marker in _last_user(body) and msgs and msgs[-1].get("role") == "user":
            return body
    raise AssertionError(f"no request opened the {marker!r} turn")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _work_session(host) -> str:
    return host.rpc.call("session.create", {"profile": "work"}, timeout=READY_TIMEOUT)["session_id"]


def _when_idle(send):
    """``send()`` as a client retries it: a turn still settling after ``message.complete`` (title, memory)
    keeps its session running, and a reload refuses with 4009 until it settles."""
    def attempt():
        try:
            return send()
        except AssertionError as exc:
            if "'code': 4009" not in str(exc):
                raise
            return None
    return wait_until(attempt, RELOAD_TIMEOUT, "the reload to be accepted once the profile's turns settled")


def _reload(host, sid: str) -> dict:
    return _when_idle(lambda: host.rpc.call("reload.mcp", {"session_id": sid, "confirm": True}, timeout=RELOAD_TIMEOUT))


def test_reload_restarts_only_the_calling_profiles_servers(tmp_path: Path) -> None:
    launch_log, work_log = tmp_path / "alpha.jsonl", tmp_path / "beta.jsonl"
    with provider(_call_alpha_on_demand) as srv:
        eh = _home_with_profile(tmp_path, srv.base_url)
        with tui_host(eh) as host:
            launch_sid, work_sid = host.new_session(), _work_session(host)
            assert FINAL in host.turn(launch_sid, "LAUNCH-ONE warm up")
            assert FINAL in host.turn(work_sid, "WORK-ONE warm up")
            launch_pids, work_pids = _server_pids(launch_log), _server_pids(work_log)
            assert launch_pids and work_pids, (launch_pids, work_pids)

            result = _reload(host, work_sid)
            assert result["status"] == "reloaded" and result["failed"] == [], result
            assert set(_server_pids(work_log)) - set(work_pids), "the calling profile's server was not restarted"
            assert _server_pids(launch_log) == launch_pids, "a reload from 'work' restarted the launch profile's server"
            assert _alive(launch_pids[-1]), "a reload from 'work' stopped the launch profile's server"

            assert FINAL in host.turn(launch_sid, "LAUNCH-TWO CALL-ALPHA:afterworkreload")
            assert [c.get("arguments") for c in calls_received(launch_log, "ro_probe")] == [{"nonce": "afterworkreload"}]
            assert _server_pids(launch_log) == launch_pids, "the launch chat's call needed a new server process"

            work_pids = _server_pids(work_log)
            result = _reload(host, launch_sid)
            assert result["status"] == "reloaded", result
            assert set(_server_pids(launch_log)) - set(launch_pids), "the calling profile's server was not restarted"
            assert _server_pids(work_log) == work_pids, "a reload from the launch profile restarted 'work''s server"
            assert _alive(work_pids[-1]), "a reload from the launch profile stopped 'work''s server"

            assert FINAL in host.turn(work_sid, "WORK-TWO after the launch reload")
            before = _turn_request(srv.main_requests(), "WORK-ONE")
            after = _turn_request(srv.main_requests(), "WORK-TWO")
            assert _canon(after.get("tools")) == _canon(before.get("tools")), (
                "a reload from the launch profile changed the 'work' chat's tools[] (prompt-cache prefix)")
            assert _canon(after["messages"][0]) == _canon(before["messages"][0]), (
                "a reload from the launch profile rewrote the 'work' chat's system prompt (prompt-cache prefix)")


def test_slash_exec_reload_reaches_the_live_session_and_names_failed_servers(tmp_path: Path) -> None:
    work_log = tmp_path / "beta.jsonl"
    with provider(script()) as srv:
        broken = {"broken": {"command": str(tmp_path / "no-such-binary"), "connect_timeout": 5}}
        eh = _home_with_profile(tmp_path, srv.base_url, extra_work=broken)
        with tui_host(eh) as host:
            work_sid = _work_session(host)
            assert FINAL in host.turn(work_sid, "WORK-ONE warm up")
            work_pids = _server_pids(work_log)
            assert work_pids, "the 'work' server never started"

            def slash_reload():
                output = host.rpc.call("slash.exec", {"session_id": work_sid, "command": "/reload-mcp now"},
                                       timeout=RELOAD_TIMEOUT)["output"]
                return None if output.startswith("session busy") else output

            output = wait_until(slash_reload, RELOAD_TIMEOUT, "/reload-mcp to run once the turn settled")
            assert output.startswith("MCP servers reloaded"), output
            assert "Did not connect: broken" in output, output
            assert set(_server_pids(work_log)) - set(work_pids), (
                "slash.exec /reload-mcp never restarted the live session's server (it ran in the slash worker)")


GAMMA = "gamma"
CALL_GAMMA_RE = re.compile(r"CALL-GAMMA:([A-Za-z0-9]+)")


def _call_gamma_on_demand(record: dict[str, Any]):
    match = CALL_GAMMA_RE.search(_last_user(record["body"]))
    return script(*(((tool_name(GAMMA, "ro_probe"), {"nonce": match.group(1)}),) if match else ()))(record)


def test_a_server_added_mid_chat_is_usable_in_that_chat_after_reload(tmp_path: Path) -> None:
    """The issue's own scenario: a server is added to config while a chat is open. That chat cannot
    reach it until ``/reload-mcp``; after the reload the same chat's next turn calls it."""
    gamma_log = tmp_path / "gamma.jsonl"
    with provider(_call_gamma_on_demand) as srv:
        eh = _home_with_profile(tmp_path, srv.base_url)
        with tui_host(eh) as host:
            sid = host.new_session()
            assert FINAL in host.turn(sid, "ADD-ONE warm up")
            eh.update_config(lambda cfg: cfg["mcp_servers"].__setitem__(
                GAMMA, stdio_server(GAMMA, gamma_log, eh.tag, MCPE2E_CANARY="GAMMA-LIVE")))

            assert FINAL in host.turn(sid, "ADD-TWO CALL-GAMMA:beforereload")
            assert calls_received(gamma_log, "ro_probe") == [], "the new server was reachable before /reload-mcp"

            result = _reload(host, sid)
            assert result["status"] == "reloaded" and result["failed"] == [], result

            assert FINAL in host.turn(sid, "ADD-THREE CALL-GAMMA:afterreload")
            assert [c.get("arguments") for c in calls_received(gamma_log, "ro_probe")] == [{"nonce": "afterreload"}]
            assert any("RO:GAMMA-LIVE:afterreload" in r for r in tool_results(srv)), tool_results(srv)
