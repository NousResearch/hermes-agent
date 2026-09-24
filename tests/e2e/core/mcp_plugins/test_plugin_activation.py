"""Plugin activation with REAL MCP servers (Agent Plugins v1 portable packages).

Every test drives a real Hermes process against the recording fake LLM and the real ``mcp``
fixture server, launched from a portable package under ``<HERMES_HOME>/plugins/<dir>/``:

* live activation: a plugin enabled over the Plugins Hub RPC (``plugins.manage toggle``) while a
  chat is open in the ``tui_gateway`` host (the Ink TUI / Desktop backend) is usable in that SAME
  chat on its next turn (the call reaches the server, its canary reaches the model), the open
  chat's model-facing ``tools[]`` stays byte-identical (prompt-cache invariant: MCP tools go live
  deferred behind ``tool_call``), and a new chat gets the tool too;
* a resource-only portable server activated live is reported connected (#119751);
* ``${VAR}`` placeholders in a portable ``mcp.json`` ``env`` reach the server interpolated from the
  profile's ``.env`` (#120526), with the native ``mcp_servers`` path as the passing control, and
  that same enabled server is not called an unknown toolset at startup (#119457);
* two user plugin dirs declaring the same manifest name: the dir named like the manifest (not the
  ``.bak-*`` copy next to it) is what ``hermes plugins list`` shows and what a real turn runs,
  and the collision is reported (#121078).

Open bugs are run-time gated through ``KNOWN`` (``known_gate`` around the bug's own assertion; drop
the entry when its fix lands); only a ``KnownSymptom`` raised by that assertion whose message matches
the entry's pattern counts as the bug, so a boot failure, a precondition, a timeout or a crash stays red.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.mcp_plugins._helpers import (
    FINAL,
    KnownSymptom,
    build_home,
    calls_received,
    inbound,
    provider,
    run_chat_q,
    script,
    stdio_server,
    symptom,
    tool_name,
    tool_names,
    tool_results,
)
from tests.e2e.core.mcp_plugins._plugin_helpers import portable_stdio, reap_tagged, tui_host, write_portable_plugin
from tests.e2e.core.parity._helpers import hermes_argv
from tests.e2e.core._pending_fixes import known_gate

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs only processes carrying this test's tag
]

# Confirmed-live open bugs: test id -> (the symptom's own message pattern, "#issue reason"). Run-time
# gated around the symptom() check only (known_gate); drop an entry when its fix lands.
KNOWN: dict[str, tuple[str, str]] = {
    "test_resource_only_plugin_activated_live_is_reported_connected": (
        r"^a resource-only MCP server that completed the handshake is reported as failed: "
        r"\{.*'connected': False",
        "#119751 live activation filters resource/prompt wrappers before the connected check"),
    "test_portable_mcp_env_placeholder_is_interpolated": (
        r"^the portable plugin's server received the literal placeholder: ENV:NO-CANARY:\$\{E2E_PORTABLE_KEY\}",
        "#120526 portable mcp.json env ${VAR} reaches the server literally"),
    "test_enabled_portable_plugin_server_is_not_reported_as_an_unknown_toolset": (
        r"^`hermes chat` warned about the enabled plugin's MCP server: \[.*Unknown toolsets: .*\bplug\b",
        "#119457 startup 'Unknown toolsets' warning names a plugin-provided MCP server"),
    "test_same_name_backup_dir_does_not_shadow_the_live_plugin": (
        r"^(`hermes plugins list` shows the backup copy instead of plugins/foo \(v2\.0\.0\)"
        r"|a real turn ran the backup dir's MCP server, not plugins/foo's)",
        "#121078 the later-sorting plugins/foo.bak-* wins a same-name user collision"),
    "test_same_name_plugin_collision_is_reported": (
        r"^two user plugin dirs declare the same name 'foo' \(.+\) but no user-visible surface names both",
        "#121078 same-source manifest name collision is silent"),
}


PLUGIN = "e2eplug"
SERVER = "plug"
PLUG_TOOL = tool_name(SERVER, "ro_probe")
CALL_RE = re.compile(r"CALL-PLUGIN:([A-Za-z0-9]+)")
ENV_RE = re.compile(r"ENV:[^:\s\"\\]*:[^\s\"\\]*")


def _user_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    return content if isinstance(content, str) else json.dumps(content)


def _last_user(body: dict[str, Any]) -> str:
    users = [m for m in body.get("messages") or [] if m.get("role") == "user"]
    return _user_text(users[-1]) if users else ""


def _call_on_demand(record: dict[str, Any]):
    """Call the plugin's tool (nonce from the prompt) only on turns whose prompt asks for it."""
    match = CALL_RE.search(_last_user(record["body"]))
    return script(*(((PLUG_TOOL, {"nonce": match.group(1)}),) if match else ()))(record)


def _first_request_of_turn(bodies: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    """The request that opened the turn whose prompt carries ``marker`` (no tool result yet)."""
    for body in bodies:
        msgs = body.get("messages") or []
        if marker in _last_user(body) and msgs and msgs[-1].get("role") == "user":
            return body
    raise AssertionError(f"no request opened the {marker!r} turn: {[_last_user(b)[-80:] for b in bodies]}")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _toggle_on(host, key: str) -> list[dict[str, Any]]:
    result = host.rpc.call("plugins.manage", {"action": "toggle", "key": key, "enable": True}, timeout=180)
    assert result.get("ok") and not result.get("unchanged"), result
    return ((result.get("activation") or {}).get("live_now") or {}).get("mcp_servers") or []


# 1. live activation in an open chat ------------------------------------------------------------


def test_plugin_enabled_mid_chat_is_usable_in_that_chat_with_an_unchanged_tools_array(tmp_path: Path) -> None:
    log = tmp_path / "plug_inbound.jsonl"
    with provider(_call_on_demand) as srv:
        eh = build_home(tmp_path, srv.base_url)
        write_portable_plugin(eh, PLUGIN, {SERVER: portable_stdio(log, eh.tag, MCPE2E_CANARY="LIVE-CANARY")})
        with tui_host(eh) as host:
            sid = host.new_session()
            assert FINAL in host.turn(sid, "TURN-ONE say hi")
            assert not log.exists(), "the plugin's server ran before the plugin was enabled"

            rows = _toggle_on(host, PLUGIN)
            assert [(r["name"], r["connected"]) for r in rows] == [(SERVER, True)], rows
            assert PLUG_TOOL in rows[0]["tools"], rows

            assert FINAL in host.turn(sid, "TURN-TWO CALL-PLUGIN:nonceopen")
            received = calls_received(log, "ro_probe")
            assert [c.get("arguments") for c in received] == [{"nonce": "nonceopen"}], (
                f"the open chat never reached the just-enabled plugin's server: {received}\n"
                f"model saw: {tool_results(srv)}")
            assert any("RO:LIVE-CANARY:nonceopen" in r for r in tool_results(srv)), tool_results(srv)

            bodies = srv.main_requests()
            before = _first_request_of_turn(bodies, "TURN-ONE")
            after = _first_request_of_turn(bodies, "TURN-TWO")
            assert _canon(after.get("tools")) == _canon(before.get("tools")), (
                "enabling a plugin mid-chat changed the open chat's model-facing tools[] "
                f"(prompt-cache prefix): added {sorted(tool_names(after) - tool_names(before))}, "
                f"removed {sorted(tool_names(before) - tool_names(after))}")
            assert before["messages"][0].get("role") == "system", before["messages"][0]
            assert _canon(after["messages"][0]) == _canon(before["messages"][0]), (
                "enabling a plugin mid-chat rewrote the open chat's system prompt (prompt-cache prefix)")
            # The open chat is told what just became usable (one-shot turn note), outside the prefix.
            assert f"[plugin installed: {PLUGIN}]" in _canon(after["messages"][1:]), (
                f"the open chat was never told the plugin went live; its turn ended with {after['messages'][-1]}")

            fresh = host.new_session()
            assert FINAL in host.turn(fresh, "TURN-THREE CALL-PLUGIN:noncefresh")
            assert [c.get("arguments") for c in calls_received(log, "ro_probe")][-1] == {"nonce": "noncefresh"}
            assert any("RO:LIVE-CANARY:noncefresh" in r for r in tool_results(srv)), tool_results(srv)


# 2. #119751 resource-only portable server -----------------------------------------------------


def test_resource_only_plugin_activated_live_is_reported_connected(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    log = tmp_path / "res_inbound.jsonl"
    with provider(script()) as srv:
        eh = build_home(tmp_path, srv.base_url)
        write_portable_plugin(eh, PLUGIN, {SERVER: portable_stdio(log, eh.tag, MCPE2E_RESOURCE_ONLY="1")})
        with tui_host(eh) as host:
            host.new_session()
            rows = _toggle_on(host, PLUGIN)
            methods = [m.get("method") for m in inbound(log) if isinstance(m, dict)]
            # Guard: the server really completed the MCP handshake with this host.
            assert "initialize" in methods and "notifications/initialized" in methods, methods
            assert [r["name"] for r in rows] == [SERVER], rows
            with known_gate(KNOWN, request.node.name, raises=KnownSymptom):
                symptom(rows[0]["connected"] is True and not rows[0].get("error"),
                        f"a resource-only MCP server that completed the handshake is reported as failed: {rows[0]}")


# 3. #120526 ${VAR} in a portable mcp.json env (native config.yaml is the control) ---------------


@pytest.fixture(scope="module")
def env_echo_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One ``hermes chat -q`` turn calling ``env_echo`` on a portable-plugin server AND a native
    ``mcp_servers`` server, each told to echo a var whose value is ``${VAR}`` from the profile .env.
    Returns ``{server: tool result, "output": the process's stdout+stderr}``."""
    root = tmp_path_factory.mktemp("env_interp")
    calls = ((tool_name(SERVER, "env_echo"), {}), (tool_name("nat", "env_echo"), {}))
    with provider(script(*calls)) as srv:
        eh = build_home(root, srv.base_url, extra={"plugins": {"enabled": [PLUGIN]}})
        eh.update_config(lambda cfg: cfg["mcp_servers"].update(nat=stdio_server(
            "nat", root / "nat.jsonl", eh.tag, MCPE2E_ECHO_ENV="E2E_NATIVE_KEY",
            E2E_NATIVE_KEY="${E2E_NATIVE_KEY}")))
        with open(eh.hermes_home / ".env", "a", encoding="utf-8") as fh:
            fh.write("\nE2E_PORTABLE_KEY=portable-dotenv-value\nE2E_NATIVE_KEY=native-dotenv-value\n")
        write_portable_plugin(eh, PLUGIN, {SERVER: portable_stdio(
            root / "plug.jsonl", eh.tag, MCPE2E_ECHO_ENV="E2E_PORTABLE_KEY",
            E2E_PORTABLE_KEY="${E2E_PORTABLE_KEY}")})
        try:
            proc = run_chat_q(eh, "Echo both env vars.")
        finally:
            reap_tagged(eh)
        assert proc.returncode == 0 and FINAL in proc.stdout, (proc.returncode, proc.stdout[-800:], proc.stderr[-2000:])
        results = tool_results(srv)
    by_server = {srv_name: next((r for r in results if f"source=\"{tool_name(srv_name, 'env_echo')}\"" in r), "")
                 for srv_name in (SERVER, "nat")}
    assert all("ENV:NO-CANARY:" in r for r in by_server.values()), f"an env_echo call did not run: {results}"
    return {**by_server, "output": proc.stdout + proc.stderr}


def _env_line(result: str) -> str:
    """The server's ``ENV:<canary>:<value>`` echo inside the tool result (what the server saw)."""
    match = ENV_RE.search(result)
    assert match, f"no ENV: echo in the tool result: {result[-300:]}"
    return match.group(0)


def test_native_mcp_env_placeholder_is_interpolated(env_echo_results: dict[str, str]) -> None:
    echoed = _env_line(env_echo_results["nat"])
    assert echoed == "ENV:NO-CANARY:native-dotenv-value", (
        f"native mcp_servers env ${{VAR}} did not reach the server with the .env value: {echoed}")


def test_portable_mcp_env_placeholder_is_interpolated(env_echo_results: dict[str, str],
                                                     request: pytest.FixtureRequest) -> None:
    echoed = _env_line(env_echo_results[SERVER])
    with known_gate(KNOWN, request.node.name, raises=KnownSymptom):
        symptom("${E2E_PORTABLE_KEY}" not in echoed,
                f"the portable plugin's server received the literal placeholder: {echoed}")
    assert echoed == "ENV:NO-CANARY:portable-dotenv-value", echoed


def test_enabled_portable_plugin_server_is_not_reported_as_an_unknown_toolset(env_echo_results: dict[str, str],
                                                                              request: pytest.FixtureRequest) -> None:
    """The enabled plugin's server worked in that very run (the fixture asserts its tool result), so a
    startup warning calling it an unknown toolset is a false alarm the user sees on every launch."""
    warned = [line for line in env_echo_results["output"].splitlines() if "Unknown toolsets" in line]
    with known_gate(KNOWN, request.node.name, raises=KnownSymptom):
        symptom(not warned, f"`hermes chat` warned about the enabled plugin's MCP server: {warned}")


# 4. #121078 same manifest name in two user plugin dirs ------------------------------------------


@pytest.fixture(scope="module")
def name_collision(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """``plugins/foo`` (the upgraded copy, v2) next to ``plugins/foo.bak-x`` (the old copy, v1), both
    declaring ``name: foo``; ``foo`` enabled. Reads ``hermes plugins list``, runs one turn calling the
    plugin's MCP tool, and collects the process logs."""
    root = tmp_path_factory.mktemp("name_collision")
    with provider(script((tool_name("srv", "ro_probe"), {"nonce": "dup"}))) as srv:
        eh = build_home(root, srv.base_url, extra={"plugins": {"enabled": ["foo"]}})
        live = write_portable_plugin(eh, "foo", {"srv": portable_stdio(
            root / "live.jsonl", eh.tag, MCPE2E_CANARY="CANARY-LIVE")}, name="foo", version="2.0.0")
        backup = write_portable_plugin(eh, "foo.bak-x", {"srv": portable_stdio(
            root / "backup.jsonl", eh.tag, MCPE2E_CANARY="CANARY-BACKUP")}, name="foo", version="1.0.0")
        listing = subprocess.run(hermes_argv("plugins", "list", "--plain", "--no-bundled"), cwd=eh.project,
                                 env=eh.env(), capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        assert listing.returncode == 0, listing.stderr[-2000:]
        try:
            turn = run_chat_q(eh, "Use the foo plugin.")
        finally:
            reap_tagged(eh)
        assert turn.returncode == 0 and FINAL in turn.stdout, (turn.stdout[-800:], turn.stderr[-2000:])
        results = tool_results(srv)
    logs = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in sorted((eh.hermes_home / "logs").glob("*.log")))
    return {"live": live, "backup": backup, "listing": listing.stdout + listing.stderr,
            "results": results, "logs": logs}


def test_same_name_collision_still_loads_exactly_one_copy(name_collision: dict[str, Any]) -> None:
    """Guard for the two KNOWN cells below: one ``foo`` row, and the turn really ran one copy's server."""
    rows = [line for line in name_collision["listing"].splitlines() if re.search(r"\bfoo\s*$", line)]
    assert len(rows) == 1 and "enabled" in rows[0], name_collision["listing"]
    hits = [r for r in name_collision["results"] if "RO:CANARY-" in r]
    assert len(hits) == 1, name_collision["results"]


def test_same_name_backup_dir_does_not_shadow_the_live_plugin(name_collision: dict[str, Any],
                                                              request: pytest.FixtureRequest) -> None:
    rows = [line for line in name_collision["listing"].splitlines() if re.search(r"\bfoo\s*$", line)]
    assert rows, name_collision["listing"]
    with known_gate(KNOWN, request.node.name, raises=KnownSymptom):
        symptom("2.0.0" in rows[0],
                f"`hermes plugins list` shows the backup copy instead of plugins/foo (v2.0.0): {rows}")
        symptom(any("RO:CANARY-LIVE:dup" in r for r in name_collision["results"]),
                f"a real turn ran the backup dir's MCP server, not plugins/foo's: {name_collision['results']}")


def test_same_name_plugin_collision_is_reported(name_collision: dict[str, Any], request: pytest.FixtureRequest) -> None:
    live, backup = str(name_collision["live"]), str(name_collision["backup"])
    surfaces = {"hermes plugins list": name_collision["listing"], "logs/*.log": name_collision["logs"]}
    named_both = [where for where, text in surfaces.items() if live in text and backup in text]
    with known_gate(KNOWN, request.node.name, raises=KnownSymptom):
        symptom(named_both, f"two user plugin dirs declare the same name 'foo' ({live} and {backup}) but no "
                            f"user-visible surface names both: {', '.join(surfaces)}")
