"""Per-session cwd scoping for stdio MCP transports.

``hermes serve`` discovers MCP servers once — from ``tui_gateway/ws.py``, in
whatever directory the gateway was launched from — and ``tools.mcp_tool``
keyed every connection by server name alone. A stdio MCP child inherits its
parent's cwd, so *every* Desktop session shared one process sitting in the
launch directory. A project-sensitive server (Hindsight derives its memory
bank from ``${PWD}``) therefore answered the WRXN session with the launch
directory's project. Fixing the per-session ``_SlashWorker`` is not enough:
normal chat tool calls go through the parent process's MCP registry, which is
what these tests drive.

The probe server is a REAL stdio MCP server that reports its own startup cwd
and PID, so nothing here can pass against a mocked transport: if the transport
is not actually spawned in the session's project, the assertion fails.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent.runtime_cwd import clear_session_cwd, set_session_cwd  # noqa: E402

SERVER_NAME = "cwdprobe"

# Reports the cwd the process was SPAWNED in (captured at import, before any
# code could chdir) plus its PID, so a test can prove both "right directory"
# and "one stable process per directory".
_PROBE_SERVER = '''\
import os

from mcp.server.fastmcp import FastMCP

_START_CWD = os.path.realpath(os.getcwd())
_server = FastMCP("cwdprobe")


@_server.tool()
def whereami() -> str:
    """Report this MCP server process's startup cwd and pid."""
    return f"{_START_CWD}|{os.getpid()}"


_server.run()
'''

# Built once per test FILE (pytest runs each file in its own interpreter, see
# tests/conftest.py). Sharing the script path, the project dirs AND the fake
# HERMES_HOME across the tests in this module keeps the transport scope key
# stable, so the module spawns a handful of MCP children instead of a fresh
# pair per test.
_root: Path | None = None


def _scope_root(tmp_path_factory) -> Path:
    global _root
    if _root is None:
        root = tmp_path_factory.mktemp("mcp_session_scope", numbered=False)
        (root / "probe_server.py").write_text(_PROBE_SERVER, encoding="utf-8")
        (root / "project-a").mkdir()
        (root / "project-b").mkdir()
        (root / "hermes-home").mkdir()
        _root = root
    return _root


@pytest.fixture(scope="module", autouse=True)
def _reap_mcp_children():
    """Kill every stdio child this module spawned, in every scope."""
    yield
    from tools import mcp_tool

    mcp_tool.shutdown_mcp_servers()
    mcp_tool._kill_orphaned_mcp_children(include_active=True)


@pytest.fixture
def probe(tmp_path_factory, monkeypatch):
    """Connect the probe MCP server once and expose the two project dirs."""
    from tools import mcp_tool

    root = _scope_root(tmp_path_factory)
    # conftest's hermetic fixture points HERMES_HOME at a per-test tmp dir;
    # the profile home is part of the transport scope key, so pin it here or
    # every test would land in a different scope and respawn its servers.
    monkeypatch.setenv("HERMES_HOME", str(root / "hermes-home"))

    # Idempotent for an already-connected name: only the first test pays the
    # connect cost, the rest reuse the same registry entry.
    mcp_tool.register_mcp_servers({
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(root / "probe_server.py")],
            "connect_timeout": 60,
            "timeout": 60,
        }
    })
    assert SERVER_NAME in mcp_tool._servers, (
        "probe MCP server failed to connect: "
        f"{mcp_tool._server_connect_errors.get(SERVER_NAME)}"
    )
    try:
        yield type(
            "Probe", (), {
                "a": str(root / "project-a"),
                "b": str(root / "project-b"),
            },
        )
    finally:
        clear_session_cwd()


def _whereami() -> tuple[str, int]:
    """Call the probe tool through the real MCP handler; return (cwd, pid)."""
    from tools import mcp_tool

    handler = mcp_tool._make_tool_handler(SERVER_NAME, "whereami", 60)
    payload = json.loads(handler({}))
    assert "error" not in payload, payload
    cwd, pid = str(payload["result"]).split("|")
    return cwd, int(pid)


def _whereami_in(cwd: str) -> tuple[str, int]:
    set_session_cwd(cwd)
    try:
        return _whereami()
    finally:
        clear_session_cwd()


class TestStdioScopedBySessionCwd:
    def test_each_session_runs_the_server_in_its_own_project(self, probe):
        """Scenarios 1, 3 and 4: right cwd per session, stable process, no leak."""
        cwd_a1, pid_a1 = _whereami_in(probe.a)
        cwd_b, pid_b = _whereami_in(probe.b)
        # Called again AFTER session B ran: proves the transport is selected
        # per call from the session cwd, not last-writer-wins.
        cwd_a2, pid_a2 = _whereami_in(probe.a)

        assert cwd_a1 == probe.a
        assert cwd_b == probe.b
        assert cwd_a2 == probe.a
        assert pid_a1 != pid_b, "both projects shared one stdio process"
        # Repeated calls in one project must reuse the process — a respawn per
        # call would make every MCP call pay a cold start.
        assert pid_a1 == pid_a2

    def test_concurrent_sessions_do_not_share_a_process(self, probe):
        """Scenario 2: two live sessions calling at the same time stay isolated."""
        results: dict[str, tuple[str, int]] = {}
        errors: list[BaseException] = []
        start = threading.Barrier(2)

        def _run(label: str, cwd: str) -> None:
            try:
                start.wait(timeout=30)
                results[label] = _whereami_in(cwd)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                errors.append(exc)

        threads = [
            threading.Thread(target=_run, args=("a", probe.a), daemon=True),
            threading.Thread(target=_run, args=("b", probe.b), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, errors
        assert results["a"][0] == probe.a
        assert results["b"][0] == probe.b
        assert results["a"][1] != results["b"][1]

    def test_url_transport_stays_process_global(self, probe):
        """Scenario 5: HTTP transports have no cwd — they must not be cloned."""
        from tools import mcp_tool

        url_server = mcp_tool.MCPServerTask("fake-url")
        url_server._config = {"url": "https://example.invalid/mcp"}
        url_server.session = object()
        with mcp_tool._lock:
            mcp_tool._servers["fake-url"] = url_server
        try:
            set_session_cwd(probe.a)
            from_a = mcp_tool._get_connected_server_for_call("fake-url")
            set_session_cwd(probe.b)
            from_b = mcp_tool._get_connected_server_for_call("fake-url")
        finally:
            clear_session_cwd()
            with mcp_tool._lock:
                mcp_tool._servers.pop("fake-url", None)

        assert from_a is url_server
        assert from_b is url_server
        scoped = getattr(mcp_tool, "_scoped_servers", {})
        assert not [k for k in scoped if "fake-url" in str(k)], (
            "a URL transport was cloned per session cwd"
        )


class TestCwdChangeRebindsNextCall:
    def test_next_call_after_cwd_change_uses_the_new_project(self, probe):
        """Task 7: session.cwd.set must not need a schema change to take effect.

        The handler picks its transport from ``resolve_agent_cwd()`` at call
        time, so rebinding the session's cwd is enough — and it must not drag
        a sibling session along with it.
        """
        sibling_seen: list[str] = []
        sibling_bound = threading.Event()
        rebound = threading.Event()

        def _sibling() -> None:
            set_session_cwd(probe.a)
            sibling_seen.append(_whereami()[0])
            sibling_bound.set()
            rebound.wait(timeout=60)
            sibling_seen.append(_whereami()[0])

        thread = threading.Thread(target=_sibling, daemon=True)
        thread.start()
        assert sibling_bound.wait(timeout=120)

        # The changed session: bound to A, then re-bound to B (session.cwd.set).
        set_session_cwd(probe.a)
        assert _whereami()[0] == probe.a
        set_session_cwd(probe.b)
        after_change = _whereami()[0]
        clear_session_cwd()

        rebound.set()
        thread.join(timeout=120)

        assert after_change == probe.b
        # The sibling session never changed project, before or after.
        assert sibling_seen == [probe.a, probe.a]


class TestPrimaryIsTheBaselineScope:
    def test_primary_runs_in_the_resolved_agent_cwd_not_the_process_cwd(
        self, tmp_path_factory, monkeypatch,
    ):
        """The discovered instance must be the one a launch-dir session reuses.

        ``_canonical_call_cwd()`` resolves through ``resolve_agent_cwd()``, which
        prefers ``TERMINAL_CWD`` over the process cwd — and in the gateway the
        two differ (the systemd unit's cwd vs the configured ``terminal.cwd``).
        If the primary is spawned with the *process* cwd, no session ever
        matches it: every stdio server gets permanently duplicated, the
        pre-warmed discovery connection serves nobody, and the first call of
        every session pays a cold spawn inside the tool timeout.
        """
        from tools import mcp_tool

        root = _scope_root(tmp_path_factory)
        monkeypatch.setenv("HERMES_HOME", str(root / "hermes-home"))
        # The gateway case: TERMINAL_CWD is a real directory that is NOT the
        # process cwd (pytest runs from the repo root).
        monkeypatch.setenv("TERMINAL_CWD", str(root / "project-a"))
        name = "cwdprobe_primary"

        mcp_tool.register_mcp_servers({
            name: {
                "command": sys.executable,
                "args": [str(root / "probe_server.py")],
                "connect_timeout": 60,
                "timeout": 60,
            }
        })
        assert name in mcp_tool._servers, mcp_tool._server_connect_errors.get(name)

        # No session cwd bound: this is the plain launch-dir turn.
        handler = mcp_tool._make_tool_handler(name, "whereami", 60)
        payload = json.loads(handler({}))
        assert "error" not in payload, payload
        cwd, _pid = str(payload["result"]).split("|")

        assert cwd == str(root / "project-a"), (
            "the discovered primary was not spawned in the resolved agent cwd"
        )
        assert mcp_tool._call_scope(name) == (name, ""), (
            "a session sitting in the baseline directory was sent to a clone"
        )
        assert not [k for k in mcp_tool._scoped_servers if name in str(k)], (
            "the baseline session spawned a duplicate stdio child"
        )


class TestCloneNeverOwnsTheToolSchemas:
    def test_tools_list_changed_on_a_clone_does_not_touch_the_registry(self):
        """A clone must not re-register (and later globally de-register) tools.

        ``_register_tools_when_ready`` skips clones, but any server that emits
        ``notifications/tools/list_changed`` (mongodb-mcp-server does so right
        after initialize) drives ``_refresh_tools`` on EVERY instance. Without
        the same guard there, project B's transport republishes the shared tool
        schemas mid-conversation, and — worse — the clone ends up owning
        ``_registered_tool_names``, so when that one project's child dies the
        park/shutdown path de-registers the MCP tools out from under every other
        session.
        """
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from tools.mcp_tool import MCPServerTask
        from tools.registry import ToolRegistry, registry as real_registry

        clone = MCPServerTask(
            "srv", cwd="/tmp", scope_key="srv\0/home/x/.hermes\0/tmp",
        )
        clone._config = {"command": "test"}
        clone.session = MagicMock()
        clone.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[SimpleNamespace(
                name="dyn", description="d", inputSchema={"type": "object"},
            )]),
        )

        mock_registry = ToolRegistry()
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "tools.registry.registry", mock_registry,
        ):
            asyncio.run(clone._refresh_tools())

        assert clone._registered_tool_names == []
        assert not mock_registry.get_all_tool_names()
        assert "mcp__srv__dyn" not in real_registry.get_all_tool_names()
