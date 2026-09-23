"""ACP session construction runs off the event loop.

``session/new`` / ``session/load`` / ``session/resume`` / ``session/fork`` build a
full ``AIAgent`` (config load, memory-provider import, SessionDB). Done inline in
the coroutine, that build froze the loop serving every JSON-RPC request — the
host saw a server that answered ``initialize`` and then nothing (#58083).
"""

import asyncio
import threading
import time

import pytest

from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager

BUILD_SECONDS = 0.5


def _slow_factory():
    time.sleep(BUILD_SECONDS)  # stands in for the memory-provider import / AIAgent init
    from types import SimpleNamespace

    return SimpleNamespace(model="m", session_id="s", enabled_toolsets=[], disabled_toolsets=[],
                           _pending_toolsets=None)


@pytest.mark.asyncio
async def test_new_session_keeps_the_event_loop_free():
    """A concurrent coroutine keeps ticking while the agent is built."""
    server = HermesACPAgent(session_manager=SessionManager(agent_factory=_slow_factory))
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            await asyncio.sleep(0.02)

    task = asyncio.ensure_future(ticker())
    resp = await server.new_session(cwd="/tmp")
    done.set()
    await task
    assert resp.session_id
    assert ticks >= 5, f"event loop was blocked during session construction (ticks={ticks})"


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [
    lambda s: s.prompt([TextContentBlock(type="text", text="hi")], "gone"),
    lambda s: s.cancel("gone"),
    lambda s: s.set_session_model("m", "gone"),
    lambda s: s.set_session_mode("ask", "gone"),
    lambda s: s.set_config_option("edit_approval_policy", "gone", "ask"),
])
async def test_handlers_restore_unknown_sessions_off_the_loop(call):
    """``get_session`` on a not-in-memory id restores from the DB (full agent build) and
    waits on the restore lock; the per-session handlers must not do that on the loop."""
    manager = SessionManager(agent_factory=_slow_factory)

    def slow_restore(session_id, toolsets=None):
        time.sleep(BUILD_SECONDS)
        return None

    manager._restore = slow_restore
    server = HermesACPAgent(session_manager=manager)
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            await asyncio.sleep(0.02)

    task = asyncio.ensure_future(ticker())
    await call(server)
    done.set()
    await task
    assert ticks >= 5, f"event loop was blocked during the session restore (ticks={ticks})"


def test_concurrent_restores_of_one_session_build_a_single_agent():
    """Off-loop restores can now overlap: two ``get_session`` calls for the same
    not-in-memory id must share one DB restore, not construct two agents."""
    from types import SimpleNamespace

    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="m"))
    restores = []

    def slow_restore(session_id, toolsets=None):
        restores.append(session_id)
        time.sleep(0.2)
        return manager._install_state(session_id, manager._agent_factory(), "/tmp", "m", [], persist=False)

    manager._restore = slow_restore
    results = []
    threads = [threading.Thread(target=lambda: results.append(manager.get_session("sid-1"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(results) == 2 and results[0] is results[1] is not None
    assert len(restores) == 1, f"restore ran {len(restores)} agent builds for one session id"


def test_import_memory_provider_module_imports_without_constructing(tmp_path, monkeypatch):
    """The ACP startup warm-up (Windows main-thread pre-import, #58083) imports the
    configured provider's module plus the native stack it defers (hindsight imports numpy
    only in ``is_available()``), and nothing more: no provider instance, no register()."""
    import sys

    from plugins import memory as memory_plugins


    provider = tmp_path / "plugins" / "warmprov"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text(
        "import sys\nsys.modules['_warmprov_marker'] = True\n"
        "from agent.memory_provider import MemoryProvider\n"
        "def register(ctx):\n    sys.modules['_warmprov_registered'] = True\n",
        encoding="utf-8",
    )
    native = tmp_path / "plugins" / "_warm_native.py"
    native.write_text("import sys\nsys.modules['_warm_native_marker'] = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "plugins"))
    monkeypatch.setattr(memory_plugins, "_NATIVE_WARM_IMPORTS", ("_warm_native",), raising=False)
    sys.modules.pop("_warm_native_marker", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(memory_plugins, "_external_source_dirs", lambda: [tmp_path / "plugins"])
    sys.modules.pop("_warmprov_marker", None)
    sys.modules.pop("_warmprov_registered", None)

    assert memory_plugins.import_memory_provider_module("warmprov") is True
    assert sys.modules.get("_warmprov_marker") is True
    assert sys.modules.get("_warm_native_marker") is True
    assert "_warmprov_registered" not in sys.modules
    assert memory_plugins.import_memory_provider_module("no-such-provider") is False
