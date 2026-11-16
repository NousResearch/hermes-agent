"""browser_vault fills land in the tab the session's browser_exec drives (#140196).

On a shared CDP browser several tabs can sit on the same login origin (another session's, a stale
one). ``focus_page`` used to attach to the FIRST same-origin tab whose form probe passed; it now
prefers the target browser_exec's harness daemon reports as its current tab.

The supervisor runs against a real local WebSocket speaking just enough CDP; the harness daemon's
``current_tab`` answer comes from a real AF_UNIX listener at the path the harness binds.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
import threading

import pytest

from tools.browser_supervisor import _SupervisorRegistry

_ORIGIN = "https://accounts.example"
_TABS = [
    {"targetId": "T-stale", "type": "page", "url": f"{_ORIGIN}/login?old"},
    {"targetId": "T-other", "type": "page", "url": f"{_ORIGIN}/login"},
    {"targetId": "T-own", "type": "page", "url": f"{_ORIGIN}/login"},
]


class _FakeCDP:
    """Three same-origin page targets; every tab passes the form probe. Records attaches."""

    def __init__(self) -> None:
        self.attached: list[str] = []
        self.formless: set[str] = set()  # sessions whose form probe is falsy (form not rendered yet)
        self.url = ""
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        assert ready.wait(5)

    def _serve(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        async def main():
            async with serve(self._handle, "127.0.0.1", 0) as server:
                self.url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/devtools/browser/x"
                ready.set()
                await asyncio.Event().wait()

        self.loop.run_until_complete(main())

    async def _handle(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            method, params = msg["method"], msg.get("params") or {}
            result: dict = {}
            if method == "Target.getTargets":
                result = {"targetInfos": _TABS}
            elif method == "Target.attachToTarget":
                self.attached.append(params["targetId"])
                result = {"sessionId": f"S-{params['targetId']}"}
            elif method == "Runtime.evaluate":
                result = {"result": {"type": "boolean", "value": msg.get("sessionId") not in self.formless}}
            await ws.send(json.dumps({"id": msg["id"], "result": result}))


@pytest.fixture
def supervisor():
    cdp, registry = _FakeCDP(), _SupervisorRegistry()
    sup = registry.get_or_start("t", cdp.url, start_timeout=5)
    cdp.attached.clear()  # the initial attach
    try:
        yield sup, cdp
    finally:
        registry.stop_all()


def test_fill_focuses_the_tab_browser_exec_drives_not_the_first_on_the_origin(supervisor):
    sup, cdp = supervisor
    sup.set_driven_target("T-own")
    out = sup.focus_page(_ORIGIN, accept="!!document.querySelector('input[type=password]')")
    assert out["ok"] is True
    assert cdp.attached == ["T-own"]  # the stale and the other session's tab are never attached


@pytest.mark.parametrize("origin", [_ORIGIN, ""])  # "" = a fill that names no site (save_login)
def test_driven_tab_whose_form_has_not_rendered_is_still_the_only_candidate(supervisor, origin):
    sup, cdp = supervisor
    cdp.formless.add("S-T-own")
    sup.set_driven_target("T-own")
    assert sup.focus_page(origin, accept="!!document.querySelector('input[type=password]')")["ok"] is True
    assert cdp.attached == ["T-own"]  # no fall-through to a same-origin tab whose form did render


def test_driven_tab_elsewhere_falls_back_to_the_origin_walk(supervisor):
    sup, cdp = supervisor
    sup.set_driven_target("T-closed-or-on-another-site")
    assert sup.focus_page(_ORIGIN, accept="true")["ok"] is True
    assert cdp.attached == ["T-stale"]


@pytest.mark.platforms("linux", "macos")
def test_harness_current_target_reads_the_daemons_attached_tab():
    """The daemon socket the harness binds for ``env`` answers ``current_tab`` with its targetId."""
    from hermes_constants import socket_safe_tmpdir
    from tools.browser_harness_dirs import harness_current_target

    runtime = tempfile.mkdtemp(prefix="bhct-", dir=socket_safe_tmpdir())
    env = {"BH_RUNTIME_DIR": runtime, "BH_RUNTIME_DIR_SHARED": "1", "BU_NAME": "s1"}
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # windows-footgun: ok — posix-only test
    server.bind(os.path.join(runtime, "bu-s1.sock"))
    server.listen(1)
    requests: list[dict] = []

    def answer():
        conn, _ = server.accept()
        with conn:
            requests.append(json.loads(conn.recv(4096)))
            conn.sendall(b'{"targetId": "T-own", "url": "https://accounts.example/login", "title": ""}\n')

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    try:
        assert harness_current_target(env) == "T-own"
        thread.join(5)
        assert requests == [{"meta": "current_tab"}]
        assert harness_current_target({**env, "BU_NAME": "no-daemon"}) is None
    finally:
        server.close()
        shutil.rmtree(runtime, ignore_errors=True)


@pytest.mark.platforms("linux", "macos")  # the fake CLI is a shell script
def test_browser_exec_records_its_daemons_tab_on_the_vault_supervisor(tmp_path, monkeypatch):
    import stat

    import tools.browser_use_cli as bu_cli
    from tools import browser_session as bs
    from tools import browser_supervisor

    recorded: list = []

    class _Sup:
        def set_driven_target(self, target_id):
            recorded.append(target_id)

    class _Registry:
        def get_or_start(self, task_id, cdp_url, **kw):
            return _Sup()

        def get(self, task_id):
            return _Sup() if task_id == "t" else None

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    monkeypatch.setattr(bs, "get_session_info",
                        lambda key: {"cdp_url": "ws://127.0.0.1:47000/devtools/browser/x", "features": {"local": True}})
    monkeypatch.setattr(bu_cli, "harness_current_target", lambda env: "T-own")
    cli = tmp_path / "browser-use"
    cli.write_text("#!/bin/sh\ncat > /dev/null\necho ok\n", encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])

    result = json.loads(bu_cli.browser_exec("print(1)", task_id="t"))
    assert result["success"] is True, result
    assert recorded == ["T-own"]


@pytest.mark.platforms("linux", "macos")  # the fake CLI is a shell script
def test_a_call_the_bot_desktop_fence_refuses_leaves_the_driven_tab_alone(tmp_path, monkeypatch):
    import stat

    import tools.browser_use_cli as bu_cli
    from tools import browser_session as bs
    from tools import browser_supervisor

    recorded: list = []

    class _Sup:
        def set_driven_target(self, target_id):
            recorded.append(target_id)

    class _Registry:
        def get_or_start(self, task_id, cdp_url, **kw):
            return _Sup()

        def get(self, task_id):
            return _Sup()

    route = bu_cli._route_exec_current

    def bot_desktop_route(env, *a):
        env, err, sidecar, generation = route(env, *a)
        env[bu_cli._BOT_DESKTOP_BROWSER_SENTINEL] = "local"
        return env, err, sidecar, generation

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    monkeypatch.setattr(bs, "get_session_info",
                        lambda key: {"cdp_url": "ws://127.0.0.1:47000/devtools/browser/x", "features": {"local": True}})
    monkeypatch.setattr(bs, "run_fenced", lambda info, fn: {"success": False, "code": "human_has_control", "error": "x"})
    monkeypatch.setattr(bu_cli, "_route_exec_current", bot_desktop_route)
    monkeypatch.setattr(bu_cli, "harness_current_target", lambda env: "T-sidecar")
    cli = tmp_path / "browser-use"
    cli.write_text("#!/bin/sh\ncat > /dev/null\necho ok\n", encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])

    assert json.loads(bu_cli.browser_exec("print(1)", task_id="t"))["code"] == "human_has_control"
    assert recorded == []  # the refused call never ran, so the supervisor keeps the tab it had
