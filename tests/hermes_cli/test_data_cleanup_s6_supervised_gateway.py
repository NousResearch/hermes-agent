"""A data wipe must not pause a gateway that s6 relaunches: it would take the profile back mid-wipe."""
from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import uninstall
from tests.hermes_cli.test_data_uninstall import layout  # pytest fixture


def test_data_wipe_refuses_an_s6_supervised_gateway_instead_of_pausing_it(layout, monkeypatch):  # health: allow F811 -- pytest injects the imported layout fixture by parameter name
    import asyncio
    import subprocess
    import sys
    from threading import Event, Thread
    from gateway.control_socket import GatewayControlServer, build_identify_payload
    from gateway.status import get_process_start_time

    home, _, data = layout
    # The launch environment the s6 run script gives the gateway (S6ServiceManager._render_run_script).
    for name in ("INVOCATION_ID", "HERMES_DESKTOP_MANAGED", "XPC_SERVICE_NAME", "LAUNCHD_SOCKET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    # A disposable child is the declared gateway process; its identify answer is the gateway's
    # own builder, evaluated in this launch environment. Nothing here touches a real gateway.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=home)
    loop = asyncio.new_event_loop()
    ready = Event()
    paused = Event()
    holders = []

    def identify():
        return {**build_identify_payload(), "pid": child.pid,
                "start_time": get_process_start_time(child.pid)}

    def drain():
        paused.set()
        child.terminate()
        child.wait(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        return {"pid": child.pid, "pausing": True, "drain_timeout": 0}

    server = GatewayControlServer(home, verb_handlers={"identify": identify, "pause-for-update": drain})

    def serve():
        asyncio.set_event_loop(loop)
        started = loop.run_until_complete(server.start())
        holders.append(started)
        ready.set()
        if started:
            loop.run_forever()
        loop.run_until_complete(server.stop())

    thread = Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=10) and holders == [True]
        try:
            uninstall.run_data_uninstall(SimpleNamespace(yes=True))
            refused = False
        except SystemExit:
            refused = True
        assert (refused, paused.is_set(), all(path.exists() for path in data)) == (True, False, True)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        if thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        if not thread.is_alive():
            loop.close()
    assert not thread.is_alive()
