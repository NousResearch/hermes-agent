from __future__ import annotations

import threading
import time


def test_concurrent_boots_spawn_one_router(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import hermes_cli.local_runtime.bootstrap as bootstrap
    import hermes_cli.local_runtime.endpoint as endpoint
    import hermes_cli.local_runtime.supervisor as supervisor_mod

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [tmp_path / "m.gguf"])
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_tags", lambda: ["b1"])
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.default_tag", lambda: "b1")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda tag, backend: tmp_path,
    )
    monkeypatch.setattr(bootstrap, "_generate_presets", lambda mdir, path: None)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda sup: None)

    started = []
    in_start = threading.Event()

    class FakeSupervisor:
        def __init__(self, *args, **kwargs):
            self.base_url = "http://127.0.0.1:1/v1"

        def start(self):
            started.append(self)
            in_start.set()
            time.sleep(0.2)

    monkeypatch.setattr(supervisor_mod, "LlamaServerSupervisor", FakeSupervisor)

    cfg = {"local_runtime": {"enabled": True, "backend": "cpu"}}
    results = []
    first = threading.Thread(target=lambda: results.append(bootstrap.ensure_local_runtime(cfg)))
    first.start()
    assert in_start.wait(5)
    second = threading.Thread(target=lambda: results.append(bootstrap.ensure_local_runtime(cfg)))
    second.start()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(started) == 1
    assert results[0] is results[1] is started[0]
