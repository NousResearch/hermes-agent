"""Idle release must respect whole-turn activity, not just request completion."""

import json
import subprocess
import sys
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def ollama_server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path, None))
            payload = {"models": [{"name": "example:latest", "size_vram": 4096}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"done":true,"done_reason":"unload"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("enabled", [False, True])
def test_release_requires_opt_in_idle_and_pressure(tmp_path, ollama_server, enabled):
    from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy

    base_url, calls = ollama_server
    clock = [0.0]
    pressured = [False]
    probes = []

    def pressure():
        probes.append(True)
        return pressured[0]

    gate = ActivityGate(tmp_path, now=lambda: clock[0])
    policy = IdleOllamaPolicy(
        base_url=base_url,
        model="example:latest",
        enabled=enabled,
        gate=gate,
        pressure=pressure,
        idle_seconds=60,
    )
    with gate.turn():
        clock[0] = 120
        pressured[0] = True
        assert (
            not policy.tick()
        )  # generation and intervening tool execution stay protected
        assert calls == []
    assert not policy.tick()  # finishing a long turn does not count as time spent idle
    clock[0] += 61
    pressured[0] = False
    assert not policy.tick()
    assert calls == []
    pressured[0] = True
    assert policy.tick() is enabled
    if enabled:
        assert calls == [
            ("GET", "/api/ps", None),
            ("POST", "/api/generate", {"model": "example:latest", "keep_alive": 0}),
        ]
    else:
        assert calls == []
        assert probes == []


def test_overlapping_sessions_and_failure_keep_model_protected(tmp_path, ollama_server):
    from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy

    base_url, calls = ollama_server
    clock = [0.0]
    first = ActivityGate(tmp_path, now=lambda: clock[0])
    second = ActivityGate(tmp_path, now=lambda: clock[0])
    policy = IdleOllamaPolicy(
        base_url=base_url,
        model="example:latest",
        enabled=True,
        gate=first,
        pressure=lambda: True,
        idle_seconds=60,
    )
    with first.turn():
        with pytest.raises(RuntimeError, match="turn failed"):
            with second.turn():
                raise RuntimeError("turn failed")
        clock[0] = 120
        assert not policy.tick()
        assert calls == []
    clock[0] += 61
    assert policy.tick()


@pytest.mark.parametrize("reading", [None, False])
def test_unknown_or_absent_pressure_never_unloads(tmp_path, ollama_server, reading):
    from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy

    base_url, calls = ollama_server
    clock = [0.0]
    gate = ActivityGate(tmp_path, now=lambda: clock[0])
    policy = IdleOllamaPolicy(
        base_url=base_url,
        model="example:latest",
        enabled=True,
        gate=gate,
        pressure=lambda: reading,
        idle_seconds=60,
    )
    with gate.turn():
        pass
    clock[0] = 120
    assert not policy.tick()
    assert calls == []


def test_other_process_holds_activity_until_exit(tmp_path):
    from agent.ollama_idle_release import ActivityGate

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from agent.ollama_idle_release import ActivityGate
with ActivityGate(Path(sys.argv[1])).turn():
    print('ready', flush=True)
    sys.stdin.readline()
""",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        gate = ActivityGate(tmp_path)
        with gate.idle(0) as idle:
            assert idle is None
        child.communicate("finish\n", timeout=10)
        assert child.returncode == 0
        with gate.idle(0) as idle:
            assert idle is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


def test_idle_release_fences_admission_of_a_new_turn(tmp_path):
    from agent.ollama_idle_release import ActivityGate

    gate = ActivityGate(tmp_path)
    with gate.turn():
        pass
    started = threading.Event()
    admitted = threading.Event()

    def new_turn():
        started.set()
        with ActivityGate(tmp_path).turn():
            admitted.set()

    with gate.idle(0) as idle:
        assert idle is not None
        thread = threading.Thread(target=new_turn)
        thread.start()
        assert started.wait(5)
        assert not admitted.wait(0.1)
    thread.join(5)
    assert not thread.is_alive()
    assert admitted.is_set()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://ollama.com/v1",
        "http://192.0.2.1:11434/v1",
        "http://localhost:11434/proxy/v1",
        "http://localhost:11434/v1?route=other",
    ],
)
def test_nonlocal_or_ambiguous_routes_do_not_probe_or_release(tmp_path, base_url):
    from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy

    def unexpected_probe():
        pytest.fail("An unsupported route must not start pressure checks")

    policy = IdleOllamaPolicy(
        base_url=base_url,
        model="example:latest",
        enabled=True,
        gate=ActivityGate(tmp_path / "activity"),
        pressure=unexpected_probe,
    )
    assert not policy.tick()
    assert not (tmp_path / "activity").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_real_turn_entry_keeps_policy_idle_until_complete(
    tmp_path,
    monkeypatch,
    ollama_server,
    enabled,
):
    from agent import ollama_idle_runtime as runtime_module
    from agent.ollama_idle_release import ActivityGate
    from agent.turn_facade import TurnFacadeMixin

    base_url, calls = ollama_server
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"local_runtime:\n  ollama_idle_release: {str(enabled).lower()}\n",
        encoding="utf-8",
    )
    clock = [0.0]
    gate = ActivityGate(tmp_path / "activity", now=lambda: clock[0])
    runtime = runtime_module.IdleReleaseRuntime(start_thread=False)
    monkeypatch.setattr(runtime_module, "RUNTIME", runtime)
    monkeypatch.setattr(runtime_module, "ActivityGate", lambda directory: gate)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    probes = []

    def pressure():
        probes.append(True)
        return True

    monkeypatch.setattr(runtime_module, "gpu_memory_pressure", pressure)

    def turn_started(agent):
        clock[0] = 120
        runtime.poll()
        runtime.poll()
        assert calls == []
        raise RuntimeError("stop after admission")

    monkeypatch.setattr(
        "agent.background_review.cancel_background_review_for_live_turn", turn_started
    )
    agent = SimpleNamespace(base_url=base_url, model="example:latest", api_key="")
    with pytest.raises(RuntimeError, match="stop after admission"):
        TurnFacadeMixin.run_conversation(agent, "hello")
    runtime.poll()
    assert calls == []
    clock[0] += 61
    runtime.poll()
    assert bool(calls) is enabled
    if not enabled:
        assert probes == []
    # Turning off the owning profile prevents later hardware and provider work.
    (home / "config.yaml").write_text("local_runtime: {}\n", encoding="utf-8")
    previous_probes = len(probes)
    runtime.poll()
    assert len(probes) == previous_probes


def test_untagged_model_uses_ollama_latest_name(tmp_path, ollama_server):
    from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy

    base_url, calls = ollama_server
    gate = ActivityGate(tmp_path / "activity")
    with gate.turn():
        pass
    policy = IdleOllamaPolicy(
        base_url=base_url,
        model="example",
        enabled=True,
        gate=gate,
        pressure=lambda: True,
        idle_seconds=0,
    )
    assert policy.tick()
    assert calls[-1] == (
        "POST",
        "/api/generate",
        {"model": "example:latest", "keep_alive": 0},
    )
