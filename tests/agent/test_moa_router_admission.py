"""Managed-router residency must bound MoA requests without blocking other endpoints (#134608)."""

import contextvars
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import urllib.request

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from hermes_cli.config import atomic_config_write


_ROUTER = r"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
limit = int(sys.argv[1])
lock = threading.Lock()
active = peak = rejected = 0
calls = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        with lock:
            self.reply(200, {'peak': peak, 'rejected': rejected, 'calls': list(calls)})
    def do_POST(self):
        global active, peak, rejected
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model = payload['model']
        with lock:
            if active >= limit:
                rejected += 1
                self.reply(400, {'error': {'message': f'model name={model} is not running'}})
                return
            active += 1
            peak = max(peak, active)
            calls.append(model)
        try:
            time.sleep(0.15)
            self.reply(200, {'text': f'advisor {model}'})
        finally:
            with lock: active -= 1
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""


@pytest.fixture
def routers(tmp_path, monkeypatch):
    processes = []
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def start(home, limit, managed=True):
        from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor
        home.mkdir(parents=True, exist_ok=True)
        script = home / "fixture_router.py"
        script.write_text(_ROUTER, encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-u", str(script), str(limit)],
                                stdout=subprocess.PIPE, text=True)
        processes.append(proc)
        port = int(proc.stdout.readline())
        token = set_hermes_home_override(home)
        try:
            supervisor = LlamaServerSupervisor(Path(sys.executable), home / "models",
                                               models_max=limit, port=port)
            supervisor.proc = proc
            if managed:
                supervisor._write_state()  # real PID/owner/incarnation receipt, no mocked psutil
            return supervisor.base_url
        finally:
            reset_hermes_home_override(token)
    yield start
    for proc in processes:
        proc.terminate()
        proc.wait(timeout=10)
        proc.stdout.close()


def _configure(home, independent, managed=None):
    atomic_config_write(home / "config.yaml", {
        "providers": {"parallel": {"base_url": independent, "api_key": "fixture-only-key",
                                   "default_model": "independent"},
                      **({"pinned": {"base_url": managed, "api_key": "fixture-only-key"}}
                         if managed else {})},
    })


def _transport(**kwargs):
    request = urllib.request.Request(kwargs["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": kwargs["model"]}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        text = json.loads(response.read())["text"]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _stats(base):
    with urllib.request.urlopen(base + "/stats", timeout=5) as response:
        return json.loads(response.read())


def _wire(monkeypatch, call=_transport):
    from agent import moa_loop
    from agent.usage_pricing import CanonicalUsage
    monkeypatch.setattr(moa_loop, "call_llm", call)
    monkeypatch.setattr(moa_loop, "_reference_context_length", lambda *args: 32_000)
    monkeypatch.setattr(moa_loop, "_price_reference_response", lambda *args: (CanonicalUsage(), None, None, None))
    return moa_loop


@pytest.mark.parametrize("launch", ["default", "named"])
@pytest.mark.parametrize("cap", [1, 2])
def test_every_reference_is_admitted_in_its_own_profile(tmp_path, monkeypatch, routers, launch, cap):
    root = tmp_path / ".hermes"
    homes = {p: root / "profiles" / p for p in ("a", "b")}
    launch_home = root if launch == "default" else homes["b"]
    launch_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    managed = routers(root / "managed", cap)
    endpoints = {p: routers(tmp_path / f"independent-{p}", 8, managed=False) for p in homes}
    for profile, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        _configure(home, endpoints[profile], managed)
    observed = []

    def call(**kwargs):
        observed.append((kwargs["model"], str(get_hermes_home()), kwargs["base_url"]))
        return _transport(**kwargs)
    moa = _wire(monkeypatch, call)
    for step, profile in enumerate(("a", "b", "a")):
        token = set_hermes_home_override(homes[profile])
        try:
            slots = [{"provider": alias, "model": f"local-{step}-{i}"}
                     for i, alias in enumerate(("llamacpp", "llama.cpp", "llama-cpp"))]
            slots.append({"provider": "custom:pinned", "model": f"local-{step}-3"})
            slots += [{"provider": "custom:parallel", "model": f"other-{step}-{i}"} for i in range(2)]
            progress = []
            outputs = moa._run_references_parallel(slots, [{"role": "user", "content": "advise"}],
                progress_callback=lambda done, total, label: progress.append((done, total)))
            assert [row[1] for row in outputs] == [f"advisor {slot['model']}" for slot in slots]
            assert sorted(progress) == [(i, 6) for i in range(1, 7)]
            turn_calls = [(model, home, base) for model, home, base in observed
                          if model.startswith((f"local-{step}", f"other-{step}"))]
            assert all(home == str(homes[profile]) for _, home, _ in turn_calls)
            assert all(base == (managed if model.startswith("local-") else endpoints[profile])
                       for model, _, base in turn_calls)
        finally:
            reset_hermes_home_override(token)
    assert _stats(managed)["rejected"] == 0
    assert _stats(managed)["peak"] == cap
    assert all(_stats(base)["peak"] == 2 for base in endpoints.values())
    # A damaged owner receipt must never supply an admission limit for another process.
    from hermes_cli.local_runtime.endpoint import managed_model_admission
    from hermes_cli.local_runtime.recovery import read_state
    from hermes_cli.local_runtime.supervisor import state_path
    from utils import atomic_json_write
    assert managed_model_admission()["capacity"] == cap
    state = read_state()
    state["owner_start_time"] += 10000
    atomic_json_write(state_path(), state)
    assert managed_model_admission() is None


@pytest.mark.parametrize("launch", ["default", "named"])
def test_interrupt_cancels_waiting_slots_and_keeps_live_router_leases(tmp_path, monkeypatch, routers, launch):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.local_runtime.recovery import read_state
    from hermes_cli.local_runtime.supervisor import state_path
    from utils import atomic_json_write
    root = tmp_path / ".hermes"
    home = root / "profiles" / "a"
    launch_home = root if launch == "default" else root / "profiles" / "b"
    launch_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    routers(home, 1)
    independent = routers(tmp_path / "independent", 8, managed=False)
    _configure(home, independent)
    started, other_done, release, billed = (threading.Event() for _ in range(4))
    calls, late = [], []

    def call(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "first-0":
            started.set()
            assert release.wait(10)
        response = _transport(**kwargs)
        if model == "independent":
            other_done.set()
        return response

    def accounting(label, usage):
        late.append(label)
        billed.set()

    moa = _wire(monkeypatch, call)
    agent = SimpleNamespace(_interrupt_requested=False)
    token = set_hermes_home_override(home)
    try:
        # Older live receipts have no capacity field; use a conservative single slot.
        state = read_state()
        state.pop("models_max", None)
        atomic_json_write(state_path(), state)
        first_slots = [{"provider": "llamacpp", "model": f"first-{i}"} for i in range(3)]
        second_slots = [{"provider": "llamacpp", "model": f"second-{i}"} for i in range(3)]
        second_slots.append({"provider": "custom:parallel", "model": "independent"})
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(contextvars.copy_context().run, moa._run_references_parallel,
                first_slots, [], agent=agent, late_accounting_sink=accounting)
            try:
                assert started.wait(5)
                second = executor.submit(contextvars.copy_context().run, moa._run_references_parallel, second_slots, [])
                assert other_done.wait(5), "an unrelated endpoint was blocked behind local admission"
                agent._interrupt_requested = True
                interrupted = first.result(timeout=5)
                assert len(interrupted) == 3
                assert all("interrupted" in row[1].lower() for row in interrupted)
                assert calls == ["first-0", "independent"]
            finally:
                release.set()
            assert [row[1] for row in second.result(timeout=10)] == [f"advisor {s['model']}" for s in second_slots]
            assert billed.wait(5)
            assert late == ["llamacpp:first-0"]
            assert not any(name in calls for name in ("first-1", "first-2"))
    finally:
        release.set()
        reset_hermes_home_override(token)
