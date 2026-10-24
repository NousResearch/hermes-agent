"""LIVE Windows E2E for the Desktop-lifecycle cold-start skip (#76129/#76745).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). Exercises the REAL ownership predicate against REAL processes:

 1. A real child process self-registers in the REAL spawn ledger as a
    ``serve`` purpose with THIS process as its live spawner — ownership
    must hold, and ``_pause_windows_gateways_for_update`` must return None
    (no cold-start plan) even with an autostart artifact present.
 2. Kill the child (dead serve) — ownership drops, the pause plan carries
    ``cold_start_if_installed`` again.
 3. Venv-holder fallback rung: a real venv-python process with true
    ``serve`` argv is detected by the scan and, having a live parent,
    confers ownership; the token classifier rejects a ``kanban
    --preserve-cache`` lookalike (the #90778 class the salvage fixed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.live_process_fixtures import (
    HERMES_ENTRYPOINT_MARKER,
    SLEEPER_MARKER,
    hermes_backend_spawn_kwargs,
    sleeper_script_path,
)

pytestmark = pytest.mark.platforms("windows")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _wait_until(predicate, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _argv_visible(pid: int, marker: str) -> bool:
    """True once the process table shows *pid* with *marker* in its argv."""
    import psutil

    try:
        return marker in " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@pytest.fixture()
def sleeper():
    procs: list[subprocess.Popen] = []

    def _spawn(*tail: str) -> subprocess.Popen:
        p = subprocess.Popen(
            [sys.executable, sleeper_script_path(), *tail],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
        assert _wait_until(lambda: _argv_visible(p.pid, SLEEPER_MARKER)), "sleeper argv never visible"
        return p

    yield _spawn
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


@pytest.fixture()
def hermes_backend():
    """Spawn a sleeping process whose argv IS the Desktop's backend shape: ``-m hermes_cli.main <sub>``.

    ``sleeper`` deliberately cannot serve here. It spawns ``python sleeper.py <tail>``, and a Hermes
    tail behind an unrelated script is precisely the lookalike the identity matchers must refuse
    (#121156) — so a holder standing in for a REAL backend has to carry the real spawn shape, with no
    inert tail.

    ``-P`` plus the PYTHONPATH in ``hermes_backend_spawn_kwargs`` makes ``-m`` resolve to a sleeping
    stub rather than the repo's real entry point, while CWD stays at the project root so the Windows
    venv holder scan still keeps the process (it drops a ``-m hermes_cli.main`` holder whose command
    line and CWD are both outside the project root).
    """
    procs: list[subprocess.Popen] = []
    spawn_kwargs = hermes_backend_spawn_kwargs(PROJECT_ROOT)

    def _spawn(*subcommand: str) -> subprocess.Popen:
        p = subprocess.Popen(
            [sys.executable, "-P", "-m", "hermes_cli.main", *subcommand],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **spawn_kwargs,
        )
        procs.append(p)
        assert _wait_until(lambda: _argv_visible(p.pid, HERMES_ENTRYPOINT_MARKER)), "backend argv never visible"
        return p

    yield _spawn
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


def _write_ledger(entries: list[dict]) -> None:
    from hermes_cli import process_identity as pid_mod

    path = pid_mod._ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def _entry(proc: subprocess.Popen, purpose: str = "serve") -> dict:
    import psutil

    from hermes_cli import process_identity as pid_mod

    return {
        "install": pid_mod.install_id(None),
        "pid": proc.pid,
        "create_time": psutil.Process(proc.pid).create_time(),
        "purpose": purpose,
        "spawner_pid": os.getpid(),
        "spawner_create": psutil.Process(os.getpid()).create_time(),
    }


def test_live_supervised_serve_suppresses_cold_start(sleeper, monkeypatch, tmp_path):
    from hermes_cli import gateway as hermes_gateway
    from hermes_cli import gateway_windows
    from hermes_cli import update_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    serve = sleeper()
    _write_ledger([_entry(serve)])

    assert update_cmd._desktop_owns_gateway_lifecycle() is True

    # Autostart artifact present + no gateway running: WITHOUT ownership the
    # pause phase would plan a cold start; WITH it, no plan.
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    assert update_cmd._pause_windows_gateways_for_update() is None

    # Dead serve → ownership drops → plan returns.
    serve.kill()
    serve.wait()
    assert _wait_until(lambda: update_cmd._desktop_owns_gateway_lifecycle() is False)
    token = update_cmd._pause_windows_gateways_for_update()
    assert token is not None and token.get("cold_start_if_installed") is True


def test_holder_scan_fallback_respects_token_classifier(hermes_backend, monkeypatch, tmp_path):
    from hermes_cli import update_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    _write_ledger([])  # force the fallback rung

    # Real process whose argv carries genuine serve shape, visible to psutil. It must run from the
    # Desktop's own ``-m hermes_cli.main`` entry: a Hermes tail behind an unrelated script is the
    # lookalike #121156 refuses, so it would not stand in for a backend at all.
    serve_like = hermes_backend("serve")
    # Lookalike from the #90778 class — a real Hermes argv, but the wrong SUBCOMMAND. Only the token
    # classifier separates it from the serve holder, which is what this test exists to pin.
    kanban_like = hermes_backend("kanban", "--preserve-cache")

    import psutil

    # Snapshot both holder rows while both processes are alive. Building the
    # kanban row later (after the serve kill) would re-read the dead serve pid,
    # raise NoSuchProcess and hand the fallback an empty scan, which returns
    # False without ever reaching the token classifier.
    serve_row, kanban_row = (
        (p.pid, psutil.Process(p.pid).name(), " ".join(psutil.Process(p.pid).cmdline()))
        for p in (serve_like, kanban_like)
    )
    assert "--preserve-cache" in kanban_row[2]

    monkeypatch.setattr(
        "hermes_cli.main._detect_venv_python_processes", lambda: [serve_row, kanban_row]
    )
    # serve-shaped holder with a live parent (us) → owns
    assert update_cmd._desktop_owns_gateway_lifecycle() is True

    # Only the (still live) kanban lookalike left → classifier rejects → does not own
    serve_like.kill()
    serve_like.wait()
    assert kanban_like.poll() is None
    monkeypatch.setattr(
        "hermes_cli.main._detect_venv_python_processes", lambda: [kanban_row]
    )
    assert update_cmd._desktop_owns_gateway_lifecycle() is False
