"""
Class-level regression harness: a stale .env session token must not clobber
the desktop-injected token.

Bug class (user locked out of the desktop app): Electron spawns
``hermes serve`` with ``HERMES_DASHBOARD_SESSION_TOKEN=<fresh>`` +
``HERMES_DESKTOP=1``. A stale ``HERMES_DASHBOARD_SESSION_TOKEN`` left in
``~/.hermes/.env`` clobbers the injected value during
``load_hermes_dotenv(override=True)``, so ``web_server._SESSION_TOKEN``
resolves to the STALE token and the desktop's WebSocket handshake is rejected
(token_mismatch -> HTTP 403 / close 4401) — boot fails.

This harness is a REAL subprocess test by design (no mocking the seam): it
spawns a fresh ``hermes serve`` backend against a temp ``HERMES_HOME`` whose
``.env`` carries the stale token, then connects to ``/api/ws`` with each
candidate token and records which one the running server actually accepts.

  (a) HERMES_DESKTOP=1 -> the injected FRESH token must win (the regression).
  (b) no HERMES_DESKTOP  -> the .env token must win (documented .env
      precedence semantics preserved — the fix is scoped to desktop spawns).

On origin/main (before the env_loader fix) case (a) FAILS: the stale .env
token clobbers the injected one (RED). After the fix lands, (a) passes while
(b) must still pass (GREEN). Run this file against the env_loader lane's
worktree for the GREEN confirmation.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Distinct, obviously-fake tokens so probe results are unambiguous.
STALE_TOKEN = "stale-dotenv-token-00000000000000000000"
FRESH_TOKEN = "fresh-injected-token-00000000000000000000"

# Cold gateway import can take 15-30s+ on Windows (Defender .pyc scanning);
# the ready file is written only after the lifespan warmup completes, so this
# bounds the whole boot. Hard timeout — the test FAILS (never hangs) if the
# backend is not ready in time.
READY_TIMEOUT_SECONDS = 240.0
READY_POLL_INTERVAL = 0.25
# How long a single /api/ws probe waits for the accept/close verdict.
WS_PROBE_TIMEOUT_SECONDS = 15.0
# Tail of child stderr included in skip/fail diagnostics.
_STDERR_TAIL = 3000


def _repo_root() -> Path:
    """Root of the checkout under test (tests/hermes_cli/ -> repo root)."""
    return Path(__file__).resolve().parents[2]


def _build_child_env(
    hermes_home: Path, *, desktop: bool, injected_token: str, ready_file: Path
) -> dict[str, str]:
    """Env for the spawned backend: temp home + injected token (+ desktop flag)."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    # The desktop injects the fresh token unconditionally; HERMES_DESKTOP=1 is
    # what tells the env_loader the injection must survive the .env load.
    env["HERMES_DASHBOARD_SESSION_TOKEN"] = injected_token
    env["HERMES_DESKTOP_READY_FILE"] = str(ready_file)
    if desktop:
        env["HERMES_DESKTOP"] = "1"
    else:
        env.pop("HERMES_DESKTOP", None)
    return env


def _spawn_serve(
    hermes_home: Path, *, desktop: bool, ready_file: Path
) -> subprocess.Popen:
    """Spawn ``python -m hermes_cli.main serve`` from the checkout under test.

    Uses ``sys.executable`` (the interpreter running pytest) so the child
    shares the same dependency environment; cwd is the repo root so the child
    imports THIS checkout's ``hermes_cli``, not some installed copy.
    ``--port 0`` lets the OS assign a free port (discovered via the ready
    file, same mechanism the desktop app uses).
    """
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        env=_build_child_env(
            hermes_home,
            desktop=desktop,
            injected_token=FRESH_TOKEN,
            ready_file=ready_file,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_ready(proc: subprocess.Popen, ready_file: Path) -> int:
    """Block until the backend writes its ready file; return the bound port.

    Skips when the backend cannot be run in this environment at all (process
    exits before ready — missing deps, import failure, unsupported platform)
    and fails on a hard timeout so a broken boot never hangs the suite.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if ready_file.exists():
            try:
                payload = json.loads(ready_file.read_text(encoding="utf-8"))
                return int(payload["port"])
            except (ValueError, KeyError, OSError):
                pass  # still being written — keep polling
        if proc.poll() is not None:
            _out, err = proc.communicate(timeout=10)
            pytest.skip(
                "hermes serve is not runnable in this environment "
                f"(exit code {proc.returncode} before ready). "
                f"stderr tail: {err[-_STDERR_TAIL:]!r}"
            )
        time.sleep(READY_POLL_INTERVAL)
    proc.kill()
    _out, err = proc.communicate(timeout=10)
    pytest.fail(
        f"hermes serve did not become ready within {READY_TIMEOUT_SECONDS:.0f}s "
        f"(killed). stderr tail: {err[-_STDERR_TAIL:]!r}"
    )


@contextlib.contextmanager
def _serve_harness(hermes_home: Path, *, desktop: bool, ready_file: Path):
    """Spawn a real backend, yield its bound port, always tear it down."""
    proc = _spawn_serve(hermes_home, desktop=desktop, ready_file=ready_file)
    try:
        yield _wait_for_ready(proc, ready_file)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _ws_accepts_token(port: int, token: str) -> tuple[bool, str]:
    """Probe ``/api/ws?token=<token>``; return (accepted, detail).

    The backend's WS auth rejects a bad token before accepting the upgrade
    (uvicorn answers the upgrade request with HTTP 403), so a rejected probe
    surfaces as ``InvalidStatus``(403); some paths close with 4401 after the
    upgrade instead. An accepted probe either receives the server's
    ``gateway.ready`` frame or simply stays open until the probe timeout.
    """
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    from websockets.sync.client import connect

    url = f"ws://127.0.0.1:{port}/api/ws?token={token}"
    try:
        with connect(url, open_timeout=WS_PROBE_TIMEOUT_SECONDS) as ws:
            try:
                msg = ws.recv(timeout=WS_PROBE_TIMEOUT_SECONDS)
            except TimeoutError:
                return True, "connection open (no close within timeout)"
            return True, f"received frame {msg[:120]!r}"
    except InvalidStatus as exc:
        return False, f"http_status={exc.response.status_code}"
    except ConnectionClosed as exc:
        code = exc.rcvd.code if exc.rcvd is not None else exc.code
        return False, f"close_code={code}"
    except OSError as exc:
        return False, f"oserror={exc!r}"


def _make_temp_home(tmp_path: Path) -> Path:
    """Temp HERMES_HOME whose .env carries the stale session token."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        f"HERMES_DASHBOARD_SESSION_TOKEN={STALE_TOKEN}\n", encoding="utf-8"
    )
    return hermes_home


def test_stale_dotenv_token_does_not_clobber_injected_token(tmp_path):
    """
    The class regression: with HERMES_DESKTOP=1, the token injected into the
    environment must win over a stale token in <HERMES_HOME>/.env.

    RED on origin/main (the .env clobbers the injection during
    load_hermes_dotenv(override=True), so /api/ws rejects the fresh token and
    accepts the stale one — the desktop boot failure); GREEN after the
    env_loader fix.
    """
    hermes_home = _make_temp_home(tmp_path)
    ready_file = tmp_path / "ready.json"

    with _serve_harness(hermes_home, desktop=True, ready_file=ready_file) as port:
        fresh_ok, fresh_detail = _ws_accepts_token(port, FRESH_TOKEN)
        stale_ok, stale_detail = _ws_accepts_token(port, STALE_TOKEN)

    assert fresh_ok, (
        "FRESH (injected) token was REJECTED by the desktop-spawned backend — "
        "the stale .env token clobbered the injection (the bug class). "
        f"fresh detail: {fresh_detail}; stale accepted: {stale_ok} "
        f"({stale_detail})"
    )
    assert not stale_ok, (
        "STALE (.env) token was ACCEPTED by the desktop-spawned backend — "
        "the injected token lost to .env even with HERMES_DESKTOP=1. "
        f"stale detail: {stale_detail}"
    )


def test_non_desktop_spawn_keeps_dotenv_token(tmp_path):
    """
    Without HERMES_DESKTOP=1, .env precedence is unchanged: the stale .env
    token IS the server token. Guards against the fix being scoped too
    broadly (it must only protect desktop-spawned injections).
    """
    hermes_home = _make_temp_home(tmp_path)
    ready_file = tmp_path / "ready.json"

    with _serve_harness(hermes_home, desktop=False, ready_file=ready_file) as port:
        stale_ok, stale_detail = _ws_accepts_token(port, STALE_TOKEN)
        fresh_ok, fresh_detail = _ws_accepts_token(port, FRESH_TOKEN)

    assert stale_ok, (
        "STALE (.env) token was REJECTED by a non-desktop backend — the "
        "documented .env-precedence semantics regressed. "
        f"stale detail: {stale_detail}"
    )
    assert not fresh_ok, (
        "FRESH (injected) token was ACCEPTED by a non-desktop backend — "
        "injected tokens now beat .env outside desktop mode. "
        f"fresh detail: {fresh_detail}"
    )
