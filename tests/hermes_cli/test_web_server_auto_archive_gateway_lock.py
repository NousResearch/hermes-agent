"""Invariant: `hermes serve`'s opportunistic auto-archive never opens a WRITABLE
SessionDB for a profile whose gateway is live (#110405).

The gateway stand-in is a real subprocess: it takes the same ``flock(LOCK_EX)`` on
``gateway.lock`` that ``gateway.status`` uses, runs under a ``hermes gateway run``
command line and is recorded in ``gateway.pid``, so ``_check_gateway_running``
decides on kernel lock state plus live process identity rather than a patched
predicate.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hermes_cli.profiles import _check_gateway_running
from hermes_cli import web_server_sessions as wss

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX flock holder"
)

_HOLDER = (
    "import fcntl, sys, time\n"
    "handle = open(sys.argv[1], 'a+', encoding='utf-8')\n"
    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    "sys.stdout.write('locked\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(300)\n"
)


class _FakeDB:
    def __init__(self, archived):
        self._archived = archived

    def maybe_auto_archive(self, **kwargs):
        self._archived.append(kwargs)

    def close(self):
        pass


@pytest.fixture
def archive_probe(tmp_path, monkeypatch):
    """Route the sweep at an isolated profile home and record every open."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    opens: list[bool] = []
    archived: list[dict] = []

    monkeypatch.setattr(wss, "_session_db_path_for_profile", lambda profile: db_path)
    monkeypatch.setattr(wss, "_last_auto_archive_check", {})

    def _open(profile, *, read_only):
        opens.append(read_only)
        return _FakeDB(archived)

    monkeypatch.setattr(wss, "_open_session_db_for_profile", _open)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {"sessions": {"auto_archive": True, "min_interval_hours": 0}},
    )
    return tmp_path, opens, archived


@pytest.mark.spawns_gateway_lookalike  # a flock-holding stub this test reaps by PID
def test_serve_auto_archive_defers_to_a_live_gateway_for_the_profile(archive_probe):
    tmp_path, opens, archived = archive_probe
    # argv0 basename `hermes` + the `gateway run` subcommand is what
    # gateway.status's process-identity check reads off /proc for this PID.
    entrypoint = tmp_path / "hermes"
    entrypoint.write_text(_HOLDER, encoding="utf-8")
    lock_path = tmp_path / "gateway.lock"
    holder = subprocess.Popen(
        [sys.executable, str(entrypoint), str(lock_path), "gateway", "run"],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    stdout = holder.stdout
    assert stdout is not None
    try:
        assert stdout.readline().strip() == "locked"
        (tmp_path / "gateway.pid").write_text(str(holder.pid), encoding="utf-8")
        assert _check_gateway_running(tmp_path), "probe did not look like a live gateway"

        wss._maybe_auto_archive_for_profile(None)

        assert opens == [], "serve opened the store while the gateway owned it"
        assert archived == []
    finally:
        holder.terminate()  # by PID: the process this test spawned
        holder.wait(timeout=10)
        stdout.close()

    # Gateway gone: the serve-side sweep is the only archiver left and must run.
    deadline = time.monotonic() + 5
    while _check_gateway_running(tmp_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    wss._last_auto_archive_check.clear()

    wss._maybe_auto_archive_for_profile(None)

    assert opens == [False], "serve must still archive when no gateway owns the profile"
    assert len(archived) == 1
