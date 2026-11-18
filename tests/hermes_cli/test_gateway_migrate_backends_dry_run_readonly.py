"""``hermes gateway migrate --backends --apply --dry-run [--json]`` run for real changes nothing on the host.

A scratch account (``HOME``, a ``HERMES_HOME`` inside it, optionally a hand-written systemd user unit file
under ``~/.config/systemd/user``) and a real old backend: ``hermes serve`` started by hand in that home. The
CLI runs as a child process, as a user types it. The snapshot is taken right before the first dry run, once
the account is in the state a user's is: the backend is up and has answered one ``GET /api/status`` (a
running backend has already created the home's ``install_id``), and main's report has run twice (a fresh
home's first two commands create its layout and then the install lock). The dry runs must leave every file
under the account (bytes, mode, mtime), every directory (existence, mode) and the backend's process and
install id as they were. Directory mtimes are not compared: the running backend creates and removes its
own temporary files in its home while it runs, dry run or not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import psutil
import pytest

from tests.hermes_cli.live_gateway import REPO_ROOT, free_port, home_env, write_config
from tests.waiting import wait_until

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the scratch account is a POSIX home")

_UNIT = """[Unit]
Description=Hermes backend (written by hand from the docs)

[Service]
ExecStart=%h/.local/bin/hermes serve --host 127.0.0.1 --port 9119
Restart=always

[Install]
WantedBy=default.target
"""


# Not compared: main's passive update-check cache. ``hermes serve`` imports ``tui_gateway.server``, whose
# import starts ``hermes_cli.banner.prefetch_update_check()`` on a daemon thread; that thread writes
# ``<home>/source-checks/<install id>.json`` once its git/GitHub lookup returns, which can be after the
# snapshot. The old backend writes it on its own startup, dry run or not, so it says nothing about the dry run.
_NOT_COMPARED = (Path("hermes-home", "source-checks"),)


def _snapshot(root: Path) -> dict:
    """Every path under *root* except ``_NOT_COMPARED``: files (and links) by mode, size, mtime and content;
    directories by mode."""
    seen = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(rel == skip or skip in rel.parents for skip in _NOT_COMPARED):
            continue
        st = path.lstat()
        if path.is_dir() and not path.is_symlink():
            seen[str(path.relative_to(root))] = ("dir", st.st_mode)
        else:
            content = os.readlink(path) if path.is_symlink() else hashlib.sha256(path.read_bytes()).hexdigest()
            seen[str(path.relative_to(root))] = (st.st_mode, st.st_size, st.st_mtime_ns, content)
    return seen


def _status(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10) as resp:
            return json.loads(resp.read())
    except OSError:
        return None


def _hermes(env, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "migrate", "--backends", *argv],
                          cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=180, check=False)


@pytest.fixture(params=[False, True], ids=["serve-by-hand", "plus-a-user-unit-file"])
def scratch(tmp_path, request):
    user = tmp_path / "user"
    # Not the account's default home: a host's own unpinned dashboard unit (one that sets no HERMES_HOME)
    # counts as a backend of the default home, and this test must see only its own scratch backend.
    home = user / "hermes-home"
    home.mkdir(parents=True, mode=0o700)
    write_config(home)
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=scratch-user\n", encoding="utf-8")
    os.chmod(home / ".env", 0o600)
    if request.param:  # a unit file copied from the docs; systemctl may list it from the scratch config dir
        units = user / ".config" / "systemd" / "user"
        units.mkdir(parents=True)
        (units / "hermes-dashboard.service").write_text(_UNIT, encoding="utf-8")
    env = home_env(home, user)
    env["XDG_CONFIG_HOME"] = str(user / ".config")
    port = free_port()
    log_path = tmp_path / "serve.log"
    with log_path.open("w") as log:
        old = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1",
                                "--port", str(port)], cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: old.poll() is not None or _status(port) is not None, timeout=120,
                       what="the old backend answers /api/status")
            assert old.poll() is None, log_path.read_text()
            yield user, env, old, port, request.param
        finally:
            old.kill()
            old.wait(timeout=10)


def _settled(env) -> subprocess.CompletedProcess:
    """Run main's report twice (startup bookkeeping of a fresh home); the second run's result."""
    _hermes(env)
    return _hermes(env)


def test_the_dry_run_changes_no_file_unit_config_or_process(scratch):
    user, env, old, port, _with_unit = scratch
    report = _settled(env)
    assert report.returncode == 0, report.stdout + report.stderr
    assert f"--port {port}" in report.stdout
    born, answer = psutil.Process(old.pid).create_time(), _status(port)
    assert answer is not None and answer.get("install_id")
    before = _snapshot(user)

    # --address picks the scratch backend whatever else this host runs or has unit files for.
    mine = ["--apply", "--dry-run", "--address", f"127.0.0.1:{port}"]
    text, as_json = _hermes(env, *mine), _hermes(env, *mine, "--json")
    every = [_hermes(env, "--apply", "--dry-run"), _hermes(env, "--apply", "--dry-run", "--json")]

    assert _snapshot(user) == before
    assert old.poll() is None and psutil.Process(old.pid).create_time() == born
    assert (_status(port) or {}).get("install_id") == answer.get("install_id")
    assert text.stdout.startswith("Backend migration plan (dry run — nothing changed)"), text.stdout + text.stderr
    assert f"a process started by hand, pid {old.pid} on 127.0.0.1:{port}" in text.stdout
    plan = json.loads(as_json.stdout)
    assert plan["applied"] is False and plan["dry_run"] is True and plan["plan"]["old_backend"]["pid"] == old.pid
    # 0, or 3 where this host's own user systemd cannot be driven from a test child (a gateway preflight blocker)
    assert text.returncode == as_json.returncode and text.returncode in (0, 3)
    assert all(r.returncode in (0, 3, 6) for r in every), [r.stdout + r.stderr for r in every]


@pytest.mark.parametrize("scratch", [False], indirect=True, ids=["serve-by-hand"])
def test_apply_without_dry_run_exits_2_and_changes_nothing(scratch):
    user, env, old, port, _with_unit = scratch
    assert _settled(env).returncode == 0
    before = _snapshot(user)

    result = _hermes(env, "--apply")

    assert result.returncode == 2
    assert "--apply needs --dry-run in this version: it prints the migration plan and changes nothing." in result.stderr
    assert result.stdout == ""
    assert _snapshot(user) == before and old.poll() is None and _status(port) is not None
