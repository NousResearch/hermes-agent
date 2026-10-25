"""Historical frames stay old; completion runs once in a clean child."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("desktop", [False, True])
@pytest.mark.parametrize("resume", [None, {"resume_needed": True, "profiles": {"work": "old-pid"}}])
def test_historical_payload_maps_to_takeover_request_schema(tmp_path, desktop, resume):
    """Regression for the 0.21.3 rehearsal failure: the old post-swap payload has no
    ``desktop``/``assume_yes``/``windows_resume`` keys, and update_finish reads them
    with bare subscriptions — the completion died on KeyError('desktop') with the
    checkout already swapped (post_swap_7006.json on the rehearsal host)."""
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "updated checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    shutil.copy2(source / "hermes_cli/_old_updater.py", package / "_old_updater.py")
    (package / "_update_takeover.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "assert request['desktop'] == " + repr(desktop) + "\n"
        "assert request['assume_yes'] == True\n"
        "assert request['restart_update'] == False\n"
        "assert request['gateway_mode'] == True\n"
        "assert request['had_desktop_app_before_update'] == " + repr(desktop) + "\n"
        "assert request['windows_resume'] == " + repr(resume) + "\n"
        "assert request['receipt']['update_id'] == 'old-correlation'\n"
        "assert request['plan']['install_method'] == 'git'\n"
        "Path(sys.argv[2]).write_text(json.dumps({'resume_handled': True}), encoding='utf-8')\n"
        "raise SystemExit(0)\n", encoding="utf-8",
    )
    home = tmp_path / "isolated home 日本 café"
    home.mkdir()
    payload = {
        "swap": "git", "branch": "main", "pre_pull_sha": "10433003", "is_fork": True,
        "gateway_mode": True, "had_desktop_app_before_update": desktop,
        "pre_update_snapshot_id": "20260918-185530-pre-update", "pre_update_version": "0.21.3",
        "active_lazy_features": ["tool.dashboard"], "active_tool_dependencies": [],
        "plan": {"install_method": "git", "expected_sha": "old-sha"},
        "sibling_snapshots": {},
        "receipt": {"update_id": "old-correlation", "outcome": "running"},
    }
    if resume is not None:
        payload["windows_gateway_resume"] = resume
    # The retired updater reaches the takeover through the NEW tree's hand-off module; this
    # checkout is the whole tree the child sees (CI has no editable finder for the source).
    shutil.copy2(source / "hermes_cli/update_handoff.py", package / "update_handoff.py")
    # write_handoff resolves the home through hermes_constants; the child tree is the whole
    # sys.path (CI has no editable finder), so give it the names the hand-off reads (the
    # payload lands beside the receipts in the ROOT home, which is HERMES_HOME here).
    (root / "hermes_constants.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def get_hermes_home():\n    return Path(os.environ['HERMES_HOME'])\n"
        "get_default_hermes_root = get_hermes_home\n", encoding="utf-8",
    )
    program = root / "historical.py"
    program.write_text(
        "import os\n"
        "from hermes_cli.update_handoff import continue_update_in_fresh_interpreter\n"
        "code = continue_update_in_fresh_interpreter(\n"
        f"    {payload!r}, argv_tail=['update', '--yes', '--gateway', '--branch', 'main'])\n"
        "raise SystemExit(code)\n", encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('HERMES_', 'PYTHON', 'UV_'))}
    env.update(HOME=str(home), HERMES_HOME=str(home))
    result = subprocess.run([sys.executable, "-B", str(program)], env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    handoff = json.loads((home / "logs/update_receipts").glob("post_swap_*.json").__next__().read_text())
    assert handoff["had_desktop_app_before_update"] is desktop
    assert "desktop" not in handoff and "assume_yes" not in handoff


@pytest.mark.live_system_guard_bypass
def test_shipped_post_swap_argv_enters_takeover_before_current_cli(tmp_path):
    """The 2026.9.21 updater starts HEAD as ``hermes update <flags> --post-swap FILE``.

    That command must reach the historical takeover before current launch preparation or
    argparse: both belong to the replacement updater and may require dependencies the old
    environment has not installed yet.
    """
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "updated checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "import hermes_bootstrap\nraise AssertionError('current CLI parsed a legacy continuation')\n",
        encoding="utf-8",
    )
    for relative in ("hermes_bootstrap.py", "hermes_cli/update_handoff.py", "hermes_cli/_old_updater.py"):
        shutil.copy2(source / relative, root / relative)
    (package / "_update_takeover.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "assert request['desktop'] is True\n"
        "assert request['assume_yes'] is True\n"
        "assert request['argv'][1:] == ['update', '--yes', '--no-gateway-restart', "
        "'--branch', 'main', '--post-swap', request['legacy_handoff']]\n"
        "Path(sys.argv[2]).write_text(json.dumps({'resume_handled': True}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    handoff = home / "post_swap_3185.json"
    payload = {
        "legacy_handoff": str(handoff),
        "gateway_mode": False,
        "had_desktop_app_before_update": True,
        "windows_gateway_resume": None,
        "plan": {"install_method": "git"},
        "receipt": {"update_id": "old-correlation", "outcome": "running"},
    }
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "hermes_cli.main", "update", "--yes",
         "--no-gateway-restart", "--branch", "main", "--post-swap", str(handoff)],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not handoff.exists()


@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("desktop", [None, False, True])
# A Windows console/pipe in the ANSI code page: the old updater's stdout encodes strictly in cp1252,
# and its hand-off banner runs before the child starts (review S1).
@pytest.mark.parametrize("stdio", [None, "cp1252:strict"])
def test_takeover_waits_propagates_status_and_never_reenters_old_code(tmp_path, status, encoding, desktop, stdio):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "updated checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    shutil.copy2(source / "hermes_cli/_old_updater.py", package / "_old_updater.py")
    # This is the process seam, not a counterfeit installer. Actual PM and
    # product construction are exercised separately against local packages.
    (package / "_update_takeover.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "assert 'old_only' not in sys.modules\n"
        "assert sys.flags.utf8_mode == 1\n"
        "assert 'PYTHONPATH' not in os.environ\n"
        f"assert request['desktop'] is {desktop!r}, request['desktop']\n"
        "assert request['windows_resume']['profiles'] == {'work': 'old-pid'}\n"
        "assert request['pre_update_snapshot_id'] == 'preserve-snapshot'\n"
        "assert request['gateway_mode'] == True\n"
        "assert request['pre_update_version'] == 'old-version'\n"
        "assert request['home'] == os.environ['HERMES_HOME']\n"
        "with Path(request['home'], 'runs').open('a') as stream: stream.write('child\\n')\n"
        f"Path(sys.argv[2]).write_text(json.dumps({{'resume_handled': True}}), encoding={encoding!r})\n"
        f"raise SystemExit({status})\n", encoding="utf-8",
    )
    home = tmp_path / "isolated home 日本 café"
    home.mkdir()
    program = root / "historical.py"
    program.write_text(
        "import atexit, json, os, sys, types\nfrom pathlib import Path\n"
        "sys.modules['old_only'] = types.ModuleType('old_only')\n"
        "from hermes_cli._old_updater import stop_for_relaunch\n"
        "_windows_gateway_resume = {'resume_needed': True, 'profiles': {'work': 'old-pid'}}\n"
        # The June updater reaches the takeover before it declares this local.
        f"if {desktop!r} is not None:\n    had_desktop_app_before_update = {desktop!r}\n"
        "pre_update_snapshot_id = 'preserve-snapshot'\n"
        "gateway_mode = True\npre_update_version = 'old-version'\n"
        "def cleanup():\n"
        "    assert not _windows_gateway_resume['resume_needed']\n"
        "    assert 'old_only' in sys.modules\n"
        "    Path(os.environ['HERMES_HOME'], 'cleanup').write_text('ran')\n"
        "atexit.register(cleanup)\n"
        "try:\n    stop_for_relaunch()\n"
        "finally:\n    stop_for_relaunch()\n"
        "raise AssertionError('old updater resumed')\n", encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('HERMES_', 'PYTHON', 'UV_'))}
    env.update(HOME=str(home), HERMES_HOME=str(home), PYTHONPATH="/not/the/new/source")
    if stdio:
        env["PYTHONIOENCODING"] = stdio
    result = subprocess.run([sys.executable, "-B", str(program)], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False)
    assert result.returncode == status, result.stdout + result.stderr
    assert (home / "runs").read_text() == "child\n"
    assert (home / "cleanup").read_text() == "ran"
    assert "run `hermes` again" not in result.stderr


# Only the copied shim and JSON probe run; this temp checkout has no real updater.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("post_pull", [False, True])
@pytest.mark.parametrize("module_name", ["update_cmd", "unrelated"])
def test_only_known_early_updater_restarts_with_original_arguments(tmp_path, post_pull, module_name):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "updated checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("_old_updater.py", "old_updater_deps.py"):
        shutil.copy2(source / "hermes_cli" / name, package / name)
    (package / "_update_takeover.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(request['home'], 'request.json').write_text(json.dumps(request))\n"
        "Path(sys.argv[2]).write_text('{}')\n", encoding="utf-8",
    )
    # Model the historical frame at 096826bf: capture precedes the assignment;
    # after pulling, the *same* frame reaches a dependency hook. The wrapper
    # must not make that post-pull invocation look like another early update.
    (package / f"{module_name}.py").write_text(
        "from hermes_cli.old_updater_deps import _capture_active_lazy_features, _update_node_dependencies\n"
        "def _cmd_update_impl(post_pull):\n"
        "    if not post_pull:\n        _capture_active_lazy_features()\n"
        "    pre_update_version = None\n"
        "    _update_node_dependencies()\n"
        "def cmd_update(post_pull):\n    _cmd_update_impl(post_pull)\n", encoding="utf-8",
    )
    home = tmp_path / "isolated home 日本 café"
    home.mkdir()
    argv = [str(root / "historical.py"), "--profile", "work profile", "update", "--yes",
            "--keep-stash", "--switch-branch", "--force", "--gateway"]
    Path(argv[0]).write_text(
        f"from hermes_cli.{module_name} import cmd_update\n"
        f"cmd_update({post_pull!r})\n"
        "raise AssertionError('old updater resumed')\n", encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home))
    result = subprocess.run([sys.executable, "-B", *argv], env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    request = json.loads((home / "request.json").read_text())
    assert request.get("restart_update", False) is (not post_pull and module_name == "update_cmd")
    assert request["argv"] == argv


@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("cleanup_status", [0, 9])
def test_atexit_recovers_only_stopped_serves_after_cached_update(tmp_path, acknowledged, cleanup_status):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "updated checkout"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("_old_updater.py", "old_updater_deps.py"):
        shutil.copy2(source / "hermes_cli" / name, package / name)
    (package / "_update_takeover.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "home = Path(request['home'])\n"
        "cleanup = 'stopped_serves' in request\n"
        "if cleanup:\n"
        "    assert set(request) == {'root', 'home', 'argv', 'stopped_serves'}\n"
        "    assert request['stopped_serves']['pending'] is True\n"
        "    assert request['stopped_serves']['entries'][0]['profile'] == 'work profile'\n"
        "with (home / 'runs').open('a') as stream:\n"
        "    stream.write('cleanup\\n' if cleanup else 'update\\n')\n"
        f"Path(sys.argv[2]).write_text(json.dumps({{'serves_handled': {acknowledged!r}}} if cleanup else {{}}))\n"
        f"raise SystemExit({cleanup_status} if cleanup else 7)\n", encoding="utf-8",
    )
    home = tmp_path / "isolated home 日本 café"
    home.mkdir()
    program = root / "historical.py"
    program.write_text(
        "import atexit, json, os\nfrom pathlib import Path\n"
        "from hermes_cli._old_updater import stop_for_relaunch\n"
        "from hermes_cli.old_updater_deps import _relaunch_stopped_serves\n"
        "token = {'pending': True, 'entries': [{'purpose': 'serve', 'profile': 'work profile',"
        " 'host': '127.0.0.1', 'port': 8119}]}\n"
        "def cleanup():\n"
        "    try:\n"
        "        _relaunch_stopped_serves(token)\n"
        "        if not token['pending']:\n            _relaunch_stopped_serves(token)\n"
        "    finally:\n"
        "        Path(os.environ['HERMES_HOME'], 'token.json').write_text(json.dumps(token))\n"
        "atexit.register(cleanup)\n"
        "stop_for_relaunch()\n"
        "raise AssertionError('old updater resumed')\n", encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home))
    result = subprocess.run([sys.executable, "-B", str(program)], env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 7, result.stdout + result.stderr
    assert (home / "runs").read_text() == "update\ncleanup\n"
    token = json.loads((home / "token.json").read_text())
    assert token["pending"] is (not acknowledged)
    assert "Exception ignored" not in result.stderr
    assert ("restart the affected" in result.stderr) is (not acknowledged or cleanup_status != 0)


@pytest.mark.parametrize("spawn_fails", [False, True])
@pytest.mark.parametrize("discarded", [False, True])
def test_serve_resume_child_reuses_respawn_without_updater_or_supervisors(tmp_path, spawn_fails, discarded):
    root = Path(__file__).resolve().parents[2]
    context, result_path = tmp_path / "request.json", tmp_path / "result.json"
    home = tmp_path / "home"
    home.mkdir()
    entries = [
        {"purpose": "serve", "profile": "default", "host": "127.0.0.1", "port": 8119},
        {"purpose": "dashboard", "profile": "ops team 日本", "host": "::1", "port": 8120},
    ]
    if discarded:
        entries += [
            dict(entries[0]),
            {"purpose": "serve", "profile": "desktop", "port": 0},
            {"purpose": "serve", "profile": "foreign", "port": 8121, "hermes_home": str(tmp_path / "foreign")},
            {"purpose": "gateway", "profile": "not-serve", "port": 8122},
            {"purpose": "serve", "profile": "boolean-port", "port": True},
            {"purpose": "serve", "profile": "invalid-port", "port": 65536},
        ]
    context.write_text(json.dumps({"root": str(root), "home": str(home),
                                  "stopped_serves": {"pending": True, "entries": entries}}))
    program = tmp_path / "probe.py"
    program.write_text(
        "import json, os, runpy, subprocess, sys, time\nfrom pathlib import Path\n"
        "calls = []\n"
        "time.sleep = lambda *_: None\n"
        "def forbidden(*args, **kwargs):\n    raise AssertionError('updater/process control ran')\n"
        "class _Proc:\n    def poll(self):\n        return None\n"
        "def spawn(command, **kwargs):\n"
        "    assert kwargs['start_new_session'] is True\n"
        "    assert kwargs['stdin'] == subprocess.DEVNULL\n"
        "    calls.append(command)\n"
        f"    if {spawn_fails!r}:\n        raise OSError('probe spawn failed')\n"
        "    return _Proc()\n"
        "subprocess.Popen = spawn\nsubprocess.run = os.system = os.kill = forbidden\n"
        f"sys.argv = [{str(root / 'hermes_cli/update_serve_resume.py')!r}, {str(context)!r}, {str(result_path)!r}]\n"
        "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "finally:\n"
        "    assert not any(name in sys.modules for name in ('hermes_cli.update_cmd', 'hermes_cli.update_receipt', 'hermes_cli.main'))\n"
        "    Path(os.environ['HERMES_HOME'], 'commands.json').write_text(json.dumps(calls))\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home))
    result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(program)],
                            env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == int(spawn_fails or discarded), result.stdout + result.stderr
    assert json.loads(result_path.read_text()) == {"serves_handled": True}
    commands = json.loads((home / "commands.json").read_text())
    assert commands == [
        [sys.executable, str(root / "hermes"), "serve", "--host", "127.0.0.1", "--port", "8119"],
        [sys.executable, str(root / "hermes"), "--profile", "ops team 日本", "dashboard",
         "--host", "::1", "--port", "8120", "--no-open"],
    ]
    assert ("probe spawn failed" in result.stdout) is spawn_fails


def test_serve_resume_child_leaves_token_unhandled_when_imports_fail(tmp_path):
    root = Path(__file__).resolve().parents[2]
    context, result_path = tmp_path / "request.json", tmp_path / "result.json"
    context.write_text(json.dumps({"root": str(tmp_path), "home": str(tmp_path),
                                  "stopped_serves": {"pending": True, "entries": []}}))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path))
    # -S and an empty checkout make application imports genuinely unavailable;
    # this is not a mocked exception or a path into the real update machinery.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(root / "hermes_cli/update_serve_resume.py"),
         str(context), str(result_path)], env=env, capture_output=True, text=True, timeout=30,
         check=False,
    )
    assert result.returncode == 1
    assert json.loads(result_path.read_text()) == {"serves_handled": False}
    assert "Stopped serve recovery failed" in result.stderr


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("entrypoint", ["_update_takeover", "update_serve_resume"])
def test_completed_serve_token_is_acknowledged_without_preparation(tmp_path, encoding, entrypoint):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "checkout 日本 café"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("_update_takeover.py", "update_serve_resume.py"):
        shutil.copy2(source / "hermes_cli" / name, package / name)
    # Only selection is supplied. The two real entrypoints must carry the
    # request without importing PM or launching any completed backend twice.
    (package / "_launchers.py").write_text(
        "import sys\nresolve_store_python = lambda root: sys.executable\n", encoding="utf-8")
    pm_package = root / "pm"
    pm_package.mkdir()
    (pm_package / "__init__.py").write_text("", encoding="utf-8")
    (pm_package / "environments.py").write_text(
        "import os\nactivation_environment = lambda root: dict(os.environ)\n", encoding="utf-8")
    context, result_path = tmp_path / "request.json", tmp_path / "result.json"
    context.write_text(json.dumps({"root": str(root), "stopped_serves": {"pending": False}},
                                  ensure_ascii=False), encoding=encoding)
    before = context.read_bytes()
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(package / f"{entrypoint}.py"),
         str(context), str(result_path)], cwd=tmp_path, capture_output=True,
        text=True, encoding="utf-8", timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(result_path.read_text(encoding="utf-8-sig")) == {"serves_handled": True}
    assert not result_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert context.read_bytes() == before


def test_bootstrap_lock_remains_live_without_application_dependencies(tmp_path):
    """A dependency-free (-I -S) takeover child still sees another live updater's claim.

    The holder is a separate, unrelated process: a claim naming the caller's own pid is
    adopted as a fresh attempt, so only a foreign live holder can prove liveness detection.
    """
    root = Path(__file__).resolve().parents[2]
    lock = tmp_path / "lock"
    prelude = (
        "import os, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from hermes_cli.update_lock import UpdateLock\n"
        f"lock = UpdateLock(path=Path({str(lock)!r}))\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c",
         prelude + "assert lock.acquire()\nprint(os.getpid(), flush=True)\n"
                   "sys.stdin.readline()\nlock.release()\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    )
    assert holder.stdout is not None
    try:
        holder_pid = int(holder.stdout.readline())
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c",
             prelude + "assert not lock.acquire(), 'live lock was stolen'\n"
                       f"assert lock.holder.pid == {holder_pid}, lock.holder\n"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=False,
        )
    finally:
        holder.communicate("\n", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert holder.returncode == 0
    assert not lock.exists()


SHA = "c" * 40


@pytest.mark.parametrize("git", [
    "missing",
    pytest.param("failing", marks=pytest.mark.platforms("posix")),  # POSIX shell stub
])
def test_takeover_arms_the_checkout_head_when_git_cannot_answer(tmp_path, monkeypatch, git):
    """The historical takeover arms the fleet restart AFTER the tree moved. A host whose git is off
    PATH (only PM's store copy) used to raise FileNotFoundError there, reporting a committed update
    as failed (F20); a git that runs but fails silently armed nothing (F25). Either way the record
    must still name the HEAD the checkout's own ref files hold."""
    from hermes_cli import _early_recovery, _update_takeover
    from hermes_cli.update_host_obligation import read_host_obligation

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    root = tmp_path / "checkout"
    if git == "missing":
        # The git PATH and PM's store cannot find; the checkout is a linked worktree whose branch
        # is only in the common dir's packed-refs.
        monkeypatch.setattr(_early_recovery, "_git_executable", lambda *args, **kwargs: "git")
        worktree = tmp_path / "repo.git" / "worktrees" / "checkout"
        worktree.mkdir(parents=True)
        (worktree / "commondir").write_text("../..\n", encoding="utf-8")
        (worktree / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (tmp_path / "repo.git" / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{'d' * 40} refs/heads/other\n{SHA} refs/heads/main\n",
            encoding="utf-8")
        root.mkdir()
        (root / ".git").write_text(f"gitdir: {worktree}\n", encoding="utf-8")
    else:
        stub = bin_dir / "git"
        stub.write_text("#!/bin/sh\nexit 128\n", encoding="utf-8")
        stub.chmod(0o755)
        monkeypatch.setattr(_early_recovery, "_git_executable", lambda *args, **kwargs: str(stub))
        (root / ".git" / "refs" / "heads").mkdir(parents=True)
        (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (root / ".git" / "refs" / "heads" / "main").write_text(f"{SHA}\n", encoding="utf-8")

    _update_takeover._arm_fleet_obligation(root)

    assert (read_host_obligation() or {}).get("expected_sha") == SHA


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_takeover_arms_an_sha_less_obligation_for_a_git_less_root(tmp_path, monkeypatch):
    """An archive install has no HEAD to name, yet its update still owes the fleet a restart (F21):
    an SHA-less record (readers hold the fleet to the checkout) instead of no record at all."""
    from hermes_cli import _update_takeover
    from hermes_cli.update_host_obligation import read_host_obligation

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # no enclosing repo answers for it
    root = tmp_path / "archive"
    root.mkdir()

    _update_takeover._arm_fleet_obligation(root)

    record = read_host_obligation()
    assert record is not None and record["expected_sha"] == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_takeover_arms_the_host_record_without_application_dependencies(tmp_path):
    """The arm runs in the HISTORICAL interpreter, before PM installs the new dependencies. A
    ruamel-less interpreter (-I -S here) must still write the HOST record every profile reads, not
    only the arming profile's per-home breadcrumb (R1-1)."""
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "checkout"
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    rev_parse = [*git, "rev-parse", "HEAD"]
    head = subprocess.run(rev_parse, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    lock_dir, home = tmp_path / "gateway-locks", tmp_path / "home"
    home.mkdir()
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c",
         "import importlib.util, sys\nfrom pathlib import Path\n"
         f"sys.path.insert(0, {str(source)!r})\n"
         "assert importlib.util.find_spec('ruamel') is None, 'precondition: no app dependencies'\n"
         "from hermes_cli._update_takeover import _arm_fleet_obligation\n"
         f"_arm_fleet_obligation(Path({str(root)!r}))\n"],
        env={**os.environ, "HERMES_GATEWAY_LOCK_DIR": str(lock_dir), "HERMES_HOME": str(home)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    [armed] = lock_dir.glob("host-update-restart-*.json")  # the install-keyed record (review S3)
    assert json.loads(armed.read_text(encoding="utf-8-sig"))["expected_sha"] == head


def test_a_finish_child_that_cannot_start_after_the_commit_is_owed_not_failed(tmp_path, monkeypatch, capsys):
    """The historical updater already moved the tree and prepare() armed the tail. A Popen/resume
    OSError starting update_finish used to land in the preparation catch-all: receipt failed,
    .update_exit_code=1, exit 1 for a committed update (review C7, invariant 3)."""
    from hermes_cli import _update_takeover, update_custody, update_lock, update_receipt
    from hermes_constants import get_hermes_home

    class _Lock:
        holder = None

        def __init__(self, **_kwargs):
            pass

        def acquire(self):
            return True

        def release(self):
            pass

    def no_child(*_args, **_kwargs):
        raise OSError(12, "Cannot allocate memory")

    monkeypatch.setattr(update_lock, "UpdateLock", _Lock)
    monkeypatch.setattr(_update_takeover, "prepare", lambda request: (Path(sys.executable), dict(os.environ)))
    monkeypatch.setattr(update_custody, "popen_post_commit", no_child)
    saved = []
    real_finalize = update_receipt.finalize_pending_update_receipt
    monkeypatch.setattr(update_receipt, "finalize_pending_update_receipt",
                        lambda code=None, reason="": saved.append(code) or real_finalize(code, reason))
    root = tmp_path / "root"
    root.mkdir()
    context, result = tmp_path / "context.json", tmp_path / "result.json"
    context.write_text(json.dumps({"root": str(root), "gateway_mode": True, "receipt": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["_update_takeover", str(context), str(result)])

    assert _update_takeover.main() == 0

    assert saved == [0]
    assert (get_hermes_home() / ".update_exit_code").read_text(encoding="utf-8").strip() == "0"
    assert json.loads(result.read_text(encoding="utf-8"))["resume_handled"] is False
    assert "finishing steps did not run" in capsys.readouterr().err


def test_the_takeover_breadcrumb_is_written_temp_fsync_rename(tmp_path, monkeypatch):
    """With the host record unwritable the takeover falls back to the per-home breadcrumb; it was
    written in place (review C12, invariant 5): a crash mid-write left a truncated debt record."""
    from hermes_cli import _update_takeover, update_host_obligation
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(update_host_obligation, "write_host_obligation", lambda **_kw: False)
    monkeypatch.setattr(_update_takeover, "_head_sha", lambda _root: "a" * 40)
    synced, replaced = [], []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((Path(dst), len(synced)))
                        or real_replace(src, dst))

    _update_takeover._arm_fleet_obligation(tmp_path)

    crumb = get_hermes_home() / "fleet_restart_pending"
    assert replaced and replaced[-1][0] == crumb and replaced[-1][1] > 0
    assert f"expected_sha={'a' * 40}" in crumb.read_text(encoding="utf-8")


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory permissions")
def test_a_takeover_debt_another_profile_cannot_see_is_never_silent(tmp_path, monkeypatch, capsys):
    """The takeover arms after the historical updater already moved the tree, through the same
    fallback as the commit point: with the host record unwritable on a two-profile install, the
    arming profile's marker hides the debt from the other one, so the takeover must say so (it
    cannot refuse a move that already happened) (kshitijk4poor P2, review S2)."""
    from hermes_cli import _update_takeover

    homes = {name: tmp_path / "profiles" / name for name in ("coder", "writer")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv("HERMES_HOME", str(homes["coder"]))
    monkeypatch.setattr(_update_takeover, "_head_sha", lambda _root: "a" * 40)
    lock_dir.chmod(0o500)
    try:
        _update_takeover._arm_fleet_obligation(tmp_path)
    finally:
        lock_dir.chmod(0o700)

    assert (homes["coder"] / "fleet_restart_pending").is_file()
    assert "hermes gateway restart" in capsys.readouterr().err


def test_a_historical_updater_arms_the_host_record_under_the_current_mutex(tmp_path, monkeypatch):
    """An N-1 updater imported ``update_lock`` before its pull, then lazily imports the pulled
    ``update_host_obligation``: the arm must still write the record, under the same sidecar lock
    current processes hold, instead of crashing the update on the missing ``marker_mutex``."""
    from hermes_cli import update_lock
    from hermes_cli.update_host_obligation import read_host_obligation, write_host_obligation

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    monkeypatch.delattr(update_lock, "marker_mutex")  # N-1's in-memory update_lock had no such name
    assert write_host_obligation(expected_sha="a" * 40)
    assert read_host_obligation()["expected_sha"] == "a" * 40
    assert (tmp_path / "gateway-locks" / ".host-update-restart.mutex" / "record.lock").exists()
