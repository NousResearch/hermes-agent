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
        f"Path(sys.argv[2]).write_text(json.dumps({{'resume_handled': True}}), encoding='utf-8')\n"
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
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    handoff = json.loads((home / "logs/update_receipts").glob("post_swap_*.json").__next__().read_text())
    assert handoff["had_desktop_app_before_update"] is desktop
    assert "desktop" not in handoff and "assume_yes" not in handoff


@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_takeover_waits_propagates_status_and_never_reenters_old_code(tmp_path, status, encoding):
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
        "assert request['desktop'] is True\n"
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
        "had_desktop_app_before_update = True\n"
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
    result = subprocess.run([sys.executable, "-B", str(program)], env=env,
                            capture_output=True, text=True, timeout=30)
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
                            capture_output=True, text=True, timeout=30)
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
                            capture_output=True, text=True, timeout=30)
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
        "import json, os, runpy, subprocess, sys\nfrom pathlib import Path\n"
        "calls = []\n"
        "def forbidden(*args, **kwargs):\n    raise AssertionError('updater/process control ran')\n"
        "def spawn(command, **kwargs):\n"
        "    assert kwargs['start_new_session'] is True\n"
        "    assert kwargs['stdin'] == subprocess.DEVNULL\n"
        "    calls.append(command)\n"
        f"    if {spawn_fails!r}:\n        raise OSError('probe spawn failed')\n"
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
                            env=env, capture_output=True, text=True, timeout=30)
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
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(result_path.read_text(encoding="utf-8-sig")) == {"serves_handled": True}
    assert not result_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert context.read_bytes() == before


def test_bootstrap_lock_remains_live_without_application_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = (
        "import os, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from hermes_cli.update_lock import UpdateLock\n"
        f"path = Path({str(tmp_path / 'lock')!r})\n"
        "first = UpdateLock(path=path)\nassert first.acquire()\n"
        "second = UpdateLock(path=path)\nassert not second.acquire(), 'live lock was stolen'\n"
        "assert second.holder.pid == os.getpid()\n"
        "first.release()\n"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
