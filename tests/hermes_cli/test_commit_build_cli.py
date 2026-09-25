"""Exercise the installed CLI with a sealed code copy and its build stamp."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

MESSAGE = "This build doesn't get updates. Ask the developer who gave it to you for a new build."


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    root = tmp_path_factory.mktemp("commit-payload")
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "hermes_cli", root / "hermes_cli", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "install-stamp.json").write_text(json.dumps({
        "source": "commit-build", "distribution": "desktop-app", "payload": "bundled",
        "updateMechanism": "external", "commit": "a" * 40,
        "displayVersion": "1.2.3+gabcdef12", "baseVersion": "1.2.3", "tag": None,
    }))
    return root, source


# The entrypoint runs only from the temporary sealed copy, under an audit hook.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("args", [
    ["--version"], ["update"], ["update", "--check"],
    ["update", "--force", "--yes"], ["update", "--check", "--branch", "main"],
    ["update", "--channel", "canary"], ["update", "--set-channel", "stable"],
])
def test_sealed_cli_never_checks_or_spawns_updater(payload, tmp_path, args):
    root, source = payload
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_INSTALL_ROOT": str(root)}
    # Real imports and parser, no admission mocks. Audit attempts even if a caller swallows the error.
    script = f"""
import json, sys
sys.path[:0] = [{str(root)!r}, {str(source)!r}]
sys.argv = ['hermes', *{args!r}]
attempts = []
def audit(event, args):
    if event in ('socket.connect', 'subprocess.Popen', 'os.system'):
        attempts.append(event)
        raise RuntimeError('forbidden update side effect: ' + event)
sys.addaudithook(audit)
try:
    from hermes_cli.main import main
    main()
finally:
    print('AUDIT=' + json.dumps(attempts))
"""
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True, timeout=45)
    assert result.returncode == (0 if args == ["--version"] else 2), result.stdout + result.stderr
    assert "AUDIT=[]" in result.stdout, result.stdout + result.stderr
    if args == ["--version"]:
        assert "1.2.3+gabcdef12" in result.stdout
        assert "commit-build" in result.stdout
    else:
        assert MESSAGE in result.stdout
        assert "Manage updates from within the desktop app" not in result.stdout
    assert not (home / "config.yaml").exists()
    assert not (home / ".update_check").exists()
