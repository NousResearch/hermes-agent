"""`hermes gateway install` from a macOS bundle leaves the signed app untouched.

Anything written into ``Hermes.app`` breaks its code seal, and macOS then holds the next update
on a Gatekeeper prompt. The service runs the app's own signed ``bin/hermes`` instead.
"""
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build.launchers import posix_launcher
from tests.hermes_cli.test_gateway_service import _osascript_exec_argv

ROOT = Path(__file__).resolve().parents[2]


def _mac_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The payload a macOS app ships (scripts/build/agent.py) around a stub CLI: app, repo, bin/hermes."""
    app = tmp_path / "Hermes.app"
    payload = app / "Contents" / "Resources" / "agent-payload"
    repo = payload / "hermes-agent"
    (repo / "hermes_cli").mkdir(parents=True)
    for relative in ("hermes_cli/__init__.py", "hermes_cli/stderr_timestamp.py"):
        shutil.copy2(ROOT / relative, repo / relative)
    (repo / "hermes_cli" / "main.py").write_text(
        "import json, sys\ndef main():\n    print(json.dumps(sys.argv[1:]), flush=True)\n    return 7\n", encoding="utf-8")
    (repo / "install-stamp.json").write_text(json.dumps({
        "commit": "a" * 40, "distribution": "desktop-app", "updateMechanism": "electron-updater",
        "payload": "bundled"}), encoding="utf-8")
    # The store holds the app's interpreter, so the store-launcher publication gate would open.
    interpreter = Path(sys._base_executable).resolve()
    (payload / "tools").mkdir()
    (payload / "tools" / "facts.json").write_text(json.dumps(
        {"schema": 1, "packages": {"python": {"entry": str(interpreter.parents[1])}}}), encoding="utf-8")
    site = "venv/lib/site-packages"
    (payload / site).mkdir(parents=True)
    (payload / "manifest.json").write_text(json.dumps({
        "schema": 1, "target": "darwin-arm64", "repo": "hermes-agent", "venv": "venv", "store": "tools",
        "launchers": ["hermes"], "runtime": {"sitePackages": site, "commands": {"hermes": "bin/hermes"}}}),
        encoding="utf-8")
    launcher = payload / "bin" / "hermes"
    launcher.parent.mkdir()
    launcher.write_text(posix_launcher("hermes", "hermes_cli.main:main", python=str(interpreter), repo="hermes-agent",
                                       site=site, target="darwin-arm64"), encoding="utf-8")
    launcher.chmod(0o755)
    return app, repo, launcher


@pytest.mark.platforms("macos")
@pytest.mark.spawns_gateway_lookalike
def test_gateway_service_runs_the_apps_own_launcher_and_writes_nothing_into_it(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import gateway
    from hermes_agent.hermes_cli.gateway_runtime_service_identity import verify_launchd_plist

    app, repo, launcher = _mac_bundle(tmp_path)
    home = tmp_path / "home" / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home.parent))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(gateway, "PROJECT_ROOT", repo)
    sealed = sorted(app.rglob("*"))

    gateway._prepare_service_launcher()
    definition = plistlib.loads(gateway.generate_launchd_plist().encode())

    assert sorted(app.rglob("*")) == sealed, "gateway install wrote into the signed app"
    command = _osascript_exec_argv(definition["ProgramArguments"])[:-4]  # minus the shell's log redirects
    assert command[0] == str(launcher)
    verify_launchd_plist(definition, gateway.get_launchd_label(), home)
    # The service command runs: the app's launcher starts the stderr wrapper, which starts the gateway.
    env = {**os.environ, **definition["EnvironmentVariables"]}
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 7, result.stdout + result.stderr
    # stderr_timestamp stamps the child's stdout lines ("YYYY-MM-DD HH:MM:SS,mmm ").
    assert json.loads(result.stdout.split(" ", 2)[2]) == ["gateway", "run", "--external-supervisor"]
    assert sorted(app.rglob("*")) == sealed, "the service wrote into the signed app"
