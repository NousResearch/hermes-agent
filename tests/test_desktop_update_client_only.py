"""POSIX hand-off: runtime-free remote clients must not abort as a broken install.

The actual bug is in ``scripts/desktop-update/posix.sh``: a missing
``venv/bin/hermes`` used to write the repair abort (exit 3) even when the
Desktop already classified the machine as a remote-only thin client. These
tests drive the real ``posix.sh`` (daemonized launcher included) against a
temp checkout with no venv. No source-reading.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"

requires_posix_handoff = pytest.mark.skipif(
    not (os.path.exists("/bin/bash") and os.path.exists("/usr/bin/python3")),
    reason="posix.sh detaches through /bin/bash and /usr/bin/python3",
)


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hermes Test",
        "GIT_AUTHOR_EMAIL": "hermes@example.invalid",
        "GIT_COMMITTER_NAME": "Hermes Test",
        "GIT_COMMITTER_EMAIL": "hermes@example.invalid",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return (result.stdout or "").strip()


def _init_origin(path: Path) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "commit.gpgSign", "false")
    _git(path, "config", "user.name", "Hermes Test")
    _git(path, "config", "user.email", "hermes@example.invalid")
    (path / "README.md").write_text("first\n", encoding="utf-8")
    (path / "apps/desktop").mkdir(parents=True)
    (path / "apps/desktop/package.json").write_text("{}", encoding="utf-8")
    entitlements = path / "apps/desktop/electron"
    entitlements.mkdir()
    for name in ("entitlements.mac.plist", "entitlements.mac.inherit.plist"):
        shutil.copy2(REPO_ROOT / "apps/desktop/electron" / name, entitlements / name)
    _git(path, "add", ".")
    _git(path, "commit", "-m", "first")
    return _git(path, "rev-parse", "HEAD")


def _wait_result(hermes_home: Path, timeout: float = 45.0) -> dict:
    result_path = hermes_home / ".hermes-update-result.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
            if "exit_code" in payload:
                return payload
        time.sleep(0.1)
    raise AssertionError(f"hand-off never wrote a result file at {result_path}")


def _launch_posix(
    *,
    install_root: Path,
    extra_args: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "TMPDIR": str(install_root.parent),
        "PYTHONPATH": str(REPO_ROOT),
        "GIT_TERMINAL_PROMPT": "0",
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [
            "/bin/bash",
            str(SHIM_DIR / "posix.sh"),
            "--install-root",
            str(install_root),
            "--branch",
            "main",
            "--no-ui",
            *(extra_args or []),
        ],
        env=env,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _prepare_behind_clone(tmp_path: Path) -> tuple[Path, Path, str, str]:
    origin = tmp_path / "origin"
    first = _init_origin(origin)
    (origin / "README.md").write_text("second\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "second")
    second = _git(origin, "rev-parse", "HEAD")

    hermes_home = tmp_path / "home"
    install = hermes_home / "hermes-agent"
    _git(origin.parent, "clone", "--quiet", str(origin), str(install))
    _git(install, "config", "commit.gpgSign", "false")
    _git(install, "reset", "--hard", first)
    (install / ".git/info/exclude").write_text("hermes_cli\napps/desktop/release/\n", encoding="utf-8")
    # Production PYTHONPATH is the checkout; a real install already has this
    # package. The tiny origin fixture does not, so link the live module.
    (install / "hermes_cli").symlink_to(REPO_ROOT / "hermes_cli")
    _install_fake_npm(hermes_home)
    assert not (install / "venv" / "bin" / "hermes").exists()
    return install, hermes_home, first, second


@requires_posix_handoff
def test_posix_missing_runtime_without_client_only_still_aborts_repair(tmp_path: Path):
    """Local / unclassified missing venv stays the existing repair abort."""
    install = tmp_path / "hermes-agent"
    install.mkdir()
    launched = _launch_posix(install_root=install)
    assert launched.returncode == 0  # launcher exits; daemon writes the result

    payload = _wait_result(tmp_path)
    assert payload["ok"] is False
    assert payload["exit_code"] == 3
    assert "needs repair" in payload["message"]
    assert "venv/bin/hermes is missing" in payload["message"]


@requires_posix_handoff
def test_posix_client_only_missing_runtime_succeeds_and_reads_commit(tmp_path: Path):
    """The actual bug: --client-only + no venv used to take the repair abort."""
    install, hermes_home, _first, second = _prepare_behind_clone(tmp_path)
    connections = hermes_home / "connections.json"
    connections.write_text(
        json.dumps(
            {
                "launchMode": "primary",
                "primary": "vps",
                "connections": [
                    {"id": "vps", "kind": "ssh", "host": "vps.example.invalid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    before = connections.read_text(encoding="utf-8")

    launched = _launch_posix(install_root=install, extra_args=["--client-only"])
    assert launched.returncode == 0

    payload = _wait_result(hermes_home)
    assert payload["ok"] is True, payload
    assert payload["exit_code"] == 0
    assert "needs repair" not in payload["message"]
    assert payload["commit"] == second
    assert _git(install, "rev-parse", "HEAD") == second
    assert connections.read_text(encoding="utf-8") == before

    receipt_path = hermes_home / "logs" / "update_receipts" / "latest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == "client_only"
    assert any(
        step.get("name") == "fleet_restart" and step.get("skipped")
        for step in receipt["steps"]
    )
    # Own marker is cleaned by finish(); we never delete a foreign lock.
    marker = hermes_home / ".hermes-update-in-progress"
    assert not marker.exists()


@requires_posix_handoff
@pytest.mark.linux_only
def test_posix_client_only_success_still_relaunches_linux_unpacked(tmp_path: Path):
    """Client-only success must keep the existing Linux relaunch path."""
    install, hermes_home, _first, second = _prepare_behind_clone(tmp_path)
    from hermes_cli.client_only_update import _desktop_layout
    unpacked = _desktop_layout(install / "apps/desktop/release")[0]
    unpacked.mkdir(parents=True)
    stamp = hermes_home / "relaunch.stamp"
    target = unpacked / "Hermes"
    target.write_text(
        "#!/bin/sh\n"
        "printf launched > \"$HERMES_TEST_RELAUNCH_STAMP\"\n"
        "while [ ! -f \"$HERMES_TEST_RELAUNCH_DONE\" ]; do sleep 0.2; done\n",
        encoding="utf-8",
    )
    target.chmod(0o755)
    done = hermes_home / "relaunch.done"

    try:
        launched = _launch_posix(
            install_root=install,
            extra_args=[
                "--client-only",
                "--relaunch-target",
                str(target),
                "--relaunch-cwd",
                str(hermes_home),
            ],
            env_extra={
                "HERMES_TEST_RELAUNCH_STAMP": str(stamp),
                "HERMES_TEST_RELAUNCH_DONE": str(done),
            },
        )
        assert launched.returncode == 0

        payload = _wait_result(hermes_home)
        assert payload["ok"] is True, payload
        assert payload["exit_code"] == 0
        assert payload["commit"] == second

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not stamp.exists():
            time.sleep(0.1)
        assert stamp.exists(), "Linux relaunch was not attempted after client-only success"
        assert stamp.read_text(encoding="utf-8") == "launched"
    finally:
        # Cooperative stop only — the dummy was setsid-detached by posix.sh,
        # so os.kill would trip the live-system guard.
        done.write_text("1", encoding="utf-8")


def _install_fake_npm(hermes_home: Path):
    """Exercise the real packaging command path without downloading Electron."""
    npm = hermes_home / "node/bin/npm"
    npm.parent.mkdir(parents=True)
    npm.write_text('''#!/usr/bin/python3
import json, os, plistlib, shutil, subprocess, sys
from pathlib import Path
from hermes_cli.client_only_update import _desktop_layout
if 'builder' in sys.argv:
    output = Path(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('-c.directories.output=')))
    bundle, executable, resources = _desktop_layout(output)
    executable.parent.mkdir(parents=True)
    script = '#!/bin/sh\\nexit 0\\n'
    if os.environ.get('HERMES_TEST_RELAUNCH_STAMP'):
        script = '#!/bin/sh\\nprintf launched > "$HERMES_TEST_RELAUNCH_STAMP"\\nwhile [ ! -f "$HERMES_TEST_RELAUNCH_DONE" ]; do sleep 0.2; done\\n'
    executable.write_text(script)
    executable.chmod(0o755)
    if sys.platform == 'darwin':
        shutil.copyfile('/usr/bin/true', executable)
        (bundle / 'Contents/Info.plist').write_bytes(plistlib.dumps({
            'CFBundleIdentifier': 'invalid.example.hermes-updater-test',
            'CFBundleExecutable': 'Hermes', 'CFBundlePackageType': 'APPL',
            'CFBundleName': 'Hermes', 'CFBundleVersion': '1',
        }))
    (resources / 'app.asar.unpacked/dist').mkdir(parents=True)
    (resources / 'app.asar.unpacked/dist/index.html').write_text('packaged fixture')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (resources / 'install-stamp.json').write_text(json.dumps({'commit': commit}))
''', encoding="utf-8")
    npm.chmod(0o755)


@pytest.mark.macos_only
@requires_posix_handoff
def test_macos_native_handoff_installs_matching_packaged_bundle(tmp_path: Path):
    from hermes_cli.client_only_update import _desktop_layout
    install, hermes_home, _, second = _prepare_behind_clone(tmp_path)
    launched = _launch_posix(install_root=install, extra_args=["--client-only"])
    assert launched.returncode == 0
    payload = _wait_result(hermes_home)
    assert payload["ok"] and payload["commit"] == second, payload
    _, executable, resources = _desktop_layout(install / "apps/desktop/release")
    assert os.access(executable, os.X_OK)
    assert json.loads((resources / "install-stamp.json").read_text())["commit"] == second
    assert (resources / "app.asar.unpacked/dist/index.html").read_text() == "packaged fixture"


@pytest.mark.linux_only
@requires_posix_handoff
def test_linux_relaunch_gate_accepts_only_host_architecture(tmp_path):
    from hermes_cli.client_only_update import _desktop_layout
    install = tmp_path / "hermes-agent"
    bundle, executable, _ = _desktop_layout(install / "apps/desktop/release")
    other = "linux-unpacked" if bundle.name == "linux-arm64-unpacked" else "linux-arm64-unpacked"
    for target, expected in ((executable, "relaunch"), (bundle.parent / other / "Hermes", "skew")):
        result = _launch_posix(install_root=install, extra_args=[
            "--self-test-gate", "--relaunch-target", str(target),
        ])
        assert result.returncode == 0
        assert result.stdout.strip().split(":", 1)[0] == expected
