"""``hermes desktop`` from inside a BUNDLED desktop payload.

A bundled artifact ships the CLI inside ``<app>/resources/agent-payload``
and the desktop app is the artifact itself — there is no source tree to
build and the app resources are signed and read-only. These tests pin the
two halves of that: where the launcher is (hermes_cli.bundled_app) and
that cmd_gui starts it instead of running the checkout build ladder.

The layout resolver is pure path arithmetic over real directories, so
every platform's shape is exercised from any host.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.main as cli_main
from hermes_cli.bundled_app import (
    NotBundledApp,
    launch_detached,
    resolve_bundle_layout,
)
from hermes_cli.steward import is_bundled_payload

STAMP = {
    "schemaVersion": 2,
    "commit": "a" * 40,
    "commitDate": 1750000000,
    "branch": None,
    "builtAt": "2026-08-21T00:00:00+00:00",
    "dirty": False,
    "source": "ci",
    "distribution": "desktop-app",
    "updateMechanism": "electron-updater",
    "baseVersion": "0.27.0",
    "displayVersion": "0.27.0",
    "distance": 0,
    "payload": "bundled",
    "tag": "v0.27.0",
}


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ")
    path.chmod(0o755)
    return path


def _payload(app_root: Path, *, resources_name: str = "resources") -> Path:
    """Stage a payload repo under *app_root*; returns the repo root."""
    repo = app_root / resources_name / "agent-payload" / "repo"
    # repo/apps/desktop survives prunePayload(), which is exactly why a
    # package.json probe cannot tell a bundle from a checkout.
    (repo / "apps" / "desktop").mkdir(parents=True)
    (repo / "apps" / "desktop" / "package.json").write_text('{"name":"desktop"}\n')
    (repo / "install-stamp.json").write_text(json.dumps(STAMP) + "\n", encoding="utf-8")
    return repo


def _linux_bundle(tmp_path: Path) -> Path:
    app = tmp_path / "linux-unpacked"
    repo = _payload(app)
    _exe(app / "Hermes")
    # The helpers a Linux Electron tree ships beside its launcher.
    _exe(app / "chrome-sandbox")
    _exe(app / "chrome_crashpad_handler")
    _exe(app / "libffmpeg.so")
    (app / "version").write_text("40.9.3\n")
    return repo


def _windows_bundle(tmp_path: Path) -> Path:
    app = tmp_path / "Hermes"
    repo = _payload(app)
    _exe(app / "Hermes.exe")
    _exe(app / "Uninstall Hermes.exe")
    return repo


def _macos_bundle(tmp_path: Path) -> Path:
    app = tmp_path / "Hermes.app"
    repo = _payload(app / "Contents", resources_name="Resources")
    _exe(app / "Contents" / "MacOS" / "Hermes")
    return repo


def _host_bundle(tmp_path: Path) -> tuple[Path, Path]:
    """A bundle in THIS host's shape, plus its expected launcher.

    cmd_gui resolves the launcher with the real sys.platform, and the
    Linux rule depends on POSIX execute bits that do not exist on Windows
    — so the cmd_gui tests must stage the host's own shape.
    """
    if sys.platform == "win32":
        repo = _windows_bundle(tmp_path)
        return repo, tmp_path / "Hermes" / "Hermes.exe"
    repo = _linux_bundle(tmp_path)
    return repo, tmp_path / "linux-unpacked" / "Hermes"


class TestResolveBundleLayout:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX-only: the Linux launcher rule reads execute bits, "
        "which os.access reports as always-set on Windows",
    )
    def test_linux_launcher_is_the_one_non_helper_executable(self, tmp_path):
        repo = _linux_bundle(tmp_path)
        layout = resolve_bundle_layout(repo, platform="linux")
        assert layout.app_root == tmp_path / "linux-unpacked"
        assert layout.payload.name == "agent-payload"
        assert layout.launcher == tmp_path / "linux-unpacked" / "Hermes"

    def test_windows_launcher_is_never_the_uninstaller(self, tmp_path):
        repo = _windows_bundle(tmp_path)
        layout = resolve_bundle_layout(repo, platform="win32")
        assert layout.launcher == tmp_path / "Hermes" / "Hermes.exe"

    def test_macos_app_root_climbs_out_of_contents_resources(self, tmp_path):
        repo = _macos_bundle(tmp_path)
        layout = resolve_bundle_layout(repo, platform="darwin")
        assert layout.app_root == tmp_path / "Hermes.app"
        assert layout.resources == tmp_path / "Hermes.app" / "Contents" / "Resources"
        assert layout.launcher == tmp_path / "Hermes.app" / "Contents" / "MacOS" / "Hermes"

    def test_ambiguous_launcher_reports_none_rather_than_guessing(self, tmp_path):
        repo = _windows_bundle(tmp_path)
        _exe(repo.parent.parent.parent / "Other.exe")
        assert resolve_bundle_layout(repo, platform="win32").launcher is None

    def test_a_checkout_is_not_a_bundle(self, tmp_path):
        checkout = tmp_path / "hermes-agent"
        (checkout / "apps" / "desktop").mkdir(parents=True)
        with pytest.raises(NotBundledApp):
            resolve_bundle_layout(checkout)

    def test_a_sealed_tree_with_no_app_is_not_a_bundle(self, tmp_path):
        # docker/nix shape: sealed repo, no agent-payload parent.
        repo = tmp_path / "opt" / "hermes" / "repo"
        repo.mkdir(parents=True)
        with pytest.raises(NotBundledApp):
            resolve_bundle_layout(repo)


class TestShapePredicate:
    """cmd_gui and the bundle build must agree about what a bundle is."""

    def test_a_stamped_payload_is_bundled(self, tmp_path):
        assert is_bundled_payload(_linux_bundle(tmp_path))

    def test_a_surviving_desktop_package_json_does_not_make_a_bundle(self, tmp_path):
        """repo/apps/desktop survives the payload prune, so a filesystem
        probe cannot be the authority — only the stamp can."""
        checkout = tmp_path / "hermes-agent"
        (checkout / "apps" / "desktop").mkdir(parents=True)
        (checkout / "apps" / "desktop" / "package.json").write_text("{}\n")
        assert not is_bundled_payload(checkout)

    def test_a_bootstrap_artifact_is_not_bundled(self, tmp_path):
        root = tmp_path / "hermes-agent"
        root.mkdir()
        stamp = dict(STAMP, payload="bootstrap", tag=None)
        (root / "install-stamp.json").write_text(json.dumps(stamp) + "\n")
        assert not is_bundled_payload(root)

    def test_an_unreadable_stamp_is_not_bundled(self, tmp_path):
        root = tmp_path / "hermes-agent"
        root.mkdir()
        (root / "install-stamp.json").write_text("{ not json\n")
        assert not is_bundled_payload(root)


class TestLaunchDetached:
    def test_the_child_outlives_this_process_and_owns_no_stdio(self):
        seen = {}

        def fake_popen(argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs
            return SimpleNamespace(pid=1234)

        with patch("hermes_cli.bundled_app.subprocess.Popen", side_effect=fake_popen):
            pid = launch_detached(["/app/Hermes", "--no-sandbox"], cwd="/app")

        assert pid == 1234
        assert seen["argv"] == ["/app/Hermes", "--no-sandbox"]
        assert seen["kwargs"]["cwd"] == "/app"
        assert seen["kwargs"]["stdout"] is subprocess.DEVNULL
        assert seen["kwargs"]["stderr"] is subprocess.DEVNULL
        assert seen["kwargs"]["stdin"] is subprocess.DEVNULL
        # POSIX detaches with a new session; Windows with creation flags.
        detach_key = "creationflags" if sys.platform == "win32" else "start_new_session"
        assert seen["kwargs"][detach_key]

    def test_a_real_child_survives_and_is_reachable(self, tmp_path):
        """The one E2E rung: spawn a real process and prove it ran."""
        marker = tmp_path / "ran.txt"
        pid = launch_detached(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('yes')"]
        )
        assert pid > 0
        deadline = 10.0
        import time

        while deadline > 0 and not marker.exists():
            time.sleep(0.1)
            deadline -= 0.1
        assert marker.read_text() == "yes"


class TestCmdGuiOnABundle:
    """The behavior the checkout ladder got wrong: never build in a bundle."""

    @staticmethod
    def _args(**overrides):
        base = dict(
            source=False, build_only=False, fake_boot=False, ignore_existing=False,
            hermes_root=None, cwd=None, skip_build=False, force_build=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    @staticmethod
    def _run(monkeypatch, repo: Path, args, *, launcher_ok: bool = True):
        """Run cmd_gui against *repo*, recording builds and launches."""
        builds: list[list[str]] = []
        launches: list[list[str]] = []

        def record_run(cmd, *a, **kw):
            builds.append([str(c) for c in cmd])
            return subprocess.CompletedProcess(cmd, 0)

        def record_npm_install(npm, root, **kw):
            builds.append(["npm", "ci", str(root)])
            return subprocess.CompletedProcess(["npm", "ci"], 0)

        def record_popen(argv, **kw):
            launches.append([str(a) for a in argv])
            return SimpleNamespace(pid=4242)

        monkeypatch.setattr(cli_main, "PROJECT_ROOT", repo)
        monkeypatch.setattr(cli_main, "_resolve_node_runtime_npm", lambda: "/usr/bin/npm")
        monkeypatch.setattr(cli_main, "_desktop_build_needed", lambda *a, **k: True)
        monkeypatch.setattr(cli_main, "_write_desktop_build_stamp", lambda *a, **k: None)
        monkeypatch.setattr(
            cli_main, "_run_npm_install_deterministic", record_npm_install
        )
        monkeypatch.setattr(
            cli_main, "_stop_desktop_processes_locking_build", lambda *a, **k: []
        )
        monkeypatch.setattr(
            cli_main, "_desktop_linux_sandbox_fixup", lambda *a, **k: launcher_ok
        )
        monkeypatch.setattr(
            cli_main, "_desktop_linux_needs_no_sandbox", lambda: not launcher_ok
        )
        monkeypatch.setattr(
            cli_main,
            "_desktop_linux_sandbox_helper_is_regular_file",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(cli_main, "_detect_linux_password_store", lambda: None)

        with patch("hermes_cli.bundled_app.subprocess.Popen", side_effect=record_popen), \
             patch.object(cli_main.subprocess, "run", side_effect=record_run), \
             pytest.raises(SystemExit) as exit_info:
            cli_main.cmd_gui(args)

        return exit_info.value.code, builds, launches

    def test_launches_the_bundle_and_builds_nothing(self, tmp_path, monkeypatch):
        repo, launcher = _host_bundle(tmp_path)
        code, builds, launches = self._run(monkeypatch, repo, self._args())

        assert code == 0
        # THE regression: repo/apps/desktop/package.json exists in a bundle,
        # so the old probe fell through to npm ci + npm run pack inside the
        # app's own signed resources.
        assert builds == []
        assert launches == [[str(launcher)]]

    def test_a_checkout_still_takes_the_build_ladder(self, tmp_path, monkeypatch):
        checkout = tmp_path / "hermes-agent"
        (checkout / "apps" / "desktop").mkdir(parents=True)
        (checkout / "apps" / "desktop" / "package.json").write_text("{}\n")
        code, builds, launches = self._run(monkeypatch, checkout, self._args())

        assert launches == []
        assert any("npm" in " ".join(cmd) for cmd in builds), builds

    @pytest.mark.parametrize("flag", ["source", "build_only", "force_build"])
    def test_build_flags_are_refused_rather_than_ignored(self, tmp_path, monkeypatch, flag):
        repo, _launcher = _host_bundle(tmp_path)
        code, builds, launches = self._run(
            monkeypatch, repo, self._args(**{flag: True})
        )

        assert code == 2
        assert builds == []
        assert launches == []

    def test_a_stamped_bundle_with_no_app_reports_damage(self, tmp_path, monkeypatch):
        """A bundled stamp over a non-bundle tree must never fall back to
        the build ladder — that would npm-build inside app resources."""
        orphan = tmp_path / "somewhere" / "repo"
        orphan.mkdir(parents=True)
        (orphan / "install-stamp.json").write_text(json.dumps(STAMP) + "\n")
        code, builds, launches = self._run(monkeypatch, orphan, self._args())

        assert code == 1
        assert builds == []
        assert launches == []

    def test_a_bundle_with_no_resolvable_launcher_reports_damage(self, tmp_path, monkeypatch):
        app = tmp_path / "linux-unpacked"
        repo = _payload(app)
        _exe(app / "Hermes")
        _exe(app / "Hermes-Other")
        code, builds, launches = self._run(monkeypatch, repo, self._args())

        assert code == 1
        assert builds == []
        assert launches == []
