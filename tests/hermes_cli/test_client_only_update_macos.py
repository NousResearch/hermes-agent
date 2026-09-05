"""macOS packaging regressions for the runtime-free Desktop updater."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli.client_only_update import _desktop_layout, run_client_only_update


def git(root, *args):
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
             "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"},
    ).strip()


@pytest.fixture
def desktop_checkout(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "commit.gpgSign", "false")
    (origin / "apps/desktop").mkdir(parents=True)
    (origin / "apps/desktop/package.json").write_text("{}")
    (origin / ".gitignore").write_text("apps/desktop/release/\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "initial")
    home = tmp_path / "home"
    home.mkdir()
    root = home / "hermes-agent"
    git(tmp_path, "clone", str(origin), str(root))
    before = git(root, "rev-parse", "HEAD")
    canonical, executable, resources = _desktop_layout(root / "apps/desktop/release")
    executable.parent.mkdir(parents=True)
    executable.write_text("previous app")
    executable.chmod(0o755)
    resources.mkdir(parents=True)
    (resources / "install-stamp.json").write_text(json.dumps({"commit": before}))
    (origin / "change.txt").write_text("updated")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "updated")
    managed = home / "node/bin"
    managed.mkdir(parents=True)
    (managed / "npm").write_text("#!/bin/sh\nexit 0\n")
    (managed / "npm").chmod(0o755)
    return root, home, canonical, executable, before


@pytest.fixture
def fake_macos_signer(monkeypatch):
    """Keep fake bundle tests focused on updater promotion and signing gates."""
    from hermes_cli import main_desktop

    calls = []
    real_which = shutil.which

    def signer(desktop_dir, *, release_dir=None, **kwargs):
        calls.append((desktop_dir, release_dir))
        return True

    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", signer)
    monkeypatch.setattr(
        main_desktop,
        "_codesign_verify",
        lambda codesign, app, **kwargs: subprocess.CompletedProcess(
            [codesign, "--verify", str(app)], 0, "", ""
        ),
    )
    monkeypatch.setattr(
        main_desktop.shutil,
        "which",
        lambda name, path=None: "/usr/bin/codesign"
        if name == "codesign"
        else real_which(name, path=path),
    )
    return calls


def package_runner(root, home, *, wrong_stamp=False, fail_build=False, reopen=False):
    calls = []
    checks = 0

    def run(args, *, cwd, env=None):
        nonlocal checks
        calls.append((list(args), cwd, env))
        if args[0] == "ps":
            checks += 1
            executable = _desktop_layout(root / "apps/desktop/release")[1]
            stdout = f"12345 {executable}\n" if reopen and checks > 1 else ""
            return subprocess.CompletedProcess(args, 0, stdout, "")
        if Path(args[0]).name == "npm":
            assert args[0] == str(home / "node/bin/npm")
            assert env["CI"] == "1"
            if "builder" in args:
                if fail_build:
                    return subprocess.CompletedProcess(args, 1, "", "package failed")
                output = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("-c.directories.output=")))
                assert not output.is_relative_to(root / "apps/desktop/release")
                _, executable, resources = _desktop_layout(output)
                executable.parent.mkdir(parents=True)
                executable.write_text("new app")
                executable.chmod(0o755)
                (resources / "app.asar.unpacked/dist").mkdir(parents=True)
                (resources / "app.asar.unpacked/dist/index.html").write_text("fixture")
                commit = "wrong" if wrong_stamp else git(root, "rev-parse", "HEAD")
                (resources / "install-stamp.json").write_text(json.dumps({"commit": commit}))
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)

    return run, calls


@pytest.mark.macos_only
def test_packages_verified_app_and_retains_previous_bundle(
    desktop_checkout, fake_macos_signer, monkeypatch
):
    root, home, canonical, executable, before = desktop_checkout
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("APPLE_API_KEY", "fixture-not-a-key")
    runner, calls = package_runner(root, home)
    result = run_client_only_update(root, hermes_home=home, run=runner)
    assert result.ok and result.rebuilt_desktop
    assert fake_macos_signer[0][0] == root / "apps/desktop"
    assert fake_macos_signer[0][1].is_relative_to(home / "backups" / "desktop-client-updates")
    assert executable.read_text() == "new app"
    assert result.installed_commit != before
    resources = _desktop_layout(root / "apps/desktop/release")[2]
    assert json.loads((resources / "install-stamp.json").read_text())["commit"] == result.installed_commit
    previous = list((home / "backups/desktop-client-updates").glob("*/previous-app/Contents/MacOS/Hermes"))
    assert len(previous) == 1 and previous[0].read_text() == "previous app"
    npm_calls = [args for args, _, _ in calls if Path(args[0]).name == "npm"]
    assert npm_calls[0][1:] == ["ci", "--include=dev"]
    assert "build" in npm_calls[1] and "builder" in npm_calls[2]
    assert all("ELECTRON_RUN_AS_NODE" not in env and "APPLE_API_KEY" not in env
               for args, _, env in calls if Path(args[0]).name == "npm")


@pytest.mark.macos_only
@pytest.mark.parametrize("failure", ["wrong_stamp", "fail_build", "reopen"])
def test_failed_package_or_reopened_app_keeps_working_bundle(
    desktop_checkout, failure, fake_macos_signer
):
    root, home, canonical, executable, before = desktop_checkout
    runner, _ = package_runner(root, home, **{failure: True})
    result = run_client_only_update(root, hermes_home=home, run=runner)
    assert not result.ok and result.exit_code == 6
    assert executable.read_text() == "previous app"
    assert git(root, "rev-parse", "HEAD") == before
    assert not (home / "logs/update_receipts/latest.json").exists()


@pytest.mark.macos_only
def test_failed_swap_restores_previous_app(desktop_checkout, fake_macos_signer, monkeypatch):
    root, home, canonical, executable, before = desktop_checkout
    runner, _ = package_runner(root, home)
    rename = Path.rename

    def fail_candidate_swap(self, target):
        if Path(target) == canonical and self.name == "Hermes.app":
            raise OSError("simulated failed rename")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_candidate_swap)
    result = run_client_only_update(root, hermes_home=home, run=runner)
    assert not result.ok
    assert executable.read_text() == "previous app"
    assert git(root, "rev-parse", "HEAD") == before


@pytest.mark.macos_only
def test_signing_failure_keeps_previous_app(
    desktop_checkout, fake_macos_signer, monkeypatch
):
    root, home, canonical, executable, before = desktop_checkout
    from hermes_cli import main_desktop

    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *args, **kwargs: False)
    runner, _ = package_runner(root, home)

    result = run_client_only_update(root, hermes_home=home, run=runner)

    assert not result.ok and result.exit_code == 6
    assert "could not be signed" in result.message
    assert executable.read_text() == "previous app"
    assert git(root, "rev-parse", "HEAD") == before
    assert not (home / "logs/update_receipts/latest.json").exists()


@pytest.mark.macos_only
def test_strict_verification_failure_keeps_previous_app(
    desktop_checkout, fake_macos_signer, monkeypatch
):
    root, home, canonical, executable, before = desktop_checkout
    from hermes_cli import main_desktop

    def failed_verify(codesign, app, **kwargs):
        return subprocess.CompletedProcess(
            [codesign, "--verify", str(app)], 1, "", "invalid signature"
        )

    monkeypatch.setattr(main_desktop, "_codesign_verify", failed_verify)
    runner, _ = package_runner(root, home)

    result = run_client_only_update(root, hermes_home=home, run=runner)

    assert not result.ok and result.exit_code == 6
    assert "strict code-signature verification" in result.message
    assert executable.read_text() == "previous app"
    assert git(root, "rev-parse", "HEAD") == before
    assert not (home / "logs/update_receipts/latest.json").exists()
