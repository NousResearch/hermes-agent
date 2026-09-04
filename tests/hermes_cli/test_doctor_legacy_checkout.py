"""Tests for the legacy-desktop-checkout report in hermes doctor."""

import json
import subprocess

import pytest

import hermes_cli.doctor as doctor
from hermes_cli.doctor import check_legacy_desktop_checkout


def _git(cwd, *args):
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def embedded_context(tmp_path, monkeypatch):
    """Running tree = sealed desktop-app; legacy checkout at the managed root."""
    bundle = tmp_path / "bundle" / "repo"
    bundle.mkdir(parents=True)
    # The code-scoped stamp (hermes_cli.steward.BUILD_INFO_NAME).
    (bundle / "install-stamp.json").write_text(
        json.dumps({"commit": "a" * 40, "distribution": "desktop-app", "updateMechanism": "electron-updater"})
    )
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", bundle)

    home = tmp_path / ".hermes"
    checkout = home / "hermes-agent"
    checkout.mkdir(parents=True)
    _git(checkout, "init", "-b", "main")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "test")
    (checkout / "f.txt").write_text("x\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "initial")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    return checkout


class TestLegacyDesktopCheckout:
    def test_pristine_checkout_gets_a_deletion_suggestion(self, embedded_context, capsys):
        check_legacy_desktop_checkout()
        out = capsys.readouterr().out
        assert "Unused checkout" in out
        assert "rm -rf" in out

    def test_dirty_checkout_warns_and_suggests_nothing(self, embedded_context, capsys):
        (embedded_context / "f.txt").write_text("local work\n")
        check_legacy_desktop_checkout()
        out = capsys.readouterr().out
        assert "holds local work" in out
        assert "rm -rf" not in out
        # And the tree is untouched.
        assert (embedded_context / "f.txt").read_text() == "local work\n"

    def test_branched_checkout_counts_as_local_work(self, embedded_context, capsys):
        _git(embedded_context, "checkout", "-b", "feature/y")
        check_legacy_desktop_checkout()
        out = capsys.readouterr().out
        assert "holds local work" in out
        assert "rm -rf" not in out

    def test_stashed_checkout_counts_as_local_work(self, embedded_context, capsys):
        (embedded_context / "f.txt").write_text("stash me\n")
        _git(embedded_context, "stash")
        check_legacy_desktop_checkout()
        out = capsys.readouterr().out
        assert "holds local work" in out
        assert "rm -rf" not in out

    def test_silent_without_a_checkout(self, embedded_context, tmp_path, monkeypatch, capsys):
        import shutil
        import stat

        def _force_rm(func, path, _exc):
            # Windows: git object files are read-only; clear the bit and retry.
            import os

            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(embedded_context, onerror=_force_rm)
        check_legacy_desktop_checkout()
        assert capsys.readouterr().out == ""

    def test_silent_when_running_from_a_git_checkout(self, embedded_context, monkeypatch, capsys):
        # A git-managed install (dev tree or ejected) is not embedded; the
        # checkout at the managed root might BE the running tree.
        import hermes_cli.main as hermes_main

        monkeypatch.setattr(hermes_main, "PROJECT_ROOT", embedded_context)
        check_legacy_desktop_checkout()
        assert capsys.readouterr().out == ""
