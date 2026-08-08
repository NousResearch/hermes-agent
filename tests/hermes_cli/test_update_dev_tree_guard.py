"""`hermes update` asks before it touches a non-managed git checkout.

The update flow stashes local changes and moves the checkout to the
update branch. That is correct only at a managed install root. These
tests build real git checkouts and drive _cmd_update_impl far enough to
hit the guard; the guard exits before any git mutation, so the tests
assert the tree afterward.
"""

import subprocess

import pytest

import hermes_cli.update_cmd as update_cmd
from hermes_cli.update_cmd import _cmd_update_impl


def _git(cwd, *args):
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def dev_checkout(tmp_path):
    """A real git checkout on a feature branch with a dirty file."""
    repo = tmp_path / "src" / "hermes-agent"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "f.txt").write_text("original\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/x")
    (repo / "f.txt").write_text("uncommitted work\n")
    return repo


class _Args:
    yes = False
    branch = None
    force = False
    force_venv = False
    check = False


def _patch_project_root(monkeypatch, root):
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)


class TestDevTreeGuard:
    def test_non_interactive_refuses_and_leaves_the_tree_alone(
        self, dev_checkout, monkeypatch, capsys
    ):
        _patch_project_root(monkeypatch, dev_checkout)
        # Non-interactive: stdin is not a tty under pytest already, but make
        # it explicit so the test does not depend on the runner.
        monkeypatch.setattr(update_cmd.sys.stdin, "isatty", lambda: False, raising=False)

        with pytest.raises(SystemExit) as exc:
            _cmd_update_impl(_Args(), gateway_mode=False)

        assert exc.value.code == 3
        out = capsys.readouterr().out
        assert "not the managed install" in out
        assert "git pull" in out
        # The tree is untouched: same branch, same dirty content, no stash.
        assert _git(dev_checkout, "rev-parse", "--abbrev-ref", "HEAD") == "feature/x"
        assert (dev_checkout / "f.txt").read_text() == "uncommitted work\n"
        assert _git(dev_checkout, "stash", "list") == ""

    def test_interactive_no_cancels(self, dev_checkout, monkeypatch, capsys):
        _patch_project_root(monkeypatch, dev_checkout)
        monkeypatch.setattr(update_cmd.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(update_cmd.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")

        with pytest.raises(SystemExit) as exc:
            _cmd_update_impl(_Args(), gateway_mode=False)

        assert exc.value.code == 3
        assert "canceled" in capsys.readouterr().out
        assert _git(dev_checkout, "rev-parse", "--abbrev-ref", "HEAD") == "feature/x"

    def test_gateway_mode_refuses(self, dev_checkout, monkeypatch, capsys):
        _patch_project_root(monkeypatch, dev_checkout)

        with pytest.raises(SystemExit) as exc:
            _cmd_update_impl(_Args(), gateway_mode=True)

        assert exc.value.code == 3
        assert "--yes" in capsys.readouterr().out

    def test_yes_proceeds_past_the_guard(self, dev_checkout, monkeypatch, capsys):
        _patch_project_root(monkeypatch, dev_checkout)
        args = _Args()
        args.yes = True
        # Stop the update right after the guard: the pre-update backup is
        # the first thing past it. Raising there proves the guard let the
        # flow continue without running the real update.
        sentinel = RuntimeError("past-the-guard")

        def _boom(_args):
            raise sentinel

        monkeypatch.setattr(update_cmd._m(), "_run_pre_update_backup", _boom)

        with pytest.raises(RuntimeError, match="past-the-guard"):
            _cmd_update_impl(args, gateway_mode=False)

        assert "Continuing (--yes)" in capsys.readouterr().out

    def test_managed_root_needs_no_question(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        repo = home / "hermes-agent"
        repo.mkdir(parents=True)
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        (repo / "f.txt").write_text("x\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initial")
        _patch_project_root(monkeypatch, repo)

        sentinel = RuntimeError("past-the-guard")

        def _boom(_args):
            raise sentinel

        monkeypatch.setattr(update_cmd._m(), "_run_pre_update_backup", _boom)

        with pytest.raises(RuntimeError, match="past-the-guard"):
            _cmd_update_impl(_Args(), gateway_mode=False)

        assert "not the managed install" not in capsys.readouterr().out
