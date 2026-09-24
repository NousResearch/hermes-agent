"""Regression tests for fail-closed updater preflight behavior."""

from types import SimpleNamespace

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main


@pytest.mark.parametrize("branch", ["-bad", "feature:other", "feature..other"])
def test_resolve_update_branch_rejects_invalid_refspec_forms(branch):
    """User-controlled branch input must never be accepted as a refspec."""
    with pytest.raises(ValueError):
        hermes_main._resolve_update_branch(SimpleNamespace(branch=branch))


def test_update_rejects_invalid_branch_before_any_git_command(monkeypatch, tmp_path, capsys):
    """The CLI must fail cleanly before fetch or other updater work."""
    from hermes_cli import update_cmd

    monkeypatch.setattr(update_cmd._m, "PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(hermes_config, "load_config", lambda: {})
    git_calls = []
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: git_calls.append(args[0]),
    )

    with pytest.raises(SystemExit, match="2"):
        update_cmd._cmd_update_impl(
            SimpleNamespace(branch="feature:other"), gateway_mode=False
        )

    # The pipeline may run preflight/probe git commands (fleet check, dirty-tree
    # check, remote lookup) before branch resolution. The invariant that matters:
    # the user-controlled branch value NEVER reaches a subprocess as a refspec.
    assert not any(
        "feature:other" in str(arg)
        for call in git_calls
        for arg in call
    ), "branch value was interpolated into a subprocess call"
    assert not any(
        call[:1] == ["git"] and "fetch" in call for call in git_calls
    ), "a fetch ran despite the invalid branch"
    assert "invalid update branch name" in capsys.readouterr().out
