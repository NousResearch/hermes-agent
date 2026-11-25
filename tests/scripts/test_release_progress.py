"""Release progress logging: stderr only, grouped, and commands stay readable."""
import subprocess

import pytest

from scripts.releases import progress


def test_logged_run_logs_each_command_and_its_outcome(capsys):
    run = progress.logged_run(lambda argv: "a\nb")
    assert run(["git", "ls-remote", "origin"]) == "a\nb"
    err = capsys.readouterr().err
    assert "  $ git ls-remote origin" in err
    assert "ok in" in err and "(2 lines)" in err
    # Wrapping twice does not double the logs.
    assert progress.logged_run(run) is run


def test_logged_run_logs_a_failure_and_reraises(capsys):
    def fail(argv):
        raise subprocess.CalledProcessError(128, argv)

    with pytest.raises(subprocess.CalledProcessError):
        progress.logged_run(fail)(["git", "push", "origin", "refs/tags/v1.2.3"])
    assert "failed after" in capsys.readouterr().err


def test_long_arguments_are_cut_and_newlines_escaped():
    shown = progress.describe_command(["gh", "api", "--raw-field", "body=" + "x" * 500 + "\nmore"])
    assert "…(+" in shown and "\n" not in shown
    assert len(shown) < 260


def test_group_balances_and_reports_failure(capsys):
    with pytest.raises(RuntimeError):
        with progress.group("Advance"):
            raise RuntimeError("nope")
    err = capsys.readouterr().err
    assert err.startswith("::group::Advance\n")
    assert "✗ Advance failed after" in err and "RuntimeError: nope" in err
    assert err.rstrip().endswith("::endgroup::")
