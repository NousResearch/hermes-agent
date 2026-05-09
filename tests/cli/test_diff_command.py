"""Tests for the /diff command — shows git changes in working directory."""

import subprocess
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.commands import resolve_command


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "session-diff-test"
    cli._pending_input = MagicMock()
    cli._status_bar_visible = True
    cli.model = "openai/gpt-4o"
    cli.provider = "openai"
    return cli


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_diff_command_registered():
    cmd = resolve_command("diff")
    assert cmd is not None
    assert cmd.cli_only is True


def test_diff_command_has_stat_args_hint():
    cmd = resolve_command("diff")
    assert cmd.args_hint == "[--stat]"


# ---------------------------------------------------------------------------
# process_command routing
# ---------------------------------------------------------------------------


def test_process_command_diff_dispatches():
    cli = _make_cli()
    with patch.object(cli, "_handle_diff_command", create=True) as mock:
        result = cli.process_command("/diff")
    assert result is True
    mock.assert_called_once_with("/diff")


# ---------------------------------------------------------------------------
# _handle_diff_command behaviour
# ---------------------------------------------------------------------------


def _fake_run(outcomes):
    """Return a side_effect that yields subprocess results from *outcomes* in order."""
    call_count = {"n": 0}

    def _run(cmd, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        result = outcomes[n] if n < len(outcomes) else MagicMock(returncode=0, stdout="", stderr="")
        if isinstance(result, Exception):
            raise result
        return result

    return _run


def _mock_proc(stdout="", returncode=0):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


def test_not_in_git_repo_prints_message():
    cli = _make_cli()
    printed = []
    not_a_repo = subprocess.CalledProcessError(128, ["git", "rev-parse"])
    with (
        patch("subprocess.run", side_effect=_fake_run([not_a_repo])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
    ):
        cli._handle_diff_command("/diff")

    assert any("not a git" in p.lower() for p in printed)


def test_git_not_installed_prints_message():
    cli = _make_cli()
    printed = []
    with (
        patch("subprocess.run", side_effect=_fake_run([FileNotFoundError("git")])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
    ):
        cli._handle_diff_command("/diff")

    assert any("not installed" in p.lower() for p in printed)


def test_both_staged_and_unstaged_full_diffs_shown():
    cli = _make_cli()
    printed = []
    run_calls = []

    inside_wt = _mock_proc()
    diff_stat = _mock_proc(stdout=" foo.py | 2 ++\n 1 file changed")
    staged_stat = _mock_proc(stdout=" bar.py | 1 +\n 1 file changed")
    staged_full = _mock_proc(stdout="diff --git a/bar.py b/bar.py\n+staged-line")
    unstaged_full = _mock_proc(stdout="diff --git a/foo.py b/foo.py\n+unstaged-line")

    def _tracking_run(cmd, **kwargs):
        run_calls.append(list(cmd))
        if "rev-parse" in cmd:
            return inside_wt
        if "--stat" in cmd and "--cached" not in cmd:
            return diff_stat
        if "--stat" in cmd and "--cached" in cmd:
            return staged_stat
        if cmd[:3] == ["git", "diff", "--cached"]:
            return staged_full
        if cmd[:2] == ["git", "diff"]:
            return unstaged_full
        return _mock_proc()

    with (
        patch("subprocess.run", side_effect=_tracking_run),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
        patch("cli._rich_text_from_ansi", side_effect=lambda t: t),
        patch.object(cli, "_console_print", create=True) as console_print,
    ):
        cli._handle_diff_command("/diff")

    cached_full_calls = [
        c for c in run_calls
        if c[:3] == ["git", "diff", "--cached"] and "--stat" not in c
    ]
    unstaged_full_calls = [
        c for c in run_calls
        if c[:2] == ["git", "diff"] and "--cached" not in c and "--stat" not in c
    ]
    assert cached_full_calls, "expected `git diff --cached` for staged full diff"
    assert unstaged_full_calls, "expected `git diff` for unstaged full diff"

    headers = " ".join(printed).lower()
    assert "staged" in headers and "unstaged" in headers

    rendered = [str(call.args[0]) for call in console_print.call_args_list]
    assert any("staged-line" in r for r in rendered)
    assert any("unstaged-line" in r for r in rendered)


def test_no_changes_prints_no_changes():
    cli = _make_cli()
    printed = []
    inside_wt = _mock_proc()
    no_diff = _mock_proc(stdout="")
    no_staged = _mock_proc(stdout="")
    with (
        patch("subprocess.run", side_effect=_fake_run([inside_wt, no_diff, no_staged])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
    ):
        cli._handle_diff_command("/diff")

    assert any("no changes" in p.lower() for p in printed)


def test_unstaged_changes_shown():
    cli = _make_cli()
    printed = []
    inside_wt = _mock_proc()
    diff_stat = _mock_proc(stdout=" foo.py | 2 ++\n 1 file changed")
    no_staged = _mock_proc(stdout="")
    diff_full = _mock_proc(stdout="diff --git a/foo.py b/foo.py\n+hello")
    with (
        patch("subprocess.run", side_effect=_fake_run([inside_wt, diff_stat, no_staged, diff_full])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
        patch("cli._rich_text_from_ansi", side_effect=lambda t: t),
        patch.object(cli, "_console_print", create=True),
    ):
        cli._handle_diff_command("/diff")

    assert any("unstaged" in p.lower() for p in printed)


def test_staged_changes_shown():
    cli = _make_cli()
    printed = []
    inside_wt = _mock_proc()
    no_diff = _mock_proc(stdout="")
    staged_stat = _mock_proc(stdout=" bar.py | 1 +\n 1 file changed")
    staged_full = _mock_proc(stdout="diff --git a/bar.py b/bar.py\n+world")
    with (
        patch("subprocess.run", side_effect=_fake_run([inside_wt, no_diff, staged_stat, staged_full])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
        patch("cli._rich_text_from_ansi", side_effect=lambda t: t),
        patch.object(cli, "_console_print", create=True),
    ):
        cli._handle_diff_command("/diff")

    assert any("staged" in p.lower() for p in printed)


def test_stat_only_flag_skips_full_diff():
    cli = _make_cli()
    run_calls = []

    def _tracking_run(cmd, **kwargs):
        run_calls.append(list(cmd))
        if "rev-parse" in cmd:
            return _mock_proc()
        return _mock_proc(stdout=" foo.py | 2 ++\n 1 file changed")

    with (
        patch("subprocess.run", side_effect=_tracking_run),
        patch("cli._cprint"),
        patch("cli._rich_text_from_ansi", side_effect=lambda t: t),
        patch.object(cli, "_console_print", create=True),
    ):
        cli._handle_diff_command("/diff --stat")

    full_diff_calls = [
        c for c in run_calls
        if c[:2] == ["git", "diff"] and "--stat" not in c and "--cached" not in c
    ]
    assert full_diff_calls == [], (
        f"Full diff should not run with --stat flag, got: {full_diff_calls}"
    )


def test_timeout_prints_message():
    cli = _make_cli()
    printed = []
    inside_wt = _mock_proc()
    with (
        patch("subprocess.run", side_effect=_fake_run([inside_wt, subprocess.TimeoutExpired("git", 10)])),
        patch("cli._cprint", side_effect=lambda t: printed.append(t)),
    ):
        cli._handle_diff_command("/diff")

    assert any("timed out" in p.lower() for p in printed)


# ---------------------------------------------------------------------------
# Source-inspection: _handle_diff_command wired in process_command
# ---------------------------------------------------------------------------


def test_diff_routing_in_process_command_source():
    import inspect
    import cli as cli_mod

    src = inspect.getsource(cli_mod.HermesCLI.process_command)
    assert ('"diff"' in src or "'diff'" in src), (
        "process_command must route canonical == 'diff'"
    )
    assert "_handle_diff_command" in src
