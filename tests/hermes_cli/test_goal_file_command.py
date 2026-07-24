"""Tests for ``/goal --file <path>`` and the shared goal-input resolver.

Covers the Classic CLI handler, the pure ``_parse_goal_file_arg`` syntax
boundary (Windows backslashes preserved on any host), and the shared
``resolve_goal_input`` resolver used by the Classic CLI, TUI/Desktop
backend, and messaging gateway. Real temporary files back the success,
directory, blank, and invalid-UTF-8 cases; an unreadable file is tested
by mocking ``Path.read_text`` (``chmod 000`` is unreliable under root).
"""
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.goals import GoalContract, GoalInputError, _parse_goal_file_arg, resolve_goal_input


class _GoalManager:
    """Minimal stand-in for GoalManager that records set() calls."""

    def __init__(self):
        self.calls = []

    def set(self, goal, *, contract=None):
        self.calls.append((goal, contract))
        # Mirror the real has_contract(): only truthy when a non-empty
        # contract was supplied, so the handler's rendering branch is
        # exercised accurately.
        has = contract is not None and not contract.is_empty()
        return SimpleNamespace(
            goal=goal,
            max_turns=10,
            has_contract=lambda: has,
            contract=contract if has else GoalContract(),
        )


class _CLI(CLICommandsMixin):
    def __init__(self, manager):
        self._manager = manager
        self._pending_input = Queue()

    def _get_goal_manager(self):
        return self._manager


def _capture_output(monkeypatch):
    output = []
    monkeypatch.setattr("cli._cprint", output.append)
    return output


# ── pure syntax boundary (_parse_goal_file_arg) ───────────────────────


def test_parse_goal_file_arg_returns_none_for_ordinary_text():
    assert _parse_goal_file_arg("build a rocket") is None
    assert _parse_goal_file_arg("") is None
    assert _parse_goal_file_arg(None) is None


def test_parse_goal_file_arg_extracts_path_after_token():
    assert _parse_goal_file_arg("--file goal.txt") == "goal.txt"
    assert _parse_goal_file_arg("--file  goal.txt") == "goal.txt"


def test_parse_goal_file_arg_preserves_native_windows_backslashes():
    # The pure boundary must keep backslashes byte-for-byte so POSIX CI
    # can assert Windows-style paths survive without host path semantics.
    win = r"--file C:\Users\Alice\goal.txt"
    assert _parse_goal_file_arg(win) == r"C:\Users\Alice\goal.txt"


def test_parse_goal_file_arg_strips_one_pair_of_outer_quotes():
    assert _parse_goal_file_arg('--file "my goal.txt"') == "my goal.txt"
    assert _parse_goal_file_arg("--file 'my goal.txt'") == "my goal.txt"
    # Internal backslashes inside quotes survive.
    assert _parse_goal_file_arg(r'--file "C:\my goal.txt"') == r"C:\my goal.txt"


def test_parse_goal_file_arg_allows_unquoted_path_with_spaces():
    # shlex would split this; the remainder parser keeps it whole.
    assert _parse_goal_file_arg("--file my goal.txt") == "my goal.txt"


def test_parse_goal_file_arg_missing_path_raises():
    with pytest.raises(GoalInputError, match="Usage"):
        _parse_goal_file_arg("--file")
    with pytest.raises(GoalInputError, match="Usage"):
        _parse_goal_file_arg("--file   ")


def test_parse_goal_file_arg_mismatched_outer_quotes_raises():
    with pytest.raises(GoalInputError, match="mismatched quotes"):
        _parse_goal_file_arg('--file "goal.txt')


def test_parse_goal_file_arg_mismatched_quote_types_raises():
    # Different opening/closing quote chars is a typo, not a valid path.
    with pytest.raises(GoalInputError, match="mismatched quotes"):
        _parse_goal_file_arg("--file \"goal.txt'")
    with pytest.raises(GoalInputError, match="mismatched quotes"):
        _parse_goal_file_arg("--file 'goal.txt\"")


def test_parse_goal_file_arg_empty_quoted_path_raises_usage():
    # ``--file ""`` / ``--file ''`` is a missing path, not an empty path that
    # later surfaces as a confusing "not a regular file" error.
    with pytest.raises(GoalInputError, match="Usage"):
        _parse_goal_file_arg('--file ""')
    with pytest.raises(GoalInputError, match="Usage"):
        _parse_goal_file_arg("--file ''")


# ── resolver: file loading ────────────────────────────────────────────


def test_resolve_goal_input_inline_parses_contract():
    headline, contract = resolve_goal_input("Ship feature\nverify: pytest -q")
    assert headline == "Ship feature"
    assert contract.verification == "pytest -q"


def test_resolve_goal_input_file_utf8_with_spaces(tmp_path):
    goal_path = tmp_path / "release goal.txt"
    goal_path.write_text("Ship café support\n\nverify: pytest -q\n", encoding="utf-8")
    headline, contract = resolve_goal_input(f'--file "{goal_path}"')
    assert headline == "Ship café support"
    assert contract.verification == "pytest -q"


def test_resolve_goal_input_relative_path_uses_explicit_cwd(tmp_path):
    (tmp_path / "goal.txt").write_text("local goal", encoding="utf-8")
    headline, contract = resolve_goal_input("--file goal.txt", cwd=str(tmp_path))
    assert headline == "local goal"
    assert contract.is_empty()


def test_resolve_goal_input_relative_cwd_missing_on_host_rejected(tmp_path):
    # SSH/Docker case: the session cwd exists only inside the terminal
    # backend, not on the Hermes host. A relative path must NOT silently
    # fall back to the process cwd.
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(GoalInputError, match="absolute path"):
        resolve_goal_input("--file goal.txt", cwd=str(bogus))


def test_resolve_goal_input_tilde_expansion(tmp_path, monkeypatch):
    home = tmp_path / "alice"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / "goal.txt").write_text("home goal", encoding="utf-8")
    headline, _ = resolve_goal_input("--file ~/goal.txt")
    assert headline == "home goal"


def test_resolve_goal_input_bad_user_tilde_raises_input_error():
    # ``~not-a-real-user`` makes Path.expanduser() raise RuntimeError (not
    # OSError). That must surface as a GoalInputError so the handlers'
    # GoalInputError catch shows a readable message instead of an uncaught
    # traceback. The exact unknown user is host-dependent; pick one that
    # resolves to no home dir.
    with pytest.raises(GoalInputError, match="could not expand"):
        resolve_goal_input("--file ~this-user-definitely-does-not-exist-xyz/goal.txt")


def test_resolve_goal_input_contract_extracted_from_file(tmp_path):
    goal_path = tmp_path / "goal.txt"
    goal_path.write_text(
        "Migrate auth to JWT\n"
        "verify: the auth suite passes\n"
        "constraints: keep /login shape\n"
        "boundaries: services/auth only\n"
        "stop when: schema change needs sign-off\n",
        encoding="utf-8",
    )
    headline, contract = resolve_goal_input(f"--file {goal_path}")
    assert headline == "Migrate auth to JWT"
    assert contract.verification == "the auth suite passes"
    assert contract.constraints == "keep /login shape"
    assert contract.boundaries == "services/auth only"
    assert contract.stop_when == "schema change needs sign-off"


def test_resolve_goal_input_file_content_resembling_subcommand_is_data(tmp_path):
    for content in ("status", "clear", "draft something", "--file another.txt"):
        goal_path = tmp_path / f"{abs(hash(content))}.txt"
        goal_path.write_text(content, encoding="utf-8")
        headline, contract = resolve_goal_input(f"--file {goal_path}")
        assert headline == content
        assert contract.is_empty()


def test_resolve_goal_input_allow_file_false_rejects_before_io(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SHOULD NOT BE READ", encoding="utf-8")
    with pytest.raises(GoalInputError, match="messaging platforms"):
        resolve_goal_input(f"--file {secret}", allow_file=False)
    # The file is never opened: assert by proving content stays unread is
    # implicit — the resolver would have to read it to return the goal,
    # and instead it raised.


@pytest.mark.parametrize("label,content", [
    ("missing", None),
    ("blank", "  \n\n "),
])
def test_resolve_goal_input_missing_and_blank_files(tmp_path, label, content):
    if content is None:
        path = tmp_path / "missing.txt"
        with pytest.raises(GoalInputError, match="file not found"):
            resolve_goal_input(f"--file {path}")
    else:
        path = tmp_path / "blank.txt"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(GoalInputError, match="empty"):
            resolve_goal_input(f"--file {path}")


def test_resolve_goal_input_directory_rejected(tmp_path):
    with pytest.raises(GoalInputError, match="not a regular file"):
        resolve_goal_input(f"--file {tmp_path}")


def test_resolve_goal_input_invalid_utf8_rejected(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(GoalInputError, match="not valid UTF-8"):
        resolve_goal_input(f"--file {bad}")


def test_resolve_goal_input_unreadable_file_rejected(tmp_path):
    goal_path = tmp_path / "locked.txt"
    goal_path.write_text("goal", encoding="utf-8")
    # chmod 000 is unreliable under root / Windows / some CI; mock read_text.
    with patch.object(__import__("pathlib").Path, "read_text", side_effect=PermissionError("denied")):
        with pytest.raises(GoalInputError, match="could not read"):
            resolve_goal_input(f"--file {goal_path}")


# ── Classic CLI handler ───────────────────────────────────────────────


def test_goal_file_loads_quoted_utf8_path_and_starts_goal(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)
    goal_path = tmp_path / "release goal.txt"
    goal_path.write_text("Ship café support\n\nverify: pytest -q\n", encoding="utf-8")

    cli._handle_goal_command(f'/goal --file "{goal_path}"')

    assert len(manager.calls) == 1
    goal, contract = manager.calls[0]
    assert goal == "Ship café support"
    assert contract is not None
    assert contract.verification == "pytest -q"
    assert cli._pending_input.get_nowait() == goal
    assert any("Goal set" in line for line in output)


def test_goal_file_content_is_not_reinterpreted_as_subcommand(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    _capture_output(monkeypatch)
    goal_path = tmp_path / "goal.txt"
    goal_path.write_text("status", encoding="utf-8")

    cli._handle_goal_command(f"/goal --file {goal_path}")

    assert manager.calls[0][0] == "status"
    assert cli._pending_input.get_nowait() == "status"


def test_goal_file_read_error_does_not_replace_goal(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)
    missing = tmp_path / "missing.txt"

    cli._handle_goal_command(f"/goal --file {missing}")

    assert manager.calls == []
    assert cli._pending_input.empty()
    assert any("--file" in line and "file not found" in line for line in output)


def test_goal_file_rejects_empty_file(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)
    goal_path = tmp_path / "empty.txt"
    goal_path.write_text(" \n", encoding="utf-8")

    cli._handle_goal_command(f"/goal --file {goal_path}")

    assert manager.calls == []
    assert cli._pending_input.empty()
    assert any("is empty" in line for line in output)


def test_goal_file_directory_is_atomic(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)

    cli._handle_goal_command(f"/goal --file {tmp_path}")

    assert manager.calls == []
    assert cli._pending_input.empty()
    assert any("not a regular file" in line for line in output)


def test_goal_file_missing_path_token_is_atomic(monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)

    cli._handle_goal_command("/goal --file")

    assert manager.calls == []
    assert cli._pending_input.empty()
    assert any("Usage" in line for line in output)


def test_goal_file_invalid_utf8_is_atomic(tmp_path, monkeypatch):
    manager = _GoalManager()
    cli = _CLI(manager)
    output = _capture_output(monkeypatch)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")

    cli._handle_goal_command(f"/goal --file {bad}")

    assert manager.calls == []
    assert cli._pending_input.empty()
    assert any("UTF-8" in line for line in output)
