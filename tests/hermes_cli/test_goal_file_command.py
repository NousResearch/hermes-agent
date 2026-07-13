from queue import Queue
from types import SimpleNamespace

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _GoalManager:
    def __init__(self):
        self.calls = []

    def set(self, goal, *, contract=None):
        self.calls.append((goal, contract))
        return SimpleNamespace(
            goal=goal,
            max_turns=10,
            has_contract=lambda: contract is not None,
            contract=contract,
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
    assert any("could not read" in line for line in output)


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
