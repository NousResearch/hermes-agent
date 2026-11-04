"""Lifecycle guard: a lifecycle phrase that is only DATA to a non-executing command is not blocked.

The terminal-path scan already exempts search-pattern arguments (grep, rg, sqlite3, ...). Printed
text (``echo``/``printf``) and a git message (``git commit -m "... gateway restart ..."``) are
the same class: the shell never executes them, so blocking them only teaches people to route
around the guard. Execution smuggled into those arguments stays blocked.
"""
import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as blocked

CMD = "hermes gateway restart"


@pytest.mark.parametrize("command", [
    f"echo '{CMD}'",
    f'echo "{CMD}"',
    f'printf "%s\\n" "{CMD}"',
    f'git commit -m "docs: run {CMD} after deploy"',
    f'git commit -m "fix: x\n\nRun {CMD} after deploy."',
    f'git tag -a v1 -m "{CMD} documented"',
    f'git stash push -m "before {CMD}"',
    f'grep "{CMD}" gateway.log',
])
def test_lifecycle_phrase_as_data_is_allowed(command):
    assert not blocked(command), command


@pytest.mark.parametrize("command", [
    CMD,
    f'git commit -m "$(hermes gateway restart)"',
    f'git commit -m "`hermes gateway restart`"',
    f'echo "$(hermes gateway restart)"',
    f'git commit -m "x" && {CMD}',
    f'echo ok; {CMD}',
    f'echo ok\n{CMD}',
    f'bash -c "{CMD}"',
    f'echo "{CMD}" | sh',
    f'echo "{CMD}" | bash',
    f'git -c core.pager="{CMD}" log',
    f'git push origin {CMD}',
    f'echo "unbalanced\n{CMD}',
])
def test_executed_lifecycle_command_stays_blocked(command):
    assert blocked(command), command
