"""The lifecycle guard must refuse an in-tree `hermes update`, not only a gateway restart.

`_GATEWAY_LIFECYCLE_PATTERN` branch A matched ``hermes gateway restart|stop|uninstall``,
so ``hermes update`` -- which drains the gateway, SIGKILLs it, replaces the code tree and
migrates ``config.yaml`` from inside the process it is killing -- passed the guard. That is
strictly more destructive than the restart that *is* refused (#144547).

The verb must be recognised with and without the ``gateway`` token, and through the shell
wrappers the payload walk already covers (``bash -c``, ``ssh host '...'``). A hyphen must
NOT count as the separator before the verb, or ``hermes-update-*.md`` filenames become
false positives.
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


@pytest.mark.parametrize("command", [
    "hermes update",
    "hermes update --backup -y",
    "hermes -c update",
    "hermes upgrade",
    "cd ~ && hermes update -y",
    "nohup hermes update --backup -y &",
    "bash -c 'hermes update'",
    "ssh host 'hermes update'",
    "sh -c \"hermes update\"",
    "set -e; hermes update",
])
def test_in_tree_update_is_refused(command, tmp_path):
    assert guard(command, cwd=str(tmp_path)) is True


@pytest.mark.parametrize("command", [
    "hermes gateway restart",
    "hermes gateway stop",
    "hermes gateway uninstall",
    "systemctl --user restart hermes-gateway",
])
def test_existing_restart_coverage_is_unchanged(command, tmp_path):
    assert guard(command, cwd=str(tmp_path)) is True


@pytest.mark.parametrize("command", [
    "git status",
    "cat ~/.hermes/logs/update.log",
    "grep -rn update ~/.hermes/logs",
    "hermes --version",
    "hermes model",
    "hermes gateway start",
    "python3 script.py --update-flag",
    "cat kb/rules/hermes-update-guard.md",
])
def test_unrelated_commands_are_not_blocked(command, tmp_path):
    """`update` inside an argument, a path or a flag is not an invocation."""
    assert guard(command, cwd=str(tmp_path)) is False
