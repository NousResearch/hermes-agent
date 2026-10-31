"""Windows Scheduled Task ops on the gateway task are gateway-lifecycle commands.

``hermes_cli/gateway_windows.py`` installs and supervises the gateway as a
Scheduled Task named ``Hermes_Gateway`` / ``Hermes_Gateway_<profile>``
(``_TASK_NAME_DEFAULT``). Stopping or deleting that task is the same operation
the guard already hard-blocks on the other two platforms:

    macOS    launchctl bootout gui/501/ai.hermes.gateway   (Branch B)
    Linux    systemctl stop hermes-gateway                 (Branch C)
    Windows  schtasks /end /tn Hermes_Gateway_alice         (Branch E, here)

Before Branch E the Windows forms returned False, so the identical intent was
hard-blocked on macOS/Linux and silently allowed on Windows, leaving only the
bypassable approval layer (``tools/approval.py``, skipped on ``force=True``) —
the same asymmetry #80260 closed for ``bootout``/``remove``/``disable``.

These are pure string-predicate tests: no Scheduled Task is created or touched,
so they run identically on every host.
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command


@pytest.mark.parametrize("command", [
    # schtasks, op flag before the task name
    "schtasks /end /tn Hermes_Gateway",
    "schtasks /end /tn Hermes_Gateway_alice",
    "schtasks /delete /tn Hermes_Gateway /f",
    # ...and after it: flag order is not fixed on the command line
    "schtasks /tn Hermes_Gateway_alice /end",
    "schtasks /tn Hermes_Gateway /delete /f",
    # `/change /disable` is what makes a stop durable across boots
    "schtasks /change /disable /tn Hermes_Gateway",
    "schtasks /tn Hermes_Gateway /change /disable",
    # PowerShell cmdlets, label as an argument
    "Stop-ScheduledTask -TaskName Hermes_Gateway_alice",
    "Unregister-ScheduledTask -TaskName Hermes_Gateway -Confirm:$false",
    "Disable-ScheduledTask -TaskName Hermes_Gateway",
    # ...and label from a pipeline, where it precedes the cmdlet
    "Get-ScheduledTask -TaskName Hermes_Gateway* | Stop-ScheduledTask",
    # case-insensitive, and the hyphen/dot spellings of the label
    "SCHTASKS /END /TN HERMES_GATEWAY",
    "schtasks /end /tn hermes-gateway",
    "schtasks /end /tn hermes.gateway",
])
def test_windows_scheduled_task_gateway_ops_are_blocked(command):
    assert contains_gateway_lifecycle_command(command), f"should block: {command!r}"


@pytest.mark.parametrize("command", [
    # Unrelated tasks stay usable — the label anchor carries the whole decision.
    "schtasks /end /tn BackupJob",
    "schtasks /delete /tn NightlyIndex /f",
    "Stop-ScheduledTask -TaskName SomeOtherTask",
    # Read-only inspection is not a lifecycle op.
    "schtasks /query /tn Hermes_Gateway",
    "Get-ScheduledTask -TaskName Hermes_Gateway",
    # Starting is benign, matching Branch A's exclusion of `hermes gateway start`.
    "schtasks /run /tn Hermes_Gateway",
    "schtasks /create /tn Hermes_Gateway /tr hermes.exe /sc onlogon",
    # `/change` alone (e.g. re-pointing the run-as user) is not a stop.
    "schtasks /change /tn Hermes_Gateway /ru SYSTEM",
])
def test_benign_and_unrelated_scheduled_task_commands_are_not_blocked(command):
    assert not contains_gateway_lifecycle_command(command), f"should allow: {command!r}"


def test_windows_branch_matches_the_other_platforms_for_the_same_intent():
    """Stop-the-gateway must be blocked on all three platforms, not two."""
    assert contains_gateway_lifecycle_command("launchctl bootout gui/501/ai.hermes.gateway")
    assert contains_gateway_lifecycle_command("systemctl stop hermes-gateway")
    assert contains_gateway_lifecycle_command("schtasks /end /tn Hermes_Gateway")
