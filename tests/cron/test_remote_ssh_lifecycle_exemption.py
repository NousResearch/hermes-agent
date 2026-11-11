"""Restarting a gateway on a different host over ssh is not a self-restart.

The lifecycle guard exists to stop a job from killing the process it is running
inside (the supervisor then revives it and the job fires again). An ``ssh`` to
another machine terminates a different process, so that command must be allowed.
An ssh whose target is this machine, or cannot be determined, stays blocked.
"""

import pytest

from cron.lifecycle_guard import (
    GatewayLifecycleBlocked,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command_or_referenced_script as guard,
)


def test_ssh_to_a_remote_host_may_restart_that_hosts_gateway():
    command = "ssh ops@gateway.example.com 'hermes gateway restart'"
    assert guard(command) is False
    check_gateway_lifecycle(command)  # must not raise


def test_ssh_option_values_are_not_mistaken_for_the_target_host():
    command = "ssh -i ~/.ssh/id_ed25519 -p 22 ops@10.1.2.3 systemctl restart hermes-gateway"
    assert guard(command) is False


def test_ssh_to_localhost_still_blocks_a_self_restart():
    with pytest.raises(GatewayLifecycleBlocked):
        check_gateway_lifecycle("ssh user@localhost hermes gateway restart")


def test_a_local_lifecycle_command_beside_a_remote_ssh_stays_blocked():
    command = "ssh ops@other.example.com true; hermes gateway restart"
    with pytest.raises(GatewayLifecycleBlocked):
        check_gateway_lifecycle(command)


def test_an_ssh_piped_into_a_shell_stays_blocked():
    # The payload is not provably remote once it is handed to another interpreter.
    command = "echo 'hermes gateway restart' | ssh ops@other.example.com sh"
    with pytest.raises(GatewayLifecycleBlocked):
        check_gateway_lifecycle(command)
