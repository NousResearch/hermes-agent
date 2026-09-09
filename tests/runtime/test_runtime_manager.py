"""
Tests for Runtime System and RuntimeManager.
"""

import pytest
from runtime import RuntimeManager, LocalRuntime, SSHRuntime, ContainerRuntime, SandboxRuntime, RemoteRuntime


def test_runtime_manager_defaults():
    mgr = RuntimeManager()
    active = mgr.get_runtime()
    assert active is not None
    assert active.runtime_id == "local_default"
    assert active.is_connected() is True


def test_runtime_registration_and_switching():
    mgr = RuntimeManager()
    ssh = SSHRuntime("ssh_prod", "Production Server", "10.0.0.1")
    mgr.register_runtime(ssh)

    runtimes = mgr.list_runtimes()
    assert len(runtimes) == 2

    switched = mgr.set_active_runtime("ssh_prod")
    assert switched is True
    assert mgr.get_runtime().runtime_id == "ssh_prod"
