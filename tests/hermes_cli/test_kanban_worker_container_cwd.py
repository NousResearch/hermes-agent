"""TERMINAL_CWD bridging for kanban workers spawned by the in-gateway dispatcher.

Workers inherit ``_HERMES_GATEWAY=1`` from the gateway, which normally stops the
CLI config loader from exporting ``terminal.cwd``. The dispatcher pins
TERMINAL_CWD to the task's host workspace, which a container backend cannot use,
so a worker on a non-local backend must export its own profile's cwd instead.
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli.cli_config_load import _mirror_config_to_env

HOST_WORKSPACE = "/Users/someone/.hermes/kanban/workspaces/t_1"


def _mirror(terminal: dict, **env: str) -> str | None:
    base = {"_HERMES_GATEWAY": "1", "TERMINAL_CWD": HOST_WORKSPACE}
    base.update(env)
    with patch.dict(os.environ, base, clear=False):
        _mirror_config_to_env({"terminal": dict(terminal)}, True)
        return os.environ.get("TERMINAL_CWD")


def test_container_kanban_worker_exports_its_profile_cwd():
    cwd = _mirror({"backend": "docker", "cwd": "/workspace"}, HERMES_KANBAN_TASK="t_1")
    assert cwd == "/workspace"


def test_gateway_process_keeps_bridged_cwd():
    """Without HERMES_KANBAN_TASK this is the gateway itself: its bridge owns the value."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_KANBAN_TASK", None)
        cwd = _mirror({"backend": "docker", "cwd": "/workspace"})
    assert cwd == HOST_WORKSPACE


@pytest.mark.parametrize("backend", ["local", None])
def test_local_kanban_worker_keeps_dispatcher_pin(backend):
    """A local-backend worker can use the host workspace, so the dispatcher's pin stays."""
    terminal = {"cwd": "/somewhere/else"}
    if backend:
        terminal["backend"] = backend
    cwd = _mirror(terminal, HERMES_KANBAN_TASK="t_1")
    assert cwd == HOST_WORKSPACE


def test_container_kanban_worker_with_placeholder_cwd_is_untouched():
    """A placeholder cwd is dropped for container backends; nothing to export."""
    cwd = _mirror({"backend": "docker", "cwd": "."}, HERMES_KANBAN_TASK="t_1")
    assert cwd == HOST_WORKSPACE


def test_kanban_worker_keeps_gateway_marker():
    """The exception only changes TERMINAL_CWD; the gateway-lifecycle guard still sees the marker."""
    with patch.dict(os.environ, {"_HERMES_GATEWAY": "1", "HERMES_KANBAN_TASK": "t_1"}, clear=False):
        _mirror_config_to_env({"terminal": {"backend": "docker", "cwd": "/workspace"}}, True)
        assert os.environ.get("_HERMES_GATEWAY") == "1"
