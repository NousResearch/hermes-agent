"""Kanban worker MCP overrides must target the server entry the runtime migration registers.

Regression for #111707: dispatcher-owned workers on the codex app-server runtime injected
``mcp_servers.hermes-mcp.env.*`` while the migration writes ``[mcp_servers.hermes-tools]``;
codex then saw an env-only entry with no transport and refused to start.
"""

import subprocess
import tomllib

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, KANBAN_ENV_KEYS, non_dispatcher_owned_context
from agent.transports import codex_app_server as cas
from hermes_cli.codex_runtime_plugin_migration import migrate


class _RecordingPopen:
    commands: list[list[str]] = []

    def __init__(self, cmd, *args, **kwargs):
        type(self).commands.append(list(cmd))
        self.stdin = self.stdout = self.stderr = None
        self.pid = 1
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """Return ``launch(env) -> list[str]`` of the ``mcp_servers.*`` overrides in the worker argv."""
    _RecordingPopen.commands = []
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    for key in (*KANBAN_ENV_KEYS, DELEGATED_CHILD_ENV_MARKER, "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)

    def _launch(env: dict[str, str]) -> list[str]:
        with monkeypatch.context() as ctx:
            for key, value in env.items():
                ctx.setenv(key, value)
            client = cas.CodexAppServerClient(codex_bin="codex", codex_home=str(tmp_path / "codex"))
            client._closed = True
        cmd = _RecordingPopen.commands.pop()
        return [arg for arg in cmd if arg.startswith("mcp_servers.")]

    return _launch


def _migrated_server_names(tmp_path) -> set[str]:
    codex_home = tmp_path / "codex"
    report = migrate({"mcp_servers": {}}, codex_home=codex_home, discover_plugins=False, expose_hermes_tools=True)
    assert not report.errors
    return set(tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))["mcp_servers"])


def test_worker_overrides_target_the_migrated_server(launch, tmp_path):
    """Producer/consumer contract: every ``-c mcp_servers.<name>.env.*`` override the worker
    launcher emits names an entry the migration actually writes to config.toml."""
    overrides = launch({
        "HERMES_KANBAN_TASK": "11111111-1111-4111-8111-111111111111",
        "HERMES_KANBAN_RUN_ID": "42",
        "HERMES_KANBAN_DB": str(tmp_path / "board" / "kanban.db"),
    })
    assert overrides, "dispatcher-owned worker must scope the managed MCP endpoint"
    targeted = {arg.split(".env.", 1)[0].removeprefix("mcp_servers.") for arg in overrides}
    migrated = _migrated_server_names(tmp_path)
    assert targeted <= migrated, f"overrides target {targeted - migrated}, which codex has no transport for"
    assert any(arg.startswith(f"mcp_servers.{next(iter(targeted))}.env.HERMES_KANBAN_TASK=") for arg in overrides)


def test_only_dispatcher_owned_workers_get_mcp_overrides(launch):
    """An ordinary launch and a worker that does not own the dispatcher's task emit no
    ``mcp_servers.*`` override at all (nothing to scope, nothing for codex to reject)."""
    assert launch({}) == []
    with non_dispatcher_owned_context():
        assert launch({"HERMES_KANBAN_TASK": "11111111-1111-4111-8111-111111111111"}) == []
