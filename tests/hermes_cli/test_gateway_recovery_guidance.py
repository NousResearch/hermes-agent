"""Self-targeting refusals explain safe legacy/forced-duplicate recovery (#52036)."""

from argparse import Namespace
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("verb", ["stop", "uninstall"])
def test_cli_refusal_names_external_shell_and_scoped_recovery(monkeypatch, capsys, verb):
    from hermes_cli.gateway import gateway_command

    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    with pytest.raises(SystemExit) as error:
        gateway_command(Namespace(gateway_command=verb, all=False, system=False))
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "separate shell outside" in output
    assert "legacy installation" in output and "--force" in output
    assert "only the conflicting" in output
    assert "systemctl" not in output


@pytest.mark.parametrize("verb", ["stop", "restart", "uninstall"])
def test_terminal_refusal_keeps_the_guard_and_recovery_hint(monkeypatch, tmp_path, verb):
    from tools.terminal_tool_guards import gateway_lifecycle_block

    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    result = gateway_lifecycle_block(command=f"hermes gateway {verb}", env=SimpleNamespace(cwd=str(tmp_path)),
                                    env_type="local", cwd=str(tmp_path), workdir=None, session_key="recovery-test")
    output = json.loads(result)
    assert output["exit_code"] == 1
    assert "Blocked" in output["error"]
    assert "legacy installation" in output["error"] and "--force" in output["error"]
    assert "only the conflicting" in output["error"]
