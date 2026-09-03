from __future__ import annotations

from types import SimpleNamespace

import pytest

import gateway.wisdom_command as wisdom_command
from cli import HermesCLI


class _Service:
    def __init__(self) -> None:
        self.store = SimpleNamespace(active_org_id=lambda: "org-1")


@pytest.mark.parametrize(
    ("command", "expected_args"),
    [
        ("/wisdom browse deploy", "browse deploy"),
        (
            "/collective-wisdom-install skill-1@v2",
            "install skill-1@v2",
        ),
    ],
)
def test_cli_wisdom_dispatches_shared_controller(
    monkeypatch,
    capsys,
    tmp_path,
    command,
    expected_args,
):
    seen: list[str] = []

    def execute(_controller, raw_args, _service, _context):
        seen.append(raw_args)
        return wisdom_command.WisdomView(
            "Collective Wisdom",
            actions=[
                wisdom_command.WisdomAction(
                    "Browse", "browse", local_command="/wisdom browse"
                )
            ],
        )

    monkeypatch.setattr("hermes_wisdom.service.WisdomService", _Service)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(wisdom_command.WisdomCommandController, "execute", execute)

    cli = SimpleNamespace(session_id="session-1")
    HermesCLI._handle_wisdom_command(cli, command)

    assert seen == [expected_args]
    output = capsys.readouterr().out
    assert "Collective Wisdom" in output
    assert "Browse: /wisdom browse" in output
