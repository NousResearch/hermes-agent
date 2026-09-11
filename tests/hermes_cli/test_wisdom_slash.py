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
    monkeypatch.setattr("hermes_wisdom.entitlement.require_entitlement", lambda: None)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(wisdom_command.WisdomCommandController, "execute", execute)

    cli = object.__new__(HermesCLI)
    cli.session_id = "session-1"
    assert cli.process_command(command) is True

    assert seen == [expected_args]
    output = capsys.readouterr().out
    assert "Collective Wisdom" in output
    assert "Browse: /wisdom browse" in output


def test_cli_wisdom_alias_dispatch_is_denied_before_service_creation(monkeypatch, capsys):
    def denied():
        raise PermissionError("not entitled")

    monkeypatch.setattr("hermes_wisdom.entitlement.require_entitlement", denied)
    monkeypatch.setattr(
        "hermes_wisdom.service.WisdomService",
        lambda: pytest.fail("service must not be constructed when entitlement is absent"),
    )
    cli = object.__new__(HermesCLI)
    cli.session_id = "session-1"

    assert cli.process_command("/collective-wisdom-install skill-1") is True
    assert "could not continue" in capsys.readouterr().out.lower()
