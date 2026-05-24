import sys

import pytest

from hermes_cli import main as cli_main


def test_doctor_json_flag_requires_path_authority(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["hermes", "doctor", "--json"])
    monkeypatch.setattr(cli_main, "cmd_doctor", lambda args: calls.append(args))

    with pytest.raises(SystemExit) as exc:
        cli_main.main()

    assert exc.value.code == 2
    assert calls == []
    assert "--path-authority" in capsys.readouterr().err


def test_doctor_path_authority_allows_scoped_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "doctor", "--path-authority", "--json", "--no-live", "--strict"],
    )
    monkeypatch.setattr(cli_main, "cmd_doctor", lambda args: calls.append(args))

    cli_main.main()

    assert len(calls) == 1
    args = calls[0]
    assert args.path_authority is True
    assert args.json is True
    assert args.no_live is True
    assert args.strict is True
