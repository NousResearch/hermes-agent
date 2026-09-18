"""Strict profile identifiers must not admit trailing shell line separators."""
from pathlib import Path

import pytest
from fastapi import HTTPException
from hermes_cli import profiles


@pytest.mark.parametrize("validator", [profiles.validate_profile_name, profiles.validate_alias_name])
def test_strict_identifier_rejects_trailing_newline(validator):
    validator("work-bot")
    with pytest.raises(ValueError):
        validator("work-bot\n")


def test_setup_command_rejects_newline_before_shell_dispatch(tmp_path, monkeypatch):
    from hermes_cli.web_routers.profiles import _profile_setup_command

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _profile_setup_command("default") == "hermes setup"
    with pytest.raises(HTTPException) as error:
        _profile_setup_command("default\n")
    assert error.value.status_code == 400
