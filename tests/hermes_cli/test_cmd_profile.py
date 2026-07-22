"""Focused tests for the profile command output contract."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import cmd_profile
from hermes_cli.profiles import ProfileInfo


def test_profile_list_json_emits_only_the_structured_envelope(capsys):
    profiles = [
        ProfileInfo(
            name="default",
            path=Path("/tmp/hermes"),
            is_default=True,
            gateway_running=False,
            model="example/model",
            provider="example",
        )
    ]

    with patch("hermes_cli.profiles.list_profiles", return_value=profiles), \
         patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        cmd_profile(SimpleNamespace(profile_action="list", json=True))

    output = capsys.readouterr().out
    assert json.loads(output) == {
        "profiles": [
            {
                "name": "default",
                "path": "/tmp/hermes",
                "is_default": True,
                "model": "example/model",
                "provider": "example",
                "has_env": False,
                "skill_count": 0,
                "gateway_running": False,
                "description": "",
                "description_auto": False,
                "distribution_name": None,
                "distribution_version": None,
                "distribution_source": None,
                "has_alias": False,
            }
        ]
    }
    assert "Profile" not in output
    assert "Gateway" not in output


def test_profile_list_json_empty_result_is_still_valid_json(capsys):
    with patch("hermes_cli.profiles.list_profiles", return_value=[]), \
         patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        cmd_profile(SimpleNamespace(profile_action="list", json=True))

    assert json.loads(capsys.readouterr().out) == {"profiles": []}
