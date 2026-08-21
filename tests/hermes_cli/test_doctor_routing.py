"""Routing explainability doctor tests."""

from argparse import Namespace
import json
import os
import subprocess
import sys

from hermes_cli import doctor


def test_routing_doctor_json_explains_selected_route_without_secrets(monkeypatch, tmp_path, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """gateway:\n  multiplex_profiles: true\nprofile_routes:\n  - name: private\n    platform: discord\n    chat_id: '12345'\n    profile: analyst\n"""
    )
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    rc = doctor.run_doctor(
        Namespace(
            routing=True,
            routing_profile="default",
            routing_platform=" discord ",
            routing_chat_id=" 12345 ",
            routing_thread_id=None,
            routing_user_id="user-secret",
            json=True,
            fix=False,
            ack=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_profile"] == "analyst"
    assert payload["match"]["route"] == "private"
    assert payload["match"]["reason"] == "specificity"
    assert payload["dimensions"]["platform"] == "discord"
    assert payload["dimensions"]["chat_id"] != "12345"
    assert payload["dimensions"]["user_id"] != "user-secret"
    assert "user-secret" not in json.dumps(payload)
    assert payload["side_effects"] == {"network": False, "gateway": False, "writes": False}


def test_routing_doctor_reports_ambiguous_routes_deterministically(monkeypatch, tmp_path, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """profile_routes:\n  - name: first\n    platform: telegram\n    chat_id: '1'\n    profile: one\n  - name: second\n    platform: telegram\n    chat_id: '1'\n    profile: two\n"""
    )
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    rc = doctor.run_doctor(
        Namespace(
            routing=True,
            routing_profile="default",
            routing_platform="telegram",
            routing_chat_id="1",
            routing_thread_id=None,
            routing_user_id=None,
            json=True,
            fix=False,
            ack=None,
        )
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "ambiguous_route"
    assert payload["error"]["routes"] == ["first", "second"]
    assert payload["selected_profile"] is None


def test_routing_doctor_reports_invalid_config_without_running_gateway(monkeypatch, tmp_path, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("profile_routes: [not-a-map]\n")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    rc = doctor.run_doctor(
        Namespace(
            routing=True,
            routing_profile="default",
            routing_platform="telegram",
            routing_chat_id="1",
            routing_thread_id=None,
            routing_user_id=None,
            json=True,
            fix=False,
            ack=None,
        )
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_route_config"
    assert "not-a-map" not in json.dumps(payload)


def test_routing_doctor_cli_uses_temp_home_and_is_deterministic(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """profile_routes:\n  - name: support\n    platform: telegram\n    chat_id: '42'\n    profile: support\n"""
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    command = [
        sys.executable, "-m", "hermes_cli.main", "doctor", "--routing",
        "--routing-profile", "default", "--platform", "telegram",
        "--chat-id", "42", "--json",
    ]
    first = subprocess.run(command, cwd=os.getcwd(), env=env, capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=os.getcwd(), env=env, capture_output=True, text=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["selected_profile"] == "support"
