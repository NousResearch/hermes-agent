import json
from argparse import Namespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.honcho import cli


def _with_home(path):
    token = set_hermes_home_override(path)
    return token


def test_cmd_compartments_set_writes_host_scoped_api_key_file(tmp_path, capsys):
    token = _with_home(tmp_path)
    old_profile = cli._profile_override
    cli._profile_override = None
    try:
        (tmp_path / "honcho.json").write_text(
            json.dumps({"baseUrl": "http://honcho.local", "hosts": {"hermes": {"workspace": "primary"}}}),
            encoding="utf-8",
        )

        cli.cmd_compartments(
            Namespace(
                compartment_action="set",
                name="ops",
                workspace="james-ops-prod",
                api_key_file="runtime-keys/james-ops-prod.jwt",
                api_key=None,
                base_url=None,
                environment=None,
                session_strategy=None,
                manual_session_name=None,
            )
        )

        cfg = json.loads((tmp_path / "honcho.json").read_text(encoding="utf-8"))
        ops = cfg["hosts"]["hermes"]["compartments"]["ops"]
        assert ops == {
            "workspace": "james-ops-prod",
            "apiKeyFile": "runtime-keys/james-ops-prod.jwt",
        }
        assert "apiKey" not in ops
        assert "Compartment 'ops' configured" in capsys.readouterr().out
    finally:
        cli._profile_override = old_profile
        reset_hermes_home_override(token)


def test_cmd_compartments_list_shows_file_backed_compartment_without_secret(tmp_path, capsys):
    token = _with_home(tmp_path)
    old_profile = cli._profile_override
    cli._profile_override = None
    try:
        (tmp_path / "honcho.json").write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "workspace": "primary",
                            "compartments": {
                                "personal": {
                                    "workspace": "james-personal-prod",
                                    "apiKeyFile": "runtime-keys/james-personal-prod.jwt",
                                }
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        cli.cmd_compartments(Namespace(compartment_action="list"))

        output = capsys.readouterr().out
        assert "personal" in output
        assert "james-personal-prod" in output
        assert "apiKeyFile" in output
        assert "runtime-keys/james-personal-prod.jwt" in output
        assert "apiKey=" not in output
    finally:
        cli._profile_override = old_profile
        reset_hermes_home_override(token)


def test_cmd_agents_set_writes_host_scoped_compartment_allowlist(tmp_path, capsys):
    token = _with_home(tmp_path)
    old_profile = cli._profile_override
    cli._profile_override = None
    try:
        (tmp_path / "honcho.json").write_text(
            json.dumps({"hosts": {"hermes": {"workspace": "primary"}}}),
            encoding="utf-8",
        )

        cli.cmd_agents(
            Namespace(
                agent_action="set",
                name="echo",
                compartments=["ops"],
            )
        )

        cfg = json.loads((tmp_path / "honcho.json").read_text(encoding="utf-8"))
        assert cfg["hosts"]["hermes"]["agents"] == {"echo": ["ops"]}
        assert "Agent 'echo' can access: ops" in capsys.readouterr().out
    finally:
        cli._profile_override = old_profile
        reset_hermes_home_override(token)


def test_cmd_agents_list_shows_allowlists(tmp_path, capsys):
    token = _with_home(tmp_path)
    old_profile = cli._profile_override
    cli._profile_override = None
    try:
        (tmp_path / "honcho.json").write_text(
            json.dumps(
                {
                    "hosts": {
                        "hermes": {
                            "workspace": "primary",
                            "agents": {
                                "echo": ["ops"],
                                "miloh": ["ops", "personal"],
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        cli.cmd_agents(Namespace(agent_action="list"))

        output = capsys.readouterr().out
        assert "echo" in output
        assert "ops" in output
        assert "miloh" in output
        assert "ops, personal" in output
    finally:
        cli._profile_override = old_profile
        reset_hermes_home_override(token)
