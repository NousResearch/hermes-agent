from pathlib import Path
from types import SimpleNamespace

from hermes_cli import vault


class _Console:
    def __init__(self):
        self.lines = []

    def print(self, value=""):
        self.lines.append(str(value))


def test_sdk_endpoint_mapping():
    assert vault._sdk_endpoints("") == ("", "")
    assert vault._sdk_endpoints("https://vault.bitwarden.com") == ("", "")
    assert vault._sdk_endpoints("https://vault.bitwarden.eu/") == (
        "https://api.bitwarden.eu", "https://identity.bitwarden.eu",
    )
    assert vault._sdk_endpoints("https://vault.example.test") == (
        "https://vault.example.test/api", "https://vault.example.test/identity",
    )


def test_setup_configures_dedicated_project_without_bulk_secret_source(monkeypatch):
    from agent.secret_sources import bitwarden as bw
    from hermes_cli import config, secrets_cli

    token = "0.machine-token"
    project_id = "00000000-0000-0000-0000-000000000002"
    cfg = {"vault": {"write_backend": "bitwarden", "bitwarden": {"enabled": True}}}
    saved = {}
    env_writes = []
    console = _Console()

    monkeypatch.setenv("BWS_ACCESS_TOKEN", token)
    monkeypatch.setattr(vault, "_console", lambda: console)
    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing: Path("/fake/bws"))
    monkeypatch.setattr(
        secrets_cli, "_list_projects",
        lambda binary, access_token, output, server_url="": [{"id": project_id, "name": "Hermes Credentials"}],
    )
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(config, "save_config", lambda value: saved.update(value))
    monkeypatch.setattr(config, "save_env_value", lambda name, value: env_writes.append((name, value)))
    monkeypatch.setattr(config, "get_env_path", lambda: Path("/safe/.env"))

    vault._cmd_setup_bitwarden_secrets(SimpleNamespace(
        project_id=project_id,
        server_url="https://vault.bitwarden.eu",
    ))

    section = saved["vault"]["bitwarden_secrets"]
    assert saved["vault"]["write_backend"] == "bitwarden_secrets"
    assert saved["vault"]["bitwarden"]["enabled"] is False
    assert section["enabled"] is True and section["project_id"] == project_id
    assert section["api_url"] == "https://api.bitwarden.eu"
    assert section["identity_url"] == "https://identity.bitwarden.eu"
    assert "secrets" not in saved
    assert env_writes == [("BWS_ACCESS_TOKEN", token)]
    assert token not in "\n".join(console.lines)
