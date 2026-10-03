"""Exercise the shipped Agent Toolkit entry through CLI and dashboard installation.

Only the user-input and remote-probe boundaries are replaced. Catalog parsing,
credential persistence, config merging and runtime header expansion stay real.
"""

import sys

import pytest
import yaml
from dotenv import dotenv_values


@pytest.fixture
def toolkit(monkeypatch):
    from hermes_cli import mcp_catalog
    from hermes_cli.config import invalidate_env_cache
    from hermes_cli.mcp_config import _env_key_for_server
    from hermes_constants import get_hermes_home

    monkeypatch.delenv("HERMES_OPTIONAL_MCPS", raising=False)
    entry = mcp_catalog.get_entry("agent-toolkit")
    assert entry is not None, "Agent Toolkit must be discoverable in the shipped catalog"
    key_name = _env_key_for_server(entry.name)
    # Track the key before real persistence publishes it into the environment,
    # so pytest also restores that side effect after each test.
    monkeypatch.setenv(key_name, "")
    invalidate_env_cache()
    return entry, key_name, get_hermes_home()


def test_cli_requires_hidden_key_and_persists_selected_tools(toolkit, monkeypatch, capsys):
    from hermes_cli import curses_ui, mcp_catalog, mcp_config
    from hermes_cli.config import save_config
    from tools.mcp_tool_config import _load_mcp_config

    entry, key_name, home = toolkit
    spec = next(spec for spec in entry.auth.env if spec.name == key_name)
    assert spec.secret and spec.required
    unrelated = {"url": "https://existing.example.test/mcp", "enabled": False}
    save_config({"mcp_servers": {"existing": unrelated}})
    config_path = home / "config.yaml"
    before = config_path.read_text(encoding="utf-8")
    credential = "synthetic-cli-credential-not-a-real-key"
    answers = iter(["", credential])
    prompts = []

    def prompt(label, *, default=None, password=False):
        prompts.append((label, password))
        return next(answers)

    monkeypatch.setattr(mcp_catalog, "_prompt_input", prompt)
    with pytest.raises(mcp_catalog.CatalogError, match=key_name):
        mcp_catalog.install_entry(entry)
    assert config_path.read_text(encoding="utf-8") == before

    defaults = entry.tools.default_enabled
    assert defaults, "The first install should offer a curated default selection"
    advertised = [(name, "Synthetic probe result") for name in defaults]
    advertised.append(("fixture_opt_in_tool", "Not selected by default"))
    probes = []

    def probe(name, config):
        probes.append((name, config, _load_mcp_config()[name]))
        return advertised

    def checklist(title, labels, selected):
        assert entry.name in title
        assert {advertised[index][0] for index in selected} == set(defaults)
        return {0}

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses_ui, "curses_checklist", checklist)
    mcp_catalog.install_entry(entry)

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["mcp_servers"]["existing"] == unrelated
    installed = persisted["mcp_servers"][entry.name]
    assert installed["tools"]["include"] == [advertised[0][0]]
    assert installed["headers"]["Authorization"] == f"Bearer ${{{key_name}}}"
    assert dotenv_values(home / ".env")[key_name] == credential
    assert credential not in config_path.read_text(encoding="utf-8")
    assert credential not in capsys.readouterr().out
    assert prompts == [(spec.prompt, True), (spec.prompt, True)]
    assert [name for name, _, _ in probes] == [entry.name]
    _, probe_config, runtime_config = probes[0]
    assert probe_config["url"] == entry.transport.url
    assert probe_config["headers"]["Authorization"] == f"Bearer {credential}"
    assert runtime_config["headers"]["Authorization"] == f"Bearer {credential}"


def test_dashboard_form_installs_real_entry_and_preserves_prior_selection(toolkit, monkeypatch):
    from fastapi.testclient import TestClient

    from hermes_cli import mcp_catalog, mcp_config
    from hermes_cli.config import save_config
    from hermes_cli.web_server import _SESSION_TOKEN, app
    from tools.mcp_tool_config import _load_mcp_config

    entry, key_name, home = toolkit
    client = TestClient(app)
    headers = {"X-Hermes-Session-Token": _SESSION_TOKEN}
    response = client.get("/api/mcp/catalog", headers=headers)
    assert response.status_code == 200
    offered = next(item for item in response.json()["entries"] if item["name"] == entry.name)
    assert offered["required_env"] == [
        {"name": spec.name, "prompt": spec.prompt, "required": spec.required}
        for spec in entry.auth.env
    ]
    assert not offered["installed"]

    selection = [entry.tools.default_enabled[0]]
    unrelated = {"url": "https://existing.example.test/mcp", "enabled": False}
    save_config({"mcp_servers": {
        "existing": unrelated,
        entry.name: {"url": entry.transport.url, "tools": {"include": selection}},
    }})
    credential = "synthetic-dashboard-credential-not-a-real-key"
    probes = []

    def probe(name, config):
        probes.append((name, config, _load_mcp_config()[name]))
        return [(tool, "Synthetic probe result") for tool in entry.tools.default_enabled]

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(mcp_catalog, "_prompt_input", lambda *a, **kw: pytest.fail(
        "The dashboard credential form must satisfy the installer without a terminal prompt"
    ))
    response = client.post(
        "/api/mcp/catalog/install",
        headers=headers,
        json={"name": entry.name, "env": {key_name: credential}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["ok"]
    assert [name for name, _, _ in probes] == [entry.name]
    _, probe_config, runtime_config = probes[0]
    assert probe_config["headers"]["Authorization"] == f"Bearer {credential}"
    assert runtime_config["headers"]["Authorization"] == f"Bearer {credential}"
    config_text = (home / "config.yaml").read_text(encoding="utf-8")
    servers = yaml.safe_load(config_text)["mcp_servers"]
    assert servers["existing"] == unrelated
    assert servers[entry.name]["tools"]["include"] == selection
    assert dotenv_values(home / ".env")[key_name] == credential
    assert credential not in config_text

    response = client.get("/api/mcp/catalog", headers=headers)
    installed = next(item for item in response.json()["entries"] if item["name"] == entry.name)
    assert installed["installed"] and installed["enabled"]
    assert credential not in response.text
