"""``hermes plugins export`` writes a package the importer loads, with no credential values in it."""

import json
from pathlib import Path

import pytest

from hermes_cli.agent_plugin_export import export_agent_plugin
from hermes_cli.agent_plugins import load_agent_plugin

_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"


def _profile(home: Path) -> None:
    skill = home / "skills" / "research" / "paper-notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: paper-notes\ndescription: Notes from a paper.\nmetadata:\n  hermes:\n    tags: [x]\n---\n# Body\n",
        encoding="utf-8")
    (home / "config.yaml").write_text(
        "mcp_servers:\n"
        "  fs:\n    command: npx\n    args: [server-fs, --auth-type=oauth, --max-tokens=100, '${PLUGIN_DATA}/cache']\n"
        f"    env: {{GITHUB_TOKEN: {_TOKEN}, LOG_LEVEL: info}}\n    trust: untrusted\n"
        "  remote:\n    url: https://mcp.example.com/mcp\n"
        "    headers: {Authorization: 'Bearer ${MY_TOKEN}', X-Tenant: public}\n    auth: oauth\n"
        "  local:\n    command: /usr/local/bin/my-server\n"
        "  argkey:\n    command: srv\n    args: [--api-key, private-value]\n"
        "  urlkey:\n    url: https://mcp.example.com/mcp?api_key=private-value\n"
        "  zero:\n    command: srv\n    timeout: 0\n"
        "  localarg:\n    command: npx\n    args: [server-fs, /home/user/projects]\n",
        encoding="utf-8")


def test_export_round_trips_through_the_importer_without_secrets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    out = tmp_path / "pkg"

    report = export_agent_plugin("research-kit", out, skills=["paper-notes"], servers=["fs", "remote"])
    package = load_agent_plugin(out, tmp_path / "data")

    assert [s.name for s in package.skills] == ["paper-notes"]
    assert set(package.mcp_servers) == {"fs", "remote"}
    assert package.mcp_servers["fs"]["trust"] == "untrusted"
    assert package.diagnostics == ()
    mcp = json.loads((out / "mcp.json").read_text(encoding="utf-8-sig"))
    assert mcp["mcpServers"]["fs"]["env"] == {"GITHUB_TOKEN": "${GITHUB_TOKEN}", "LOG_LEVEL": "info"}
    assert mcp["mcpServers"]["remote"]["headers"] == {"X-Tenant": "public"}
    assert not any(_TOKEN in p.read_text(encoding="utf-8-sig") for p in out.rglob("*") if p.is_file())
    assert report.notes


@pytest.mark.parametrize("server, skill_edit, match", [
    ("local", None, "is a path"),
    ("argkey", None, "credential"),
    ("urlkey", None, "credential"),
    ("zero", None, "does not validate|timeout"),
    (None, "name", "not a valid Agent Skills name"),
    ("localarg", None, "path on this machine"),
    (None, "inside", "inside the skill"),
])
def test_unportable_input_refuses_and_writes_nothing(tmp_path, monkeypatch, server, skill_edit, match):
    home = tmp_path / "home"
    _profile(home)
    skill = home / "skills" / "research" / "paper-notes"
    if skill_edit == "name":
        md = skill / "SKILL.md"
        text = md.read_text(encoding="utf-8-sig").replace("name: paper-notes", "name: ../escaped")
        md.write_text(text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    out = skill / "bundle" if skill_edit == "inside" else tmp_path / "pkg"

    with pytest.raises(ValueError, match=match):
        export_agent_plugin("x", out, skills=["paper-notes"], servers=[server] if server else [])

    assert not out.exists()
    assert not (tmp_path / "escaped").exists()


@pytest.mark.platforms("posix")  # creating a symlink needs Developer Mode on Windows
def test_skill_link_to_a_private_file_refuses(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _profile(home)
    (home / ".env").write_text(f"GITHUB_TOKEN={_TOKEN}\n", encoding="utf-8")
    (home / "skills" / "research" / "paper-notes" / "env-copy").symlink_to(home / ".env")
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(ValueError, match="links outside"):
        export_agent_plugin("x", tmp_path / "pkg", skills=["paper-notes"], servers=[])

    assert not (tmp_path / "pkg").exists()
