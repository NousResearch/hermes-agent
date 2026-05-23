from types import SimpleNamespace

from hermes_cli.self_knowledge import generators


def test_generate_capabilities_from_registry_entries(monkeypatch):
    entries = [
        SimpleNamespace(name="web_search", toolset="web", description="Search the web"),
        SimpleNamespace(name="terminal", toolset="terminal", description="Run commands"),
    ]

    monkeypatch.setattr(generators, "_load_tool_entries", lambda: entries)

    rendered = generators.generate_capabilities()

    assert "| Tool | Toolset | Description |" in rendered
    assert "| terminal | terminal | Run commands |" in rendered
    assert "| web_search | web | Search the web |" in rendered


def test_generate_toolsets_from_toolsets_module(monkeypatch):
    monkeypatch.setattr(
        generators,
        "_load_toolsets",
        lambda: ({"web": {"description": "Web tools", "tools": ["web_search"], "includes": []}}, ["web_search"]),
    )

    rendered = generators.generate_toolsets()

    assert "| Toolset | Description | Tools | Includes |" in rendered
    assert "| web | Web tools | 1 | - |" in rendered
    assert "Core default tools" in rendered


def test_generate_slash_commands_from_registry(monkeypatch):
    commands = [
        SimpleNamespace(name="voice", category="Configuration", description="Toggle voice mode", cli_only=False, gateway_only=False),
        SimpleNamespace(name="clear", category="Session", description="Clear screen", cli_only=True, gateway_only=False),
    ]
    monkeypatch.setattr(generators, "_load_commands", lambda: commands)

    rendered = generators.generate_slash_commands()

    assert "| Command | Category | Scope | Description |" in rendered
    assert "| /voice | Configuration | cli+gateway | Toggle voice mode |" in rendered
    assert "| /clear | Session | cli | Clear screen |" in rendered


def test_generate_gateway_platforms_from_files(tmp_path, monkeypatch):
    platforms = tmp_path / "gateway" / "platforms"
    platforms.mkdir(parents=True)
    (platforms / "discord.py").write_text("# adapter")
    (platforms / "__init__.py").write_text("")
    monkeypatch.setattr(generators, "PROJECT_ROOT", tmp_path)

    rendered = generators.generate_gateway_platforms()

    assert "discord" in rendered
    assert "__init__" not in rendered


def test_generate_recent_activity_handles_git_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(generators.subprocess, "run", fail)

    assert "_unavailable: git log failed" in generators.generate_recent_activity()
