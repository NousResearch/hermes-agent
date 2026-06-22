import json


def test_project_mcp_load_requires_opt_in_unless_forced(tmp_path, monkeypatch):
    from hermes_cli.mcp_project import load_project_mcp_servers

    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"proj": {"url": "http://127.0.0.1:8000/mcp"}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)
    monkeypatch.delenv("HERMES_USE_PROJECT_MCP_JSON", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)

    assert load_project_mcp_servers(config={}).servers == {}

    forced = load_project_mcp_servers(config={}, force=True)
    assert forced.path == project / ".mcp.json"
    assert forced.servers["proj"]["url"] == "http://127.0.0.1:8000/mcp"


def test_project_mcp_loads_common_shapes_and_strips_markdown_url(tmp_path, monkeypatch):
    from hermes_cli.mcp_project import load_project_mcp_servers

    project = tmp_path / "project"
    project.mkdir()
    (project / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "http": {"url": "<http://127.0.0.1:8000/mcp>"},
                    "stdio": {
                        "command": "npx",
                        "args": ["-y", "some-mcp-server"],
                        "env": {"SOME_ENV": "value"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)

    loaded = load_project_mcp_servers(
        config={"mcp": {"use_project_mcp_json": True}},
    )

    assert loaded.servers["http"]["url"] == "http://127.0.0.1:8000/mcp"
    assert loaded.servers["stdio"]["command"] == "npx"
    assert loaded.servers["stdio"]["args"] == ["-y", "some-mcp-server"]
    assert loaded.servers["stdio"]["env"] == {"SOME_ENV": "value"}


def test_project_mcp_loads_utf8_bom_json(tmp_path, monkeypatch):
    from hermes_cli.mcp_project import load_project_mcp_servers

    project = tmp_path / "project"
    project.mkdir()
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"bom": {"url": "http://127.0.0.1:8000/mcp"}}}),
        encoding="utf-8-sig",
    )
    monkeypatch.chdir(project)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)

    loaded = load_project_mcp_servers(config={}, force=True)

    assert loaded.servers["bom"]["url"] == "http://127.0.0.1:8000/mcp"


def test_project_mcp_merge_prefers_config_yaml():
    from hermes_cli.mcp_project import merge_mcp_server_configs

    merged = merge_mcp_server_configs(
        {"dup": {"url": "https://config.example/mcp"}},
        {
            "dup": {"url": "https://project.example/mcp"},
            "project-only": {"command": "npx"},
        },
    )

    assert merged["dup"]["url"] == "https://config.example/mcp"
    assert merged["project-only"]["command"] == "npx"
