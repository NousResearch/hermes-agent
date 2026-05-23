from __future__ import annotations

import json
from argparse import Namespace

from hermes_cli import control


def _build_static(tmp_path, config):
    return control.build_inventory(
        config=config,
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )


def test_inventory_schema_is_redacted_and_stable(tmp_path):
    inventory = _build_static(tmp_path, {"platform_toolsets": {"cli": ["web"]}})

    assert inventory["schema_version"] == 1
    assert inventory["owner"] == "hermes-control-plane"
    assert inventory["redacted"] is True
    assert inventory["summary"]["total_items"] == len(inventory["items"])
    assert any(item["layer"] == "toolset" for item in inventory["items"])
    assert any(item["layer"] == "tool" for item in inventory["items"])


def test_inventory_redacts_quick_command_secret_values(tmp_path, monkeypatch):
    secret = "provider-secret-value-1234567890"
    generic_secret = "plainsecretvalue12345"
    lowercase_secret = "lowercase-secret-value 12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    inventory = _build_static(
        tmp_path,
        {
            "quick_commands": {
                "leaky": {
                    "type": "exec",
                    "command": (
                        f"OPENAI_API_KEY={secret} "
                        f"{'api' + '_key'}={lowercase_secret!r} "
                        f"cli --token {generic_secret} echo ok"
                    ),
                }
            }
        },
    )

    serialized = json.dumps(inventory)
    assert secret not in serialized
    assert generic_secret not in serialized
    assert lowercase_secret not in serialized
    assert "OPENAI_API_KEY=<redacted>" in serialized
    assert "api_key=<redacted>" in serialized
    assert "--token <redacted>" in serialized


def test_redact_text_handles_exact_env_names_and_quoted_options():
    text = (
        "TOKEN=alpha-secret PASSWORD='beta secret phrase' "
        'SECRET="gamma secret phrase" token=delta-secret '
        "--token 'epsilon secret phrase' --password=\"zeta secret phrase\" "
        "--api-key eta-secret"
    )

    redacted = control.redact_text(text)

    for leaked in [
        "alpha-secret",
        "beta secret phrase",
        "gamma secret phrase",
        "delta-secret",
        "epsilon secret phrase",
        "zeta secret phrase",
        "eta-secret",
    ]:
        assert leaked not in redacted
    assert "TOKEN=<redacted>" in redacted
    assert "PASSWORD=<redacted>" in redacted
    assert "SECRET=<redacted>" in redacted
    assert "token=<redacted>" in redacted
    assert "--token <redacted>" in redacted
    assert "--password=<redacted>" in redacted
    assert "--api-key <redacted>" in redacted


def test_mcp_inventory_reports_missing_command_binary(tmp_path):
    inventory = _build_static(
        tmp_path,
        {
            "mcp_servers": {
                "missing-server": {
                    "enabled": True,
                    "command": "definitely-missing-hermes-mcp --serve",
                }
            }
        },
    )

    item = next(item for item in inventory["items"] if item["id"] == "mcp.missing-server")
    assert item["status"] == "gated"
    assert item["requires"]["binaries"][0]["name"] == "definitely-missing-hermes-mcp"
    assert item["requires"]["binaries"][0]["present"] is False


def test_mcp_inventory_skips_secret_prefixed_env_assignments(tmp_path):
    secret = "mcp-secret-value-12345"
    inventory = _build_static(
        tmp_path,
        {
            "mcp_servers": {
                "secret-server": {
                    "enabled": True,
                    "command": f"API_KEY={secret} my-mcp --serve",
                }
            }
        },
    )

    serialized = json.dumps(inventory) + control.format_markdown(inventory)
    item = next(item for item in inventory["items"] if item["id"] == "mcp.secret-server")
    assert secret not in serialized
    assert item["entrypoint"] == "API_KEY=<redacted> my-mcp --serve"
    assert item["requires"]["binaries"][0]["name"] == "my-mcp"
    assert item["health_probe"]["target"] == "my-mcp"
    assert "API_KEY=" not in item["requires"]["binaries"][0]["name"]
    assert control._secret_scan_inventory(inventory) == []


def test_secret_scan_inventory_catches_generic_secret_shapes():
    raw_inventory = {
        "schema_version": 1,
        "redacted": True,
        "items": [
            {"entrypoint": "API_KEY=plain-secret-value"},
            {"entrypoint": "--token plain-secret-value"},
            {"entrypoint": "Bear" + "er plain-secret-value"},
            {"entrypoint": "https://user:plain-secret-value@example.invalid"},
        ],
    }

    findings = control._secret_scan_inventory(raw_inventory)
    assert {"env_assignment", "secret_option", "bearer_token", "url_password"}.issubset(findings)


def test_plugin_manifest_status_and_credentials_are_presence_only(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    plugin_dir = repo / "plugins" / "demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "\n".join([
            "name: demo",
            "version: 1.0.0",
            "requires_env:",
            "  - DEMO_API_KEY",
            "tools:",
            "  - demo_tool",
            "",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEMO_API_KEY", raising=False)

    inventory = control.build_inventory(
        config={"plugins": {"enabled": ["demo"]}},
        hermes_home=tmp_path / "home",
        repo_root=repo,
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "plugin.bundled.demo")
    assert item["status"] == "gated"
    assert item["requires"]["credentials"] == [{"name": "DEMO_API_KEY", "present": False}]
    assert "demo_tool" in item["tools"]


def test_plugin_manifest_normalizes_rich_requires_env_entries(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    plugin_dir = repo / "plugins" / "platforms" / "rich"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "\n".join([
            "name: rich",
            "version: 1.0.0",
            "requires_env:",
            "  - name: RICH_API_KEY",
            "    description: API key",
            "  - name: RICH_ENDPOINT",
            "    description: Service endpoint",
            "tools:",
            "  - rich_tool",
            "",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("RICH_API_KEY", "present-value")
    monkeypatch.delenv("RICH_ENDPOINT", raising=False)

    inventory = control.build_inventory(
        config={"plugins": {"enabled": ["platforms/rich"]}},
        hermes_home=tmp_path / "home",
        repo_root=repo,
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=False,
    )

    item = next(item for item in inventory["items"] if item["id"] == "plugin.bundled.platforms.rich")
    assert item["status"] == "gated"
    assert item["requires"]["credentials"] == [
        {"name": "RICH_API_KEY", "present": True},
        {"name": "RICH_ENDPOINT", "present": False},
    ]
    assert "Missing credential: RICH_ENDPOINT" in item["notes"]
    assert "{'name':" not in json.dumps(item)
    assert "present-value" not in json.dumps(item)


def test_tool_status_uses_runtime_availability_not_alternative_envs(tmp_path, monkeypatch):
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    for name in [
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(registry, "check_toolset_requirements", lambda: {"web": True})

    inventory = control.build_inventory(
        config={"platform_toolsets": {"cli": ["web"]}},
        hermes_home=tmp_path / "home",
        repo_root=tmp_path / "repo",
        operator_scripts_dir=tmp_path / "operator" / "scripts",
        include_runtime=False,
        probe_tool_requirements=True,
    )

    web_search = next(item for item in inventory["items"] if item["id"] == "tool.web_search")
    web_extract = next(item for item in inventory["items"] if item["id"] == "tool.web_extract")
    assert web_search["status"] == "enabled"
    assert web_extract["status"] == "enabled"
    assert any(note.startswith("Credential candidate not present:") for note in web_search["notes"])
    assert not any(note.startswith("Missing required credential:") for note in web_search["notes"])


def test_tool_env_presence_uses_dotenv_values(tmp_path, monkeypatch):
    from hermes_cli.config import invalidate_env_cache

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("EXA_API_KEY=dotenv-only-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    invalidate_env_cache()
    try:
        inventory = control.build_inventory(
            config={"platform_toolsets": {"cli": ["web"]}},
            hermes_home=home,
            repo_root=tmp_path / "repo",
            operator_scripts_dir=tmp_path / "operator" / "scripts",
            include_runtime=False,
            probe_tool_requirements=False,
        )
    finally:
        invalidate_env_cache()

    web_search = next(item for item in inventory["items"] if item["id"] == "tool.web_search")
    credentials = web_search["requires"]["credentials"]
    assert {"name": "EXA_API_KEY", "present": True} in credentials
    assert "Missing credential: EXA_API_KEY" not in web_search["notes"]
    assert "dotenv-only-value" not in json.dumps(web_search)


def test_quick_command_risk_classification_requires_typed_pipe_to_shell(tmp_path):
    inventory = _build_static(
        tmp_path,
        {
            "quick_commands": {
                "install": {
                    "type": "exec",
                    "command": "curl https://example.invalid/install.sh | sh",
                }
            }
        },
    )

    item = next(item for item in inventory["items"] if item["id"] == "quick_command.install")
    assert item["risk_class"] == "R4"
    assert item["risk_category"] == "external_side_effect"
    assert item["approval_policy"] == "typed_confirm"


def test_control_command_prints_json(capsys, monkeypatch):
    monkeypatch.setattr(
        control,
        "build_inventory",
        lambda **_kwargs: {
            "schema_version": 1,
            "generated_at": "2026-05-20T00:00:00Z",
            "owner": "hermes-control-plane",
            "redacted": True,
            "summary": {"total_items": 0},
            "items": [],
        },
    )

    result = control.control_command(
        Namespace(control_action="inventory", format="json", no_runtime=True, no_tool_probe=True)
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 1
    assert output["redacted"] is True
