from hermes_cli.mcp_security import validate_mcp_server_entry


def test_validate_local_stdio_server_allows_basic_local_config():
    issues = validate_mcp_server_entry(
        "filesystem",
        {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {},
        },
    )
    assert issues == []


def test_validate_blocks_remote_host_when_not_allowlisted():
    issues = validate_mcp_server_entry(
        "github",
        {
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer abc123"},
        },
    )
    assert any("not on the MCP host allowlist" in issue for issue in issues)


def test_validate_blocks_literal_secret_shaped_env_values():
    issues = validate_mcp_server_entry(
        "remote_api",
        {
            "url": "http://localhost:8000/mcp",
            "env": {"OPENAI_API_KEY": "sk-testabcdefghijklmnopqrstuvwxyz123456"},
        },
    )
    assert any("literal secret-shaped env value" in issue for issue in issues)


def test_validate_allows_env_placeholders_for_mcp_credentials():
    issues = validate_mcp_server_entry(
        "remote_api",
        {
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "Bearer ${MCP_REMOTE_API_KEY}"},
        },
    )
    assert issues == []
