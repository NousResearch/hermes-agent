EXCLUDED_BUDDY_MCPS = {
    "n8n",
    "notion",
    "azure",
    "msgraph",
    "microsoft_graph",
    "halopsa",
    "connectwise",
    "itglue",
}


def _configured_mcp_names(config):
    servers = config.get("mcp_servers") or {}
    return set(servers)


def _excluded_buddy_mcps(config):
    return sorted(_configured_mcp_names(config) & EXCLUDED_BUDDY_MCPS)


def test_buddy_config_allows_clearthought_and_personal_tools():
    config = {
        "mcp_servers": {
            "clearthought": {
                "command": "npx",
                "args": ["-y", "@waldzellai/clear-thought-onepointfive"],
            },
            "filesystem-personal": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            },
        }
    }

    assert _excluded_buddy_mcps(config) == []


def test_buddy_config_rejects_business_mcp_names():
    config = {
        "mcp_servers": {
            "clearthought": {
                "command": "npx",
                "args": ["-y", "@waldzellai/clear-thought-onepointfive"],
            },
            "notion": {
                "command": "npx",
                "args": ["-y", "@notionhq/notion-mcp-server"],
            },
            "halopsa": {"command": "node", "args": ["server.js"]},
        }
    }

    assert _excluded_buddy_mcps(config) == ["halopsa", "notion"]


def test_buddy_delegation_prefers_spark_sidecars():
    config = {
        "delegation": {
            "model": "gpt-5.3-codex-spark",
            "reasoning_effort": "high",
            "max_concurrent_children": 3,
            "subagent_auto_approve": True,
        }
    }

    delegation = config["delegation"]
    assert delegation["model"] == "gpt-5.3-codex-spark"
    assert delegation["max_concurrent_children"] == 3
    assert delegation["subagent_auto_approve"] is True
