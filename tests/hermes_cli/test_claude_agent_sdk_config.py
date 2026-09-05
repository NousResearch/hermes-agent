"""Tests for the agent.claude_agent_sdk config block.

The claude-agent-sdk provider reads its behavioural flags exclusively from
config.yaml (AGENTS.md keeps non-secret behavioural settings out of
HERMES_* environment variables), so the canonical defaults must be
registered in DEFAULT_CONFIG — the example file alone does not make them
real config options for default-driven config tooling.
"""

from __future__ import annotations

import inspect


from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli import config_defaults


class TestClaudeAgentSdkDefaults:
    def test_default_config_has_the_block(self):
        agent = DEFAULT_CONFIG.get("agent")
        assert isinstance(agent, dict)
        assert "claude_agent_sdk" in agent

    def test_canonical_defaults(self):
        # Upstream-conservative: every default is falsy/conservative. Pinned
        # PER KEY (not whole-dict equality) so adding a new key to the block
        # doesn't break unrelated pins — each key's relationship is the
        # contract, not the dict's exact shape.
        block = DEFAULT_CONFIG["agent"]["claude_agent_sdk"]
        # No partial-message deltas unless the operator opts in.
        assert block["streaming"] is False
        # Refuse to start over a metered key unless explicitly allowed.
        assert block["allow_metered_key"] is False
        # No persona file appended by default.
        assert block["append_file"] == ""
        # null = the built-in 22000-character whole-append budget.
        assert block["append_total_max_chars"] is None
        # "" = current behavior: the HERMES_TERMINAL_SECURITY_MODE mapping
        # stands; a non-empty value is an SDK permission_mode literal.
        assert block["permission_mode"] == ""
        # No child-process environment overrides unless explicitly configured.
        assert block["env"] == {}
        # [] = full SDK settings isolation; deployments that keep tool
        # grants in ~/.claude/settings.json opt in with e.g. ["user"].
        assert block["setting_sources"] == []
        # null = no per-query budget cap (current behavior).
        assert block["max_budget_usd"] is None
        # null = the transport's 10 MiB NDJSON message limit. The explicit
        # config key keeps the operator override in the canonical schema.
        assert block["max_buffer_size"] is None
        # null = the built-in 600s soft budget; the value itself lives in the
        # transport (activity-aware — see claude_agent_sdk_session.run_turn).
        assert block["turn_timeout"] is None
        # null = streaming-dependent built-in (90s with streaming on,
        # disabled with streaming off); 0 = explicitly disabled.
        assert block["post_tool_quiet_timeout"] is None
        # Hybrid in-process MCP bridge off by default — the wide bridge
        # exposes agent-level tools whose enablement is a security choice;
        # operators opt in explicitly. This pin fails if a well-meaning
        # rebase flips the default: any deployment that upgrades would
        # silently start routing the full registry into the SDK.
        assert block["hybrid_mcp_bridge"] is False
        # Empty exclude list = expose everything the bridge can reach.
        # A non-empty default here would surprise operators who read the
        # docs and expect the whole registry.
        assert block["hybrid_mcp_bridge_exclude"] == []
        # Every default in the block must be falsy — a new key that defaults
        # truthy is a behavior change and needs its own explicit pin here.
        for key, value in block.items():
            assert not value, f"default for {key!r} must be conservative/falsy"

    def test_bypass_permissions_contract_documents_audited_emulation(self):
        source = inspect.getsource(config_defaults)
        permission_comment = source[
            source.index("# SDK permission mode"):
            source.index('"permission_mode": ""')
        ]
        assert "bypassPermissions" in permission_comment
        assert "not forwarded verbatim" in permission_comment
        assert "immutable" in permission_comment
        assert "audited" in permission_comment


class TestUserConfigMerge:
    """A pre-existing config.yaml without the block gets the defaults via
    the deep merge; explicit user values survive it.
    """

    def _load(self, tmp_path, monkeypatch, user_cfg):
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump(user_cfg))

        monkeypatch.setenv("HERMES_HOME", str(home))
        # Force a fresh reimport of config.py so the HERMES_HOME is honored.
        import importlib
        import hermes_cli.config as cfg_mod

        importlib.reload(cfg_mod)
        return cfg_mod.load_config()

    def test_config_without_block_gets_defaults(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, monkeypatch, {"agent": {"max_turns": 5}})
        # Per-key pins (not whole-dict equality) — see test_canonical_defaults.
        block = cfg["agent"]["claude_agent_sdk"]
        assert block["streaming"] is False
        assert block["allow_metered_key"] is False
        assert block["append_file"] == ""
        assert block["env"] == {}
        assert set(block) == set(DEFAULT_CONFIG["agent"]["claude_agent_sdk"])
        # The user's own key survives beside the filled-in block.
        assert cfg["agent"]["max_turns"] == 5

    def test_explicit_user_values_survive_merge(self, tmp_path, monkeypatch):
        cfg = self._load(
            tmp_path,
            monkeypatch,
            {"agent": {"claude_agent_sdk": {"streaming": True}}},
        )
        assert cfg["agent"]["claude_agent_sdk"]["streaming"] is True
        # Keys the user didn't set still arrive from DEFAULT_CONFIG.
        assert cfg["agent"]["claude_agent_sdk"]["allow_metered_key"] is False
        assert cfg["agent"]["claude_agent_sdk"]["append_file"] == ""
