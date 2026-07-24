"""Tests for the agent.claude_agent_sdk config block.

The claude-agent-sdk provider reads its behavioural flags exclusively from
config.yaml (AGENTS.md keeps non-secret behavioural settings out of
HERMES_* environment variables), so the canonical defaults must be
registered in DEFAULT_CONFIG — the example file alone does not make them
real config options for default-driven config tooling.
"""

from __future__ import annotations


from hermes_cli.config import DEFAULT_CONFIG


class TestClaudeAgentSdkDefaults:
    def test_default_config_has_the_block(self):
        agent = DEFAULT_CONFIG.get("agent")
        assert isinstance(agent, dict)
        assert "claude_agent_sdk" in agent

    def test_canonical_defaults(self):
        # Upstream-conservative: no partial-message deltas, refuse to start
        # over a metered key, no persona file appended.
        assert DEFAULT_CONFIG["agent"]["claude_agent_sdk"] == {
            "streaming": False,
            "allow_metered_key": False,
            "append_file": "",
        }


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
        assert cfg["agent"]["claude_agent_sdk"] == {
            "streaming": False,
            "allow_metered_key": False,
            "append_file": "",
        }
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
