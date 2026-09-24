"""Per-platform ``skip_context_files`` gateway wiring (#26860).

Messaging platforms can opt out of the filesystem-heavy context-file
discovery (SOUL.md, AGENTS.md, .cursorrules walks) that runs during
AIAgent construction — especially impactful on Windows where stat() and
directory walks are 10-100x slower. The agent-side parameters already
exist (agent/agent_init.py); these tests pin the gateway wiring:
config -> signature -> AIAgent kwargs.
"""


from gateway.run import GatewayRunner


class TestSkipContextFilesSignature:
    """A toggled skip_context_files must invalidate the agent cache."""

    RUNTIME = {"provider": "openrouter", "base_url": "", "api_mode": ""}

    def test_signature_differs_when_toggled(self):
        sig_off = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
            skip_context_files=False,
        )
        sig_on = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
            skip_context_files=True,
        )
        assert sig_off != sig_on, (
            "skip_context_files changes the frozen system prompt (context "
            "files in vs out) — the cache signature must change with it"
        )

    def test_signature_stable_when_unchanged(self):
        sig_a = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
            skip_context_files=True,
        )
        sig_b = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
            skip_context_files=True,
        )
        assert sig_a == sig_b

    def test_default_matches_explicit_false(self):
        """Back-compat: omitting the param must hash like False so existing
        cached agents aren't all invalidated by this change."""
        sig_default = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
        )
        sig_false = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", self.RUNTIME, ["hermes-telegram"], "",
            skip_context_files=False,
        )
        assert sig_default == sig_false


