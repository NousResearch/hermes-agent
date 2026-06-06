"""Tests for Mem0 config.yaml threading + runtime profile -> agent_id mapping.

Covers the gap where ``memory.mem0.{user_id,agent_id}`` in ``config.yaml``
was silently dropped on the floor: agent_init never passed the subconfig
through to the plugin, and the plugin's ``_load_config`` only read env vars
plus ``$HERMES_HOME/mem0.json``.  After the fix:

1. ``initialize(provider_config=...)`` is the new threading hop.
2. Precedence for ``agent_id``:
       explicit config (env / mem0.json / yaml) > runtime profile
       (``agent_identity`` kwarg) > hardcoded ``"hermes"`` default.
3. ``user_id`` keeps existing precedence:
       runtime kwarg > explicit config > default.
"""

import json

from plugins.memory.mem0 import Mem0MemoryProvider


# ---------------------------------------------------------------------------
# provider_config threading from config.yaml
# ---------------------------------------------------------------------------


class TestProviderConfigThreading:
    """`provider_config` kwarg carries memory.mem0.* from config.yaml."""

    def test_provider_config_sets_user_id_and_agent_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"user_id": "jereme", "agent_id": "stack"},
        )

        assert provider._user_id == "jereme"
        assert provider._agent_id == "stack"

    def test_provider_config_carries_rerank(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"rerank": False},
        )

        assert provider._rerank is False

    def test_missing_provider_config_falls_back_to_env_and_json(self, monkeypatch, tmp_path):
        """No provider_config kwarg: pre-fix behaviour is preserved."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_USER_ID", "env-user")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")

        assert provider._user_id == "env-user"
        # No agent_id env / yaml / runtime override -> hermes default.
        assert provider._agent_id == "hermes"

    def test_empty_provider_config_keys_dont_clobber_env(self, monkeypatch, tmp_path):
        """``user_id: ""`` in yaml must not wipe out an MEM0_USER_ID env value."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_USER_ID", "env-user")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"user_id": "", "agent_id": None},
        )

        assert provider._user_id == "env-user"
        # agent_id had no explicit source (env unset, yaml empty) ->
        # falls through to default since no agent_identity either.
        assert provider._agent_id == "hermes"

    def test_provider_config_non_dict_is_ignored(self, monkeypatch, tmp_path):
        """Hostile / mistyped provider_config (e.g. a string) shouldn't crash."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session", provider_config="not-a-dict")

        assert provider._user_id == "hermes-user"
        assert provider._agent_id == "hermes"


# ---------------------------------------------------------------------------
# agent_identity (profile name) -> agent_id mapping
# ---------------------------------------------------------------------------


class TestAgentIdentityMapping:
    """`agent_identity` kwarg (profile name) becomes the default agent_id."""

    def test_agent_identity_sets_agent_id_when_no_explicit_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session", agent_identity="seo-expert")

        assert provider._agent_id == "seo-expert"

    def test_explicit_env_agent_id_wins_over_agent_identity(self, monkeypatch, tmp_path):
        """A user who pinned MEM0_AGENT_ID expects it to win."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_AGENT_ID", "pinned-by-env")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session", agent_identity="seo-expert")

        assert provider._agent_id == "pinned-by-env"

    def test_explicit_json_agent_id_wins_over_agent_identity(self, monkeypatch, tmp_path):
        """mem0.json's agent_id is an explicit user choice — outranks profile."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "mem0.json").write_text(json.dumps({"agent_id": "pinned-by-json"}))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session", agent_identity="seo-expert")

        assert provider._agent_id == "pinned-by-json"

    def test_explicit_yaml_agent_id_wins_over_agent_identity(self, monkeypatch, tmp_path):
        """`memory.mem0.agent_id` in config.yaml outranks runtime profile."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            agent_identity="seo-expert",
            provider_config={"agent_id": "pinned-by-yaml"},
        )

        assert provider._agent_id == "pinned-by-yaml"

    def test_no_explicit_no_identity_falls_back_to_hermes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")

        assert provider._agent_id == "hermes"


# ---------------------------------------------------------------------------
# user_id precedence (runtime kwarg always wins)
# ---------------------------------------------------------------------------


class TestUserIdPrecedence:
    """user_id keeps existing precedence: runtime kwarg > config > default."""

    def test_runtime_user_id_wins_over_yaml(self, monkeypatch, tmp_path):
        """Gateway-supplied user_id (e.g. Telegram chat id) outranks yaml."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            user_id="tg_user_42",
            provider_config={"user_id": "jereme"},
        )

        assert provider._user_id == "tg_user_42"

    def test_yaml_user_id_wins_over_env(self, monkeypatch, tmp_path):
        """config.yaml is a stronger signal than env defaults."""
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_USER_ID", "env-user")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"user_id": "yaml-user"},
        )

        assert provider._user_id == "yaml-user"

    def test_json_user_id_wins_over_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_USER_ID", "env-user")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "mem0.json").write_text(json.dumps({"user_id": "json-user"}))

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")

        assert provider._user_id == "json-user"


# ---------------------------------------------------------------------------
# Filter outputs reflect threaded values
# ---------------------------------------------------------------------------


class TestFilterOutputsAfterThreading:
    """Once provider_config sets user_id/agent_id, downstream filters reflect them.

    This is the end-to-end guarantee: a profile that sets
    ``memory.mem0.user_id: jereme`` actually writes under that user.
    """

    def test_write_filters_use_yaml_threaded_ids(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"user_id": "jereme", "agent_id": "stack"},
        )

        wf = provider._write_filters()
        assert wf == {"user_id": "jereme", "agent_id": "stack"}

    def test_read_filters_use_yaml_threaded_user(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = Mem0MemoryProvider()
        provider.initialize(
            "test-session",
            provider_config={"user_id": "jereme"},
        )

        rf = provider._read_filters()
        assert rf == {"user_id": "jereme"}

    def test_profile_only_yaml_agent_id_yields_per_profile_attribution(
        self, monkeypatch, tmp_path
    ):
        """End-to-end multi-profile scenario:

        Three Hermes profiles share user_id=jereme but each writes under its
        own agent_id (= profile name) via the agent_identity runtime kwarg.
        """
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        monkeypatch.delenv("MEM0_AGENT_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        common_yaml = {"user_id": "jereme"}

        stack = Mem0MemoryProvider()
        stack.initialize("s", provider_config=common_yaml, agent_identity="stack")

        seo = Mem0MemoryProvider()
        seo.initialize("s", provider_config=common_yaml, agent_identity="seo-expert")

        staff = Mem0MemoryProvider()
        staff.initialize("s", provider_config=common_yaml, agent_identity="staff-agent")

        assert stack._write_filters() == {"user_id": "jereme", "agent_id": "stack"}
        assert seo._write_filters() == {"user_id": "jereme", "agent_id": "seo-expert"}
        assert staff._write_filters() == {"user_id": "jereme", "agent_id": "staff-agent"}

        # Read filters share the user but ignore agent_id — so cross-profile
        # recall sees everything under user_id=jereme, regardless of which
        # profile wrote it.
        for p in (stack, seo, staff):
            assert p._read_filters() == {"user_id": "jereme"}
