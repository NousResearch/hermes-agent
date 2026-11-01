"""Tests for the built-in /seal and /unseal session-handoff commands.

Covers the prompt builders (agent.seal_prompt), the registry wiring that makes
them first-class gateway commands, and the core invariant of the bridge: the
handoff file is keyed by the CHAT (platform + chat id), not the session id, so
the pointer written before /new is still found after it.
"""

import pytest

from agent.seal_prompt import (
    build_seal_prompt,
    build_unseal_prompt,
    handoff_dir,
    handoff_path,
)


class TestHandoffKeying:
    def test_handoff_is_keyed_by_chat_not_session(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Same chat, two different session ids (before/after /new) → one file.
        p1 = handoff_path("telegram:-100123")
        p2 = handoff_path("telegram:-100123")
        assert p1 == p2
        assert p1.parent == tmp_path / "handoffs"

    def test_different_chats_get_different_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert handoff_path("telegram:-100123") != handoff_path("telegram:-100456")
        assert handoff_path("telegram:-100123") != handoff_path("discord:100123")

    def test_chat_key_is_path_safe(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = handoff_path("telegram:../../etc/passwd")
        assert p.parent == tmp_path / "handoffs"
        assert ".." not in p.name and "/" not in p.name

    def test_empty_chat_key_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert handoff_path("").name == "cli.md"

    def test_config_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as cfg_mod

        monkeypatch.setattr(
            cfg_mod, "load_config",
            lambda: {"gateway": {"seal": {"handoff_dir": str(tmp_path / "bridge")}}},
        )
        assert handoff_dir() == tmp_path / "bridge"


class TestBuildSealPrompt:
    def test_prompt_carries_handoff_path_and_fields(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        prompt = build_seal_prompt("", session_id="sess_123", chat_key="telegram:-100123")
        assert str(tmp_path / "handoffs" / "telegram_-100123.md") in prompt
        for field in ("skill:", "ssot:", "open:", "sealed_session:", "consumed:"):
            assert field in prompt
        assert "sess_123" in prompt

    def test_operator_note_appended(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        prompt = build_seal_prompt("keep the BOM verdict", session_id="s", chat_key="c")
        assert "keep the BOM verdict" in prompt

    def test_no_note_no_operator_section(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        prompt = build_seal_prompt("   ", session_id="s", chat_key="c")
        assert "Operator note" not in prompt


class TestBuildUnsealPrompt:
    def test_prompt_points_at_same_file_as_seal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        key = "telegram:-100123"
        seal = build_seal_prompt("", session_id="s", chat_key=key)
        unseal = build_unseal_prompt("", session_id="fresh_sess", chat_key=key)
        path = str(handoff_path(key))
        assert path in seal and path in unseal

    def test_unseal_forbids_transcript_replay(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        prompt = build_unseal_prompt("", session_id="s", chat_key="c")
        assert "consumed: yes" in prompt
        assert "never 'load the old session'" in prompt


class TestSealRegistryWiring:
    @pytest.mark.parametrize("name", ["seal", "unseal"])
    def test_registered_and_gateway_only(self, name):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

        cmd = resolve_command(name)
        assert cmd is not None
        assert cmd.name == name
        assert cmd.gateway_only is True
        assert name in GATEWAY_KNOWN_COMMANDS

    def test_dispatch_table_lists_both(self):
        from gateway.run_inbound import GatewayInboundMixin

        assert {"seal", "unseal"} <= GatewayInboundMixin._HM_CANONICAL_COMMANDS
        for name in ("seal", "unseal"):
            assert callable(getattr(GatewayInboundMixin, f"_hm_cmd_{name}"))
