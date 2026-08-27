"""Tests for /llm-pipeline slash-command shared logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli import llm_pipeline_switch as lps


class TestParseArgs:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("", None),
            ("   ", None),
            ("status", None),
            ("show", None),
            ("on", {"enabled": True}),
            ("enable", {"enabled": True}),
            ("true", {"enabled": True}),
            ("off", {"enabled": False}),
            ("disable", {"enabled": False}),
            ("false", {"enabled": False}),
            ("providers ollama-launch openai", {"providers": ["ollama-launch", "openai"]}),
            ("providers ollama-launch,openai", {"providers": ["ollama-launch", "openai"]}),
            ("providers Ollama-Launch  ,  OpenAI", {"providers": ["ollama-launch", "openai"]}),
        ],
    )
    def test_parse_known_args(self, raw, expected):
        parsed, errors = lps.parse_args(raw)
        assert errors == []
        assert parsed == expected

    def test_parse_unknown(self):
        parsed, errors = lps.parse_args("random")
        assert parsed is None
        assert len(errors) == 1
        assert "Unknown argument" in errors[0]


class TestCurrentState:
    def test_default_state(self):
        assert lps.get_current_state({}) == (True, [])
        assert lps.get_current_state("not-dict") == (True, [])
        assert lps.get_current_state({"agent": "nope"}) == (True, [])

    def test_normalizes_provider_list(self):
        cfg = {
            "agent": {
                "llm_pipeline": {
                    "enabled": False,
                    "providers": ["Ollama-Launch", "openai", ""],
                }
            }
        }
        assert lps.get_current_state(cfg) == (False, ["ollama-launch", "openai"])

    def test_normalizes_comma_string_providers(self):
        cfg = {
            "agent": {
                "llm_pipeline": {
                    "providers": "Ollama-Launch, openrouter, ",
                }
            }
        }
        assert lps.get_current_state(cfg) == (True, ["ollama-launch", "openrouter"])


class TestSetState:
    def test_set_creates_sections_and_returns_previous_state(self):
        cfg = {}
        old_enabled, old_providers = lps.set_state(
            cfg, enabled=False, providers=["openrouter"]
        )

        assert old_enabled is True
        assert old_providers == []
        assert cfg == {
            "agent": {
                "llm_pipeline": {
                    "enabled": False,
                    "providers": ["openrouter"],
                }
            }
        }

    def test_set_state_updates_only_requested_fields(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": True, "providers": ["x"]}}}
        old_enabled, old_providers = lps.set_state(cfg, enabled=False)

        assert old_enabled is True
        assert old_providers == ["x"]
        assert cfg["agent"]["llm_pipeline"]["enabled"] is False
        assert cfg["agent"]["llm_pipeline"]["providers"] == ["x"]

    def test_set_state_requires_dict_config(self):
        with pytest.raises(TypeError):
            lps.set_state("notadict")  # type: ignore[arg-type]


class TestApply:
    def test_read_only_status_reports_current_state(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": False, "providers": ["openrouter"]}}}
        with patch.object(lps, "_native_available", return_value=True):
            status = lps.apply(cfg, None)

        assert status.success
        assert status.new_enabled is False
        assert status.old_enabled is False
        assert status.new_providers == ["openrouter"]
        assert status.old_providers == ["openrouter"]
        assert "llm-pipeline: off" in status.message
        assert "providers: openrouter" in status.message
        assert "native extension: available" in status.message

    def test_status_reports_native_unavailable(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": False}}}
        with patch.object(lps, "_native_available", return_value=False):
            status = lps.apply(cfg, None)

        assert status.success
        assert "native extension: not available" in status.message
        assert status.requires_new_session is False

    def test_no_change_returns_success_but_noop(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": True, "providers": ["openrouter"]}}}
        status = lps.apply(cfg, {"enabled": True, "providers": ["openrouter"]})

        assert status.success is True
        assert "already on" in status.message
        assert "providers: openrouter" in status.message

    def test_toggle_enabled_updates_state_and_requires_new_session(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": True, "providers": ["openrouter"]}}}
        persisted = {}

        def persist(updated):
            persisted.update(updated)

        status = lps.apply(cfg, {"enabled": False}, persist_callback=persist)

        assert status.success
        assert status.old_enabled is True
        assert status.new_enabled is False
        assert status.requires_new_session is True
        assert cfg["agent"]["llm_pipeline"]["enabled"] is False
        assert persisted["agent"]["llm_pipeline"]["enabled"] is False
        assert "enabled: on -> off" in status.message

    def test_update_providers_updates_state_and_requires_new_session(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": True, "providers": ["openrouter"]}}}
        status = lps.apply(cfg, {"providers": ["ollama-launch", "deepseek"]})

        assert status.success
        assert status.old_providers == ["openrouter"]
        assert status.new_providers == ["ollama-launch", "deepseek"]
        assert cfg["agent"]["llm_pipeline"]["providers"] == ["ollama-launch", "deepseek"]
        assert "providers: openrouter -> ollama-launch, deepseek" in status.message

    def test_persist_failure_reports_error_but_keeps_change(self):
        cfg = {"agent": {"llm_pipeline": {"enabled": True}}}

        def failing_persist(_updated):
            raise RuntimeError("disk full")

        status = lps.apply(cfg, {"enabled": False}, persist_callback=failing_persist)

        assert status.success is False
        assert status.new_enabled is False
        assert status.old_enabled is True
        assert "persist failed" in status.message
        assert "disk full" in status.message
        assert status.requires_new_session is True
        # In-memory state should still reflect the applied mutation.
        assert cfg["agent"]["llm_pipeline"]["enabled"] is False
