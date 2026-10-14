"""``compression.summary_language`` pins the language of compaction summaries."""

from __future__ import annotations

import pytest

import hermes_yaml as yaml
from agent.context_compressor import ContextCompressor
from hermes_cli.config import _validate_config_key

_FOLLOW_THE_USER = "do not translate or switch to English"


@pytest.fixture
def compressor_with(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def build(compression: dict) -> ContextCompressor:
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"compression": compression}), encoding="utf-8")
        return ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)

    return build


def test_configured_language_replaces_the_follow_the_user_rule(compressor_with):
    prompt = compressor_with({"summary_language": "English"})._build_summary_prompt(
        "[USER]: Привет", 500, None, "", True)
    assert "Write the summary in English, whatever language the turns use" in prompt
    assert _FOLLOW_THE_USER not in prompt


def test_without_user_turns_the_provenance_rule_is_kept(compressor_with):
    prompt = compressor_with({"summary_language": "English"})._build_summary_prompt(
        "[ASSISTANT]: Готово", 500, None, "", False)
    assert "Write the summary in English" in prompt
    assert "do not invent a user" in prompt


def test_unset_language_follows_the_user(compressor_with):
    prompt = compressor_with({"summary_language": ""})._build_summary_prompt("[USER]: Привет", 500, None, "", True)
    assert _FOLLOW_THE_USER in prompt


def test_summary_language_is_a_known_config_key():
    assert _validate_config_key("compression.summary_language")[0]
