from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.model_activation import (
    ModelActivationCASMismatch,
    ModelActivationError,
    ModelRoute,
    activate_model_profile,
    route_fingerprint,
)


def _write_config(home: Path) -> None:
    (home / "config.yaml").write_text(
        """
model:
  default: anthropic/claude-fable-5
  main: anthropic/claude-fable-5
  provider: anthropic
  api_mode: anthropic_messages
  routing_generation: 7
  base_url: ${CLAUDE_PROXY_URL}
  api_key: ${CLAUDE_PROXY_KEY}
  context_length: 200000
providers:
  codex:
    name: Codex Pool
    base_url: https://api.openai.com/v1
    transport: openai_responses
""".lstrip(),
        encoding="utf-8",
    )


def test_activation_writes_whole_route_and_preserves_templates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    expected = ModelRoute(
        model="anthropic/claude-fable-5",
        provider="anthropic",
        api_mode="anthropic_messages",
    )

    result = activate_model_profile(
        {
            "model": "codex/gpt-5.6-sol",
            "provider": "codex",
            "api_mode": "codex_responses",
        },
        expected_current=expected,
        expected_fingerprint=route_fingerprint(expected),
        expected_generation=7,
    )

    raw_text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    assert raw["model"] == {
        "default": "codex/gpt-5.6-sol",
        "main": "codex/gpt-5.6-sol",
        "provider": "codex",
        "api_mode": "codex_responses",
        "routing_generation": 8,
    }
    assert "${CLAUDE_PROXY" not in raw_text
    assert raw["providers"]["codex"]["base_url"] == "https://api.openai.com/v1"
    assert result.old_route == expected
    assert result.generation == 8
    assert result.runtime_rollover_required is True


def test_activation_rejects_stale_cas_without_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    before = (tmp_path / "config.yaml").read_bytes()

    with pytest.raises(ModelActivationCASMismatch):
        activate_model_profile(
            ModelRoute("codex/gpt-5.6-sol", "codex", "codex_responses"),
            expected_generation=6,
        )

    assert (tmp_path / "config.yaml").read_bytes() == before


def test_activation_rejects_provider_transport_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    with pytest.raises(ModelActivationError, match="requires api_mode codex_responses"):
        activate_model_profile(
            ModelRoute("codex/gpt-5.6-sol", "codex", "chat_completions"),
            expected_generation=7,
        )


def test_gateway_cache_watches_atomic_routing_generation():
    source = (Path(__file__).parents[2] / "gateway" / "run.py").read_text(
        encoding="utf-8"
    )
    assert '("model", "routing_generation")' in source
