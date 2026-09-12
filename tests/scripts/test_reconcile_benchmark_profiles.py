"""Focused tests for the benchmark profile reconciliation utility."""

from importlib.util import module_from_spec, spec_from_file_location

import yaml


_SPEC = spec_from_file_location(
    "reconcile_benchmark_profiles",
    "scripts/reconcile_benchmark_profiles.py",
)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_rewrite_config_updates_primary_and_auxiliary_routes(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """# Preserve user-authored ordering and comments.
model:
  provider: openrouter
  default: old-model
  reasoning_effort: none
agent:
  reasoning_effort: none
auxiliary:
  review:
    provider: openrouter
    model: old-review
    reasoning_effort: low
    max_concurrency: 9
  vision:
    provider: openrouter
    model: old-vision
    reasoning_effort: low
    max_concurrency: 9
unrelated:
  keep: true
""",
        encoding="utf-8",
    )

    changed = _MODULE._rewrite_config(
        config,
        ("openai-codex", "gpt-5.6-sol", "high"),
    )

    assert changed is True
    text = config.read_text(encoding="utf-8")
    assert "# Preserve user-authored ordering and comments." in text
    assert "unrelated:" in text
    data = yaml.safe_load(text)
    assert data["model"] == {
        "provider": "openai-codex",
        "default": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }
    assert data["agent"]["reasoning_effort"] == "high"
    assert data["auxiliary"]["review"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "high",
        "max_concurrency": 2,
    }
    assert data["auxiliary"]["vision"] == {
        "provider": "ollama-launch",
        "model": "qwen3.5:4b",
        "reasoning_effort": "none",
        "max_concurrency": 1,
    }


def test_rewrite_config_is_idempotent(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """model:
  provider: openai-codex
  default: gpt-5.5
  reasoning_effort: low
""",
        encoding="utf-8",
    )

    route = ("openai-codex", "gpt-5.5", "low")
    assert _MODULE._rewrite_config(config, route) is True
    assert _MODULE._rewrite_config(config, route) is False
