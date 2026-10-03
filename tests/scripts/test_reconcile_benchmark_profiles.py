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


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    profiles = home / "profiles"
    profiles.mkdir()
    monkeypatch.setattr(_MODULE, "INHERITED_ROUTES", {})
    monkeypatch.setattr(_MODULE, "AUXILIARY_POLICY", {"vision": ("ollama-launch", "qwen3.5:4b", "none", 1)})
    p = profiles / "worker"
    p.mkdir()
    (p / "benchmark_profile.json").write_text('{"route":{"primary":{"provider":"openai-codex","model":"gpt-5.5","reasoning_effort":"low"}}}', encoding="utf-8")
    (p / "SOUL.md").write_text("A synthetic dedicated worker soul. " * 10, encoding="utf-8")
    config = "model:\n  provider: openrouter\n  default: old-model\n  reasoning_effort: none\nauxiliary:\n  vision:\n    provider: openrouter\n    model: old-model\n    reasoning_effort: none\n    max_concurrency: 9\n"
    (home / "config.yaml").write_text(config, encoding="utf-8")
    (p / "config.yaml").write_text(config, encoding="utf-8")
    return home, p


def test_apply_backs_up_preimages_before_any_rewrite(tmp_path, monkeypatch):
    import sys
    home, profile = _home(tmp_path, monkeypatch)
    backup = tmp_path / "dated-backup"
    original = {p.relative_to(home): p.read_bytes() for p in (home / "config.yaml", profile / "config.yaml")}
    real = _MODULE._rewrite_config
    def rewrite(path, route):
        for relative, content in original.items():
            assert (backup / relative).read_bytes() == content
        return real(path, route)
    monkeypatch.setattr(_MODULE, "_rewrite_config", rewrite)
    monkeypatch.setattr(sys, "argv", ["reconcile", "--hermes-home", str(home), "--apply", "--backup-path", str(backup)])
    assert _MODULE.main() == 0
    assert yaml.safe_load((home / "benchmark_policy.json").read_text(encoding="utf-8"))["durability"]["backup_path"] == str(backup)
    before = (home / "config.yaml").read_bytes()
    import pytest
    with pytest.raises(FileExistsError):
        _MODULE.main()
    assert (home / "config.yaml").read_bytes() == before


def test_missing_auxiliary_route_refuses_apply_without_mutation(tmp_path, monkeypatch):
    import sys
    import pytest
    home, profile = _home(tmp_path, monkeypatch)
    root = home / "config.yaml"
    root.write_text("model:\n  provider: openrouter\n  default: old-model\n  reasoning_effort: none\n", encoding="utf-8")
    originals = [p.read_bytes() for p in (root, profile / "config.yaml")]
    monkeypatch.setattr(sys, "argv", ["reconcile", "--hermes-home", str(home), "--apply"])
    with pytest.raises(ValueError, match="auxiliary"):
        _MODULE.main()
    assert [p.read_bytes() for p in (root, profile / "config.yaml")] == originals
    assert not (home / "benchmark_policy.json").exists()

    # A complete mapping is not proof of successful auxiliary realization.
    for config in (root, profile / "config.yaml"):
        config.write_text("model:\n  provider: openai-codex\n  default: gpt-5.5\n  reasoning_effort: low\nauxiliary:\n  vision:\n    provider: openrouter\n    model: wrong-model\n    reasoning_effort: none\n    max_concurrency: 9\n", encoding="utf-8")
    monkeypatch.setattr(_MODULE, "_rewrite_config", lambda *_: False)
    with pytest.raises(ValueError, match="auxiliary readback mismatch"):
        _MODULE.main()
    assert not (home / "benchmark_policy.json").exists()
