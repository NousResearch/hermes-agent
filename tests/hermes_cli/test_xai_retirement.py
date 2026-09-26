"""Unit tests for hermes_cli.xai_retirement (May 15, 2026 model retirement)."""
from __future__ import annotations


import hermes_yaml as yaml

from hermes_cli.xai_retirement import (
    RetirementIssue,
    _RETIRED_MODELS,
    _looks_like_xai,
    _normalize,
    apply_migration,
    find_retired_xai_refs,
)


def test_apply_migration_preserves_long_double_quoted_scalar(tmp_path, monkeypatch):
    """Same fold-after-backslash class as #119844: the migration's own emitter must not mutate
    unrelated long quoted values while it rewrites the model key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    value = "A" * 74 + r"D:\CentBrowserPortable " + "B" * 40
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        'model:\n  provider: xai\n  model: grok-3\napprovals:\n  smart_policy: "'
        + value.replace("\\", "\\\\") + '"\n',
        encoding="utf-8",
    )

    apply_migration(cfg, [RetirementIssue("model.model", "grok-3", "grok-4")], backup=False)

    loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert loaded["model"]["model"] == "grok-4"
    assert loaded["approvals"]["smart_policy"] == value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(issues):
    return [i.config_path for i in issues]

# ---------------------------------------------------------------------------
# _normalize / _looks_like_xai
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strips_x_ai_prefix(self):
        assert _normalize("x-ai/grok-4") == "grok-4"

class TestLooksLikeXai:

    def test_non_grok_returns_false(self):
        assert not _looks_like_xai("gpt-4")
        assert not _looks_like_xai("claude-sonnet-4-6")
        assert not _looks_like_xai("openrouter/openai/gpt-4")

# ---------------------------------------------------------------------------
# find_retired_xai_refs — config scanning
# ---------------------------------------------------------------------------

class TestFindRetiredEdgeCases:
    def test_empty_config_no_issues(self):
        assert find_retired_xai_refs({}) == []

    def test_non_dict_config_returns_empty(self):
        assert find_retired_xai_refs(None) == []  # type: ignore[arg-type]
        assert find_retired_xai_refs("nope") == []  # type: ignore[arg-type]

    def test_no_xai_models_no_issues(self):
        cfg = {
            "principal": {"provider": "openai", "model": "gpt-4o"},
            "auxiliary": {"vision": {"model": "claude-sonnet-4-6"}},
            "delegation": {"model": "openai/o3"},
        }
        assert find_retired_xai_refs(cfg) == []

class TestFindRetiredPerSlot:
    def test_principal_retired(self):
        cfg = {"principal": {"model": "grok-code-fast-1"}}
        issues = find_retired_xai_refs(cfg)
        assert len(issues) == 1
        assert issues[0].config_path == "principal.model"
        assert issues[0].current_model == "grok-code-fast-1"
        assert issues[0].replacement == "grok-4.3"
        assert issues[0].reasoning_effort is None

# ---------------------------------------------------------------------------
# Migration semantics
# ---------------------------------------------------------------------------

class TestMigrationSemantics:

    def test_imagine_pro_maps_to_imagine_quality(self):
        cfg = {"plugins": {"image_gen": {"xai": {"model": "grok-imagine-image-pro"}}}}
        issue = find_retired_xai_refs(cfg)[0]
        assert issue.replacement == "grok-imagine-image-quality"

    def test_all_retired_have_replacement(self):
        for name, entry in _RETIRED_MODELS.items():
            assert entry.get("replacement"), f"{name} has no replacement"

# ---------------------------------------------------------------------------
# format_issue
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------
