"""Unit tests for hermes_cli.xai_retirement (May 15, 2026 model retirement)."""
from __future__ import annotations


from hermes_cli.xai_retirement import (
    MIGRATION_GUIDE_URL,
    RETIREMENT_DATE,
    RetirementIssue,
    _RETIRED_MODELS,
    _looks_like_xai,
    _normalize,
    find_retired_xai_refs,
    format_issue,
)


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

class TestFormatIssue:
    def test_basic_format(self):
        issue = RetirementIssue(
            config_path="principal.model",
            current_model="grok-3",
            replacement="grok-4.3",
        )
        s = format_issue(issue)
        assert "principal.model" in s
        assert "'grok-3'" in s
        assert "'grok-4.3'" in s


    def test_includes_note_when_set(self):
        issue = RetirementIssue(
            config_path="principal.model",
            current_model="grok-3",
            replacement="grok-4.3",
            note="ambiguous variant",
        )
        s = format_issue(issue)
        assert "[note: ambiguous variant]" in s


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_retirement_date_is_may_15(self):
        assert "May 15, 2026" == RETIREMENT_DATE

    def test_migration_guide_url_points_to_xai(self):
        assert MIGRATION_GUIDE_URL.startswith("https://docs.x.ai/")
        assert "may-15" in MIGRATION_GUIDE_URL.lower()

    def test_retired_models_keyset_matches_doc(self):
        # Snapshot test: if xAI's list changes we want CI to flag it.
        expected = {
            "grok-4-0709",
            "grok-4-fast-reasoning",
            "grok-4-fast-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "grok-code-fast-1",
            "grok-3",
            "grok-imagine-image-pro",
        }
        assert set(_RETIRED_MODELS.keys()) == expected
