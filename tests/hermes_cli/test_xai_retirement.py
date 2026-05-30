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

    def test_strips_xai_prefix(self):
        assert _normalize("xai/grok-4-fast") == "grok-4-fast"

    def test_lowercases(self):
        assert _normalize("Grok-Code-Fast-1") == "grok-code-fast-1"

    def test_no_prefix_passthrough(self):
        assert _normalize("grok-4.3") == "grok-4.3"

    def test_strips_whitespace(self):
        assert _normalize("  grok-4  ") == "grok-4"


class TestLooksLikeXai:
    def test_grok_prefix(self):
        assert _looks_like_xai("grok-4")
        assert _looks_like_xai("x-ai/grok-4-1-fast")

    def test_non_grok_returns_false(self):
        assert not _looks_like_xai("gpt-4")
        assert not _looks_like_xai("claude-sonnet-4-6")
        assert not _looks_like_xai("openrouter/openai/gpt-4")

    def test_none_or_empty(self):
        assert not _looks_like_xai(None)
        assert not _looks_like_xai("")
        assert not _looks_like_xai("   ")

    def test_non_string(self):
        assert not _looks_like_xai(42)
        assert not _looks_like_xai({"model": "grok-4"})


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

    def test_xai_valid_model_not_flagged(self):
        cfg = {
            "principal": {"model": "grok-4.3"},
            "auxiliary": {
                "vision": {"model": "grok-4.20-0309-reasoning"},
                "fast": {"model": "grok-4-fast"},
                "fast_1": {"model": "grok-4-1-fast"},
                "bare": {"model": "grok-4"},
            },
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

    def test_principal_with_x_ai_prefix(self):
        cfg = {"principal": {"model": "x-ai/grok-4-1-fast-non-reasoning"}}
        issues = find_retired_xai_refs(cfg)
        assert len(issues) == 1
        assert issues[0].current_model == "x-ai/grok-4-1-fast-non-reasoning"
        assert issues[0].replacement == "grok-4.3"
        assert issues[0].reasoning_effort == "none"

    def test_auxiliary_multiple_slots(self):
        cfg = {
            "auxiliary": {
                "vision":      {"model": "grok-4-fast-reasoning"},
                "compression": {"model": "grok-code-fast-1"},
                "curator":     {"model": "grok-4.3"},  # not retired
                "approval":    {"model": "gpt-4o-mini"},  # not xAI
            }
        }
        issues = find_retired_xai_refs(cfg)
        assert sorted(_paths(issues)) == [
            "auxiliary.compression.model",
            "auxiliary.vision.model",
        ]

    def test_auxiliary_unknown_slot_still_scanned(self):
        cfg = {"auxiliary": {"future_slot_xyz": {"model": "grok-3"}}}
        issues = find_retired_xai_refs(cfg)
        assert len(issues) == 1
        assert issues[0].config_path == "auxiliary.future_slot_xyz.model"

    def test_delegation_retired(self):
        cfg = {"delegation": {"model": "grok-4-fast-reasoning"}}
        issues = find_retired_xai_refs(cfg)
        assert _paths(issues) == ["delegation.model"]

    def test_tts_xai_retired(self):
        cfg = {"tts": {"xai": {"model": "grok-imagine-image-pro"}}}
        issues = find_retired_xai_refs(cfg)
        assert _paths(issues) == ["tts.xai.model"]
        assert issues[0].replacement == "grok-imagine-image-quality"

    def test_image_gen_plugin_retired(self):
        cfg = {
            "plugins": {
                "image_gen": {
                    "xai": {"model": "grok-imagine-image-pro"}
                }
            }
        }
        issues = find_retired_xai_refs(cfg)
        assert _paths(issues) == ["plugins.image_gen.xai.model"]
        assert issues[0].replacement == "grok-imagine-image-quality"

    def test_full_trap_config(self):
        cfg = {
            "principal":  {"model": "grok-4-1-fast-non-reasoning"},
            "auxiliary":  {"vision": {"model": "grok-4-fast-reasoning"}},
            "delegation": {"model": "grok-code-fast-1"},
            "tts":        {"xai": {"model": "grok-3"}},  # text model in TTS slot, but valid path
            "plugins": {"image_gen": {"xai": {"model": "grok-imagine-image-pro"}}},
        }
        issues = find_retired_xai_refs(cfg)
        assert len(issues) == 5


# ---------------------------------------------------------------------------
# Migration semantics
# ---------------------------------------------------------------------------

class TestMigrationSemantics:
    def test_non_reasoning_variant_recommends_reasoning_effort_none(self):
        cfg = {"principal": {"model": "grok-4-fast-non-reasoning"}}
        issue = find_retired_xai_refs(cfg)[0]
        assert issue.reasoning_effort == "none"

    def test_reasoning_variant_no_extra_param(self):
        cfg = {"principal": {"model": "grok-4-1-fast-reasoning"}}
        issue = find_retired_xai_refs(cfg)[0]
        assert issue.reasoning_effort is None

    def test_grok_3_maps_to_grok_4_3(self):
        cfg = {"principal": {"model": "grok-3"}}
        issue = find_retired_xai_refs(cfg)[0]
        assert issue.replacement == "grok-4.3"

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

    def test_includes_reasoning_effort_when_set(self):
        issue = RetirementIssue(
            config_path="principal.model",
            current_model="grok-4-fast-non-reasoning",
            replacement="grok-4.3",
            reasoning_effort="none",
        )
        s = format_issue(issue)
        assert 'reasoning_effort: "none"' in s

    def test_omits_reasoning_effort_when_none(self):
        issue = RetirementIssue(
            config_path="principal.model",
            current_model="grok-code-fast-1",
            replacement="grok-4.3",
            reasoning_effort=None,
        )
        s = format_issue(issue)
        assert "reasoning_effort" not in s

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
