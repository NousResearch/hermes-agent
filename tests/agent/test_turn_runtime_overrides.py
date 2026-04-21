"""Tests for agent/turn_runtime_overrides.py — shared merge helpers."""

from agent.turn_runtime_overrides import (
    HardFailReason,
    MergeResult,
    merge_multi_skill_runtime_defaults,
    merge_turn_runtime_defaults,
)


def _reasoning(effort):
    if effort == "none":
        return {"enabled": False}
    return {"enabled": True, "effort": effort}


def _event_names(result: MergeResult):
    return [ev.name for ev in result.events]


class TestMergeTurnRuntimeDefaults:
    def test_reasoning_only_merges_and_does_not_mark_model_locked(self):
        primary = {"model": "anthropic/claude-4.6"}
        skill = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(primary, skill)
        assert result.merged["reasoning_config"] == _reasoning("low")
        assert result.merged["reasoning_effort"] == "low"
        assert result.merged["model"] == "anthropic/claude-4.6"
        assert "model_locked" not in result.merged
        assert result.hard_fail is None
        assert "skill_runtime_defaults.applied" in _event_names(result)

    def test_model_override_marks_route_fixed(self):
        primary = {"model": "anthropic/claude-4.6"}
        skill = {"model": "gpt-5.4-mini", "required": []}
        result = merge_turn_runtime_defaults(primary, skill)
        assert result.merged["model"] == "gpt-5.4-mini"
        assert result.merged["model_locked"] is True
        assert result.merged["routing_reason"] == "skill_fixed"
        assert result.hard_fail is None

    def test_empty_skill_defaults_return_primary_unchanged(self):
        primary = {"model": "m", "other": 1}
        result = merge_turn_runtime_defaults(primary, {})
        assert result.merged == primary
        assert result.events == []
        assert result.hard_fail is None

    def test_reasoning_clamp_drops_skill_when_higher_than_session(self):
        primary = {}
        skill = {
            "reasoning_effort": "high",
            "reasoning_config": _reasoning("high"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(
            primary, skill, session_default_reasoning=_reasoning("low")
        )
        assert "reasoning_config" not in result.merged
        assert (
            "skill_runtime_defaults.clamped_to_session_default"
            in _event_names(result)
        )

    def test_reasoning_clamp_allows_reduction(self):
        primary = {}
        skill = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(
            primary, skill, session_default_reasoning=_reasoning("high")
        )
        assert result.merged["reasoning_config"] == _reasoning("low")

    def test_reasoning_clamp_equal_passes_through(self):
        primary = {}
        skill = {
            "reasoning_effort": "medium",
            "reasoning_config": _reasoning("medium"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(
            primary, skill, session_default_reasoning=_reasoning("medium")
        )
        assert result.merged["reasoning_config"] == _reasoning("medium")

    def test_reasoning_clamp_unknown_session_default_fails_open(self):
        """When session default is unknown, skill value passes through."""
        primary = {}
        skill = {
            "reasoning_effort": "high",
            "reasoning_config": _reasoning("high"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(primary, skill)
        assert result.merged["reasoning_config"] == _reasoning("high")

    def test_explicit_lock_model_beats_skill(self):
        primary = {"model": "anthropic/claude-4.6"}
        skill = {"model": "gpt-5.4-mini", "required": []}
        result = merge_turn_runtime_defaults(
            primary, skill, explicit_lock={"model": "anthropic/claude-4.6"}
        )
        # primary model wins; skill does not mark the route as locked.
        assert result.merged["model"] == "anthropic/claude-4.6"
        assert "model_locked" not in result.merged
        assert (
            "skill_runtime_defaults.overridden_by_explicit"
            in _event_names(result)
        )

    def test_explicit_lock_reasoning_beats_skill(self):
        primary = {"reasoning_config": _reasoning("medium")}
        skill = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(
            primary, skill, explicit_lock={"reasoning_effort": "medium"}
        )
        assert result.merged["reasoning_config"] == _reasoning("medium")
        assert (
            "skill_runtime_defaults.overridden_by_explicit"
            in _event_names(result)
        )

    def test_model_allowlist_accepts_matching_glob(self):
        primary = {}
        skill = {"model": "gpt-5.4-mini", "required": []}
        result = merge_turn_runtime_defaults(
            primary, skill, model_allowlist=["gpt-5.4-*"]
        )
        assert result.merged["model"] == "gpt-5.4-mini"
        assert result.merged["model_locked"] is True

    def test_model_allowlist_denies_non_matching(self):
        primary = {"model": "anthropic/claude-4.6"}
        skill = {"model": "gpt-5.4-mini", "required": []}
        result = merge_turn_runtime_defaults(
            primary, skill, model_allowlist=["anthropic/*"]
        )
        assert result.merged["model"] == "anthropic/claude-4.6"
        assert "model_locked" not in result.merged
        assert (
            "skill_runtime_defaults.model_not_allowlisted"
            in _event_names(result)
        )

    def test_empty_allowlist_accepts_any_model(self):
        primary = {}
        skill = {"model": "any-model", "required": []}
        result = merge_turn_runtime_defaults(
            primary, skill, model_allowlist=[]
        )
        assert result.merged["model"] == "any-model"

    def test_required_model_allowlist_denial_returns_hard_fail(self):
        primary = {"model": "anthropic/claude-4.6"}
        skill = {"model": "gpt-5.4-mini", "required": ["model"]}
        result = merge_turn_runtime_defaults(
            primary, skill, model_allowlist=["anthropic/*"], skill_name="parser"
        )
        assert result.hard_fail is not None
        assert result.hard_fail.skill_name == "parser"
        assert result.hard_fail.failing_field == "model"
        assert result.hard_fail.reason == "model_not_allowlisted"
        assert "skill_runtime_defaults.required_failed" in _event_names(result)

    def test_required_reasoning_clamp_returns_hard_fail(self):
        primary = {}
        skill = {
            "reasoning_effort": "high",
            "reasoning_config": _reasoning("high"),
            "required": ["reasoning_effort"],
        }
        result = merge_turn_runtime_defaults(
            primary,
            skill,
            session_default_reasoning=_reasoning("low"),
            skill_name="precision-job",
        )
        assert result.hard_fail is not None
        assert result.hard_fail.failing_field == "reasoning_effort"
        assert result.hard_fail.reason == "clamped_to_session_default"

    def test_required_success_has_no_hard_fail(self):
        primary = {}
        skill = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": ["reasoning_effort"],
        }
        result = merge_turn_runtime_defaults(primary, skill)
        assert result.hard_fail is None
        assert result.merged["reasoning_config"] == _reasoning("low")

    def test_unsafe_skill_name_is_scrubbed_in_events(self):
        """Control chars in skill_name must not leak into structured logs."""
        skill = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        result = merge_turn_runtime_defaults(
            {}, skill, skill_name="evil\nname: pwn"
        )
        for event in result.events:
            assert event.fields.get("skill_name_sanitized") == ""

    def test_unsafe_skill_name_scrubbed_in_hard_fail(self):
        skill = {
            "reasoning_effort": "high",
            "reasoning_config": _reasoning("high"),
            "required": ["reasoning_effort"],
        }
        result = merge_turn_runtime_defaults(
            {},
            skill,
            session_default_reasoning=_reasoning("low"),
            skill_name="bad name with spaces",
        )
        assert result.hard_fail is not None
        assert result.hard_fail.skill_name == ""


class TestMergeMultiSkillRuntimeDefaults:
    def test_single_skill_passes_through(self):
        merged, warnings = merge_multi_skill_runtime_defaults(
            [
                {
                    "reasoning_effort": "low",
                    "reasoning_config": _reasoning("low"),
                    "required": [],
                }
            ]
        )
        assert merged["reasoning_effort"] == "low"
        assert merged["reasoning_config"] == _reasoning("low")
        assert warnings == []

    def test_two_skills_same_value_merges_cleanly(self):
        defaults = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        merged, warnings = merge_multi_skill_runtime_defaults(
            [defaults, defaults]
        )
        assert merged["reasoning_effort"] == "low"
        assert warnings == []

    def test_conflicting_reasoning_effort_drops_field_with_warning(self):
        merged, warnings = merge_multi_skill_runtime_defaults(
            [
                {
                    "reasoning_effort": "low",
                    "reasoning_config": _reasoning("low"),
                    "required": [],
                },
                {
                    "reasoning_effort": "high",
                    "reasoning_config": _reasoning("high"),
                    "required": [],
                },
            ]
        )
        assert "reasoning_effort" not in merged
        assert "reasoning_config" not in merged
        assert len(warnings) == 1
        assert "reasoning_effort" in warnings[0]

    def test_conflicting_model_drops_with_warning(self):
        merged, warnings = merge_multi_skill_runtime_defaults(
            [
                {"model": "m1", "required": []},
                {"model": "m2", "required": []},
            ]
        )
        assert "model" not in merged
        assert any("model" in w for w in warnings)

    def test_empty_list_returns_empty(self):
        merged, warnings = merge_multi_skill_runtime_defaults([])
        assert merged == {}
        assert warnings == []

    def test_non_dict_entries_are_skipped(self):
        merged, warnings = merge_multi_skill_runtime_defaults(
            [
                {
                    "reasoning_effort": "low",
                    "reasoning_config": _reasoning("low"),
                    "required": [],
                },
                None,  # type: ignore[list-item]
                "bogus",  # type: ignore[list-item]
            ]
        )
        assert merged["reasoning_effort"] == "low"
        assert warnings == []

    def test_required_fields_are_unioned(self):
        merged, _ = merge_multi_skill_runtime_defaults(
            [
                {
                    "reasoning_effort": "low",
                    "reasoning_config": _reasoning("low"),
                    "required": ["reasoning_effort"],
                },
                {"model": "m1", "required": ["model"]},
            ]
        )
        assert set(merged["required"]) == {"reasoning_effort", "model"}

    def test_deterministic_order(self):
        """Same input always produces same output (dependency rule 4 safeguard)."""
        defaults_a = {
            "reasoning_effort": "low",
            "reasoning_config": _reasoning("low"),
            "required": [],
        }
        defaults_b = {"model": "m1", "required": []}
        m1, w1 = merge_multi_skill_runtime_defaults([defaults_a, defaults_b])
        m2, w2 = merge_multi_skill_runtime_defaults([defaults_a, defaults_b])
        assert m1 == m2
        assert w1 == w2
