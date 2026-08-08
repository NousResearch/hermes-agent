from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.skill_commands import build_preloaded_skills_prompt
from agent.skill_governance import (
    GovernanceClassification,
    evaluate_skill_selection,
    filter_allowed_skill_names,
    governance_context,
    probe_protected_task_class,
    rank_skill_search_results,
)


@dataclass
class _SearchResult:
    name: str
    description: str
    source: str
    identifier: str
    trust_level: str
    extra: dict = field(default_factory=dict)


def _write_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""\
---
name: {name}
description: Test skill for {name}
---

# {name}

Use the {name} workflow.
""",
        encoding="utf-8",
    )


def _configure_governance(home: Path) -> None:
    (home / "governance").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: ardyn_engineering
    protected_task_classes:
      - ardyn_engineering
    retrieval_ranking: true
""",
        encoding="utf-8",
    )
    (home / "governance" / "skills-registry.yaml").write_text(
        """\
version: 1
skills:
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
    aliases: [tooltrust]
    provenance:
      source: legacy-catalog
      lineage: tooltrust-v1
  - name: PREMP
    classification: STALE
    aliases: [premp]
    provenance:
      source: legacy-catalog
      lineage: premp-v1
  - name: ModernCurrent
    classification: CURRENT
    aliases: [modern-current]
    provenance:
      source: qualified-registry
      lineage: current-v3
  - name: ConflictCase
    classification: CONFLICTING
    aliases: [conflict-case]
    provenance:
      source: competing-registry
      lineage: conflict-v2
""",
        encoding="utf-8",
    )


def test_protected_governance_api_rejects_unknown_and_legacy_auto_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)

    auto_ctx = governance_context(mode="auto")
    tooltrust = evaluate_skill_selection("ToolTrust", context=auto_ctx)
    premp = evaluate_skill_selection("PREMP", context=auto_ctx)
    unknown = evaluate_skill_selection("UnregisteredSkill", context=auto_ctx)
    current = evaluate_skill_selection("ModernCurrent", context=auto_ctx)

    assert tooltrust.allowed is False
    assert tooltrust.classification == GovernanceClassification.COMPATIBILITY_ONLY
    assert "historical intent" in tooltrust.reason

    assert premp.allowed is False
    assert premp.classification == GovernanceClassification.STALE

    assert unknown.allowed is False
    assert unknown.classification == GovernanceClassification.UNKNOWN

    assert current.allowed is True
    assert current.classification == GovernanceClassification.CURRENT


def test_compatibility_entry_requires_explicit_historical_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)

    blocked = evaluate_skill_selection(
        "ToolTrust",
        context=governance_context(mode="explicit", historical_intent=False),
    )
    allowed = evaluate_skill_selection(
        "ToolTrust",
        context=governance_context(mode="explicit", historical_intent=True),
    )

    assert blocked.allowed is False
    assert allowed.allowed is True
    assert allowed.reason == "compatibility-only entry allowed by historical intent"


def test_preloaded_skills_fail_closed_for_protected_task_class(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "ToolTrust")
    _write_skill(skills_dir, "PREMP")
    _write_skill(skills_dir, "ModernCurrent")

    prompt, loaded_names, missing = build_preloaded_skills_prompt(
        ["ToolTrust", "PREMP", "ModernCurrent"]
    )

    assert loaded_names == ["ModernCurrent"]
    assert set(missing) == {"ToolTrust", "PREMP"}
    assert "ModernCurrent" in prompt
    assert "ToolTrust" not in prompt
    assert "PREMP" not in prompt


def test_auto_mode_filter_keeps_only_current_entries_for_protected_task_class(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)

    resolved, decisions = filter_allowed_skill_names(
        ["ToolTrust", "PREMP", "ModernCurrent", "ConflictCase"],
        context=governance_context(mode="auto"),
    )

    assert resolved == ["ModernCurrent"]
    assert [decision.classification for decision in decisions] == [
        GovernanceClassification.COMPATIBILITY_ONLY,
        GovernanceClassification.STALE,
        GovernanceClassification.CURRENT,
        GovernanceClassification.CONFLICTING,
    ]


def test_retrieval_ranking_filters_denied_entries_for_protected_task_class(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)

    ranked = rank_skill_search_results(
        [
            _SearchResult(
                name="PREMP",
                description="Legacy stale entry",
                source="github",
                identifier="github/premp",
                trust_level="trusted",
                extra={},
            ),
            _SearchResult(
                name="ModernCurrent",
                description="Qualified current entry",
                source="official",
                identifier="official/modern-current",
                trust_level="builtin",
                extra={},
            ),
            _SearchResult(
                name="ToolTrust",
                description="Compatibility-only legacy entry",
                source="github",
                identifier="github/tooltrust",
                trust_level="trusted",
                extra={},
            ),
        ],
        context=governance_context(mode="retrieval"),
    )

    assert [item.name for item in ranked] == ["ModernCurrent"]
    assert ranked[0].extra["governance"]["classification"] == "CURRENT"


def test_retrieval_ranking_preserves_unprotected_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)

    ranked = rank_skill_search_results(
        [
            _SearchResult(
                name="PREMP",
                description="Legacy stale entry",
                source="github",
                identifier="github/premp",
                trust_level="trusted",
                extra={},
            ),
            _SearchResult(
                name="ModernCurrent",
                description="Qualified current entry",
                source="official",
                identifier="official/modern-current",
                trust_level="builtin",
                extra={},
            ),
            _SearchResult(
                name="ToolTrust",
                description="Compatibility-only legacy entry",
                source="github",
                identifier="github/tooltrust",
                trust_level="trusted",
                extra={},
            ),
        ],
        context=governance_context(mode="retrieval", task_class="general"),
    )

    assert [item.name for item in ranked] == ["ModernCurrent", "ToolTrust", "PREMP"]
    assert ranked[0].extra["governance"]["classification"] == "CURRENT"
    assert ranked[1].extra["governance"]["classification"] == "COMPATIBILITY_ONLY"
    assert ranked[2].extra["governance"]["classification"] == "STALE"


def test_protected_task_probe_is_self_contained_when_skill_utils_import_fails(
    tmp_path, monkeypatch
):
    import builtins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _configure_governance(tmp_path)
    real_import = builtins.__import__

    def _deny_skill_utils(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.skill_utils":
            raise ImportError("simulated skill_utils import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _deny_skill_utils)

    probe = probe_protected_task_class()

    assert probe.safe is True
    assert probe.protected_task is True
    assert probe.task_class == "ardyn_engineering"


def test_protected_task_probe_fails_closed_on_config_parse_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("skills: [\n", encoding="utf-8")

    probe = probe_protected_task_class()

    assert probe.safe is False
    assert probe.protected_task is True
