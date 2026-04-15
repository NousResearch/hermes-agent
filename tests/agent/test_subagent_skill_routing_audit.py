from pathlib import Path
import importlib.util

from agent.subagent_profiles import get_subagent_profile


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_subagent_skill_routing.py"
_spec = importlib.util.spec_from_file_location("audit_subagent_skill_routing", _SCRIPT_PATH)
_audit = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_audit)


def test_build_audit_report_includes_profile_rows(monkeypatch):
    monkeypatch.setattr(
        _audit,
        "recommend_skills_for_profile",
        lambda profile, limit=5: [
            {"skill_name": profile.gstack_skill_hints[0], "reasons": ["gstack_hint"]}
        ]
        if profile.gstack_skill_hints
        else [{"skill_name": f"{profile.id}-baseline", "reasons": ["category_match"]}],
    )

    report = _audit.build_audit_report(limit=3)

    reviewer = next(row for row in report["profiles"] if row["profile"] == "reviewer")
    assert reviewer["top_skills"]
    assert reviewer["top_gstack_skills"] == [get_subagent_profile("reviewer").gstack_skill_hints[0]]
    assert reviewer["weak_match_count"] == 0
    assert "generated_from" in report


def test_build_audit_report_flags_generic_gstack_and_weak_matches(monkeypatch):
    monkeypatch.setattr(
        _audit,
        "list_subagent_profiles",
        lambda: [get_subagent_profile("reviewer")],
    )
    monkeypatch.setattr(
        _audit,
        "recommend_skills_for_profile",
        lambda profile, limit=5: [
            {"skill_name": "gstack", "reasons": ["gstack_surface", "generic_gstack_penalty"]},
            {"skill_name": "weak-skill", "reasons": ["weak_metadata_match_penalty"]},
        ],
    )

    report = _audit.build_audit_report(limit=2)

    reviewer = report["profiles"][0]
    assert "generic_gstack_top_result" in reviewer["warnings"]
    assert "weak_recommendation_reasons" in reviewer["warnings"]
    assert reviewer["weak_match_count"] == 2


def test_render_text_shows_warning_state():
    report = {
        "profiles": [
            {
                "profile": "reviewer",
                "description": "Review things",
                "top_skills": ["gstack-review"],
                "top_gstack_skills": ["gstack-review"],
                "excluded_skills_count": 0,
                "warnings": ["missing_gstack_match"],
                "weak_match_count": 1,
            }
        ],
        "warnings": ["reviewer: missing_gstack_match"],
    }

    text = _audit._render_text(report)

    assert "Subagent skill routing audit" in text
    assert "[reviewer]" in text
    assert "missing_gstack_match" in text
