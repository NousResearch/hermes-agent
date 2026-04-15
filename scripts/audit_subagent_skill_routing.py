#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.subagent_profiles import list_subagent_profiles
from tools.skills_tool import recommend_skills_for_profile


def build_audit_report(*, limit: int = 5) -> Dict[str, Any]:
    profiles_report: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for profile in list_subagent_profiles():
        recommended = recommend_skills_for_profile(profile, limit=limit)
        top_skills = [entry.get("skill_name") or entry.get("name") for entry in recommended]
        top_gstack_skills = [name for name in top_skills if isinstance(name, str) and name.startswith("gstack-")]
        excluded_skills = len(profile.excluded_skill_names) + len(profile.excluded_skill_categories)
        profile_warnings: List[str] = []
        weak_matches = [
            entry for entry in recommended
            if not any(
                reason in (entry.get("reasons") or [])
                for reason in ("preferred_name", "gstack_hint", "tag_match", "category_match")
            )
        ]
        generic_gstack_hits = [name for name in top_skills if name == "gstack"]

        if not recommended:
            profile_warnings.append("no_recommended_skills")
        if profile.gstack_skill_hints and not top_gstack_skills:
            profile_warnings.append("missing_gstack_match")
        if not profile.preferred_skill_names and not profile.preferred_tags and not profile.preferred_skill_categories:
            profile_warnings.append("weak_profile_metadata")
        if generic_gstack_hits:
            profile_warnings.append("generic_gstack_top_result")
        if recommended and len(weak_matches) >= max(1, len(recommended) // 2):
            profile_warnings.append("weak_recommendation_reasons")

        if profile_warnings:
            warnings.append(f"{profile.id}: {', '.join(profile_warnings)}")

        profiles_report.append(
            {
                "profile": profile.id,
                "description": profile.description,
                "top_skills": top_skills,
                "top_gstack_skills": top_gstack_skills,
                "excluded_skills_count": excluded_skills,
                "warnings": profile_warnings,
                "weak_match_count": len(weak_matches),
            }
        )

    return {
        "profiles": profiles_report,
        "warnings": warnings,
        "generated_from": str(Path(__file__).resolve()),
    }


def _render_text(report: Dict[str, Any]) -> str:
    lines = ["Subagent skill routing audit", ""]
    for profile in report["profiles"]:
        lines.append(f"[{profile['profile']}] {profile['description']}")
        lines.append(f"  top_skills: {', '.join(profile['top_skills']) if profile['top_skills'] else '(none)'}")
        lines.append(
            f"  top_gstack_skills: {', '.join(profile['top_gstack_skills']) if profile['top_gstack_skills'] else '(none)'}"
        )
        lines.append(f"  excluded_skills_count: {profile['excluded_skills_count']}")
        lines.append(f"  warnings: {', '.join(profile['warnings']) if profile['warnings'] else '(none)'}")
        lines.append("")
    if report["warnings"]:
        lines.append("Global warnings:")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("Global warnings: none")
    return "\n".join(lines)


def main() -> None:
    report = build_audit_report()
    print(_render_text(report))
    print("\nJSON:\n" + json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
