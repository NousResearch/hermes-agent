"""Static/regression tests for dashboard locale switching."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_skills_descriptions_translate_known_source_descriptions_only():
    content = (ROOT / "web" / "src" / "pages" / "SkillsPage.tsx").read_text(encoding="utf-8")

    assert "BUNDLED_SKILL_DESCRIPTION_JA[skill.description] || skill.description" in content
    assert 'if (locale !== "ja") return skill.description;' in content
    assert "SKILL_DESCRIPTION_JA[skill.name]" not in content
    assert "localizeSkillDescription(skill)" not in content


def test_achievements_plugin_fetches_payload_for_current_locale():
    content = (
        ROOT
        / "plugins"
        / "hermes-achievements"
        / "dashboard"
        / "dist"
        / "index.js"
    ).read_text(encoding="utf-8")

    assert 'localStorage.getItem("hermes-locale") === "ja" ? "ja" : "en"' in content
    assert '"locale=" + encodeURIComponent(currentLocale())' in content
    assert 'tr("Unlocked", "解除済み")' in content
    assert 'React.createElement(StatCard, { label: "解除済み"' not in content
