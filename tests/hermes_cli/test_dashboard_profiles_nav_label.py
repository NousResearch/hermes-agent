"""Static dashboard tests for the Profiles navigation copy."""
from pathlib import Path


def test_profiles_nav_label_uses_short_multi_agents_copy():
    en_i18n = Path(__file__).resolve().parents[2] / "web" / "src" / "i18n" / "en.ts"

    content = en_i18n.read_text(encoding="utf-8")

    assert 'profiles: "profiles : multi agents"' in content
    assert "Profiles: Running Multiple Agents" not in content
