"""Static dashboard tests for the Profiles navigation copy."""
from pathlib import Path


def test_profiles_nav_label_uses_short_multi_agent_copy():
    en_i18n = Path(__file__).resolve().parents[2] / "web" / "src" / "i18n" / "en.ts"

    content = en_i18n.read_text(encoding="utf-8")

    assert 'profiles: "Profiles : multi-agent"' in content
    assert "Profiles: Running Multiple Agents" not in content


def test_profiles_nav_label_is_localized_in_japanese():
    ja_i18n = Path(__file__).resolve().parents[2] / "web" / "src" / "i18n" / "ja.ts"

    content = ja_i18n.read_text(encoding="utf-8")

    assert 'profiles: "プロファイル：エージェント"' in content
    assert 'profiles: "profiles : multi agents"' not in content
    assert 'profiles: "プロファイル : マルチエージェント"' not in content
