"""Static dashboard tests for the Profiles navigation copy."""
import re
from pathlib import Path


def test_profiles_nav_label_uses_short_copy():
    en_i18n = Path(__file__).resolve().parents[2] / "web" / "src" / "i18n" / "en.ts"

    content = en_i18n.read_text(encoding="utf-8")

    # Nav label should be the clean short form, not the old verbose string
    assert re.search(r"profiles:\s*['\"]Profiles['\"]", content)
    assert "profiles : multi agents" not in content
