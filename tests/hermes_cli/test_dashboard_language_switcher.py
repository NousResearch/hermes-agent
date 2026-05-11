"""Static dashboard tests for dashboard plugin and language switcher regressions."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sidebar_language_switcher_opens_upward_like_theme_switcher():
    app_tsx = ROOT / "web" / "src" / "App.tsx"

    content = app_tsx.read_text(encoding="utf-8")

    assert "<ThemeSwitcher dropUp />" in content
    assert "<LanguageSwitcher dropUp />" in content


def test_language_switcher_supports_drop_up_positioning():
    switcher_tsx = ROOT / "web" / "src" / "components" / "LanguageSwitcher.tsx"

    content = switcher_tsx.read_text(encoding="utf-8")

    assert "dropUp" in content
    assert 'dropUp ? "left-0 bottom-full mb-1" : "right-0 top-full mt-1"' in content


def test_achievements_plugin_uses_dashboard_fetch_client_for_auth():
    """Plugin API calls must include the dashboard session token via SDK.fetchJSON."""
    bundle = ROOT / "plugins" / "hermes-achievements" / "dashboard" / "dist" / "index.js"

    content = bundle.read_text(encoding="utf-8")

    assert "async function api(path, options)" in content
    assert "return SDK.fetchJSON(url, options || {});" in content
    assert "const res = await fetch(url, options || {});" not in content
