"""Dashboard user themes: parsing a theme file and discovering ``~/.hermes/dashboard-themes/``."""

import hermes_cli.web_server_dashboard as _web_server_dashboard


class TestNormaliseThemeDefinition:
    """Tests for _normalise_theme_definition() — parses YAML theme files."""


    def test_rejects_non_dict(self):
        from hermes_cli.web_server_dashboard import _normalise_theme_definition
        assert _normalise_theme_definition("string") is None
        assert _normalise_theme_definition(None) is None
        assert _normalise_theme_definition([1, 2, 3]) is None

    def test_loose_colors_shorthand(self):
        """Bare hex strings under `colors` parse as {hex, alpha=1.0}."""
        from hermes_cli.web_server_dashboard import _normalise_theme_definition
        result = _normalise_theme_definition({
            "name": "loose",
            "colors": {"background": "#000000", "midground": "#ffffff"},
        })
        assert result is not None
        assert result["palette"]["background"] == {"hex": "#000000", "alpha": 1.0}
        assert result["palette"]["midground"] == {"hex": "#ffffff", "alpha": 1.0}
        # foreground falls back to default (transparent white)
        assert result["palette"]["foreground"]["hex"] == "#ffffff"
        assert result["palette"]["foreground"]["alpha"] == 0.0


class TestDiscoverUserThemes:
    """Tests for _discover_user_themes() — scans ~/.hermes/dashboard-themes/."""

    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert _web_server_dashboard._discover_user_themes() == []

    def test_loads_and_normalises_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "ocean.yaml").write_text(
            "name: ocean\n"
            "label: Ocean\n"
            "palette:\n"
            "  background:\n"
            "    hex: \"#0a1628\"\n"
            "    alpha: 1.0\n"
            "layout:\n"
            "  density: spacious\n"
        )
        results = _web_server_dashboard._discover_user_themes()
        assert len(results) == 1
        assert results[0]["name"] == "ocean"
        assert results[0]["label"] == "Ocean"
        assert results[0]["palette"]["background"]["hex"] == "#0a1628"
        assert results[0]["layout"]["density"] == "spacious"
        # defaults filled in
        assert "fontSans" in results[0]["typography"]


    def test_ignores_transient_profile_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        themes_dir = tmp_path / "dashboard-themes"
        themes_dir.mkdir()
        (themes_dir / "mine.yaml").write_text("name: mine\n", encoding="utf-8")

        other = tmp_path / "other-profile"
        other.mkdir()

        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(str(other))
        try:
            results = _web_server_dashboard._discover_user_themes()
        finally:
            reset_hermes_home_override(token)

        assert [r["name"] for r in results] == ["mine"]
