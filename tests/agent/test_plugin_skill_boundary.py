"""Regression tests for review findings B1/B2 on the plugin-skills-menu feature.

B1 — the plugin skill projection must be cached: repeating an interactive lookup
must not re-read plugin SKILL.md files per call (completion RPCs run it per
keystroke). Invalidation goes through ``invalidate_plugin_skill_commands()``
(``/reload-skills``, plugin enable/install) so a stale registration hint still
loses to fresh frontmatter after a lifecycle change.

B2 — the messaging gateway's stacked skill path is native/filesystem-only:
plugin skills are interactive-only (CLI/TUI/desktop), so neither
``split_stacked_skill_commands`` in native mode nor a loader fed the native map
may resolve a ``/plugin:skill`` token.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import plugins


def _make_plugin_skill(home: Path, plugin_name: str, skill: str, body: str, description: str = "") -> Path:
    """Write a real plugin tree registering one skill and enable it in config."""
    plugin = home / "plugins" / plugin_name
    md = plugin / "skills" / skill / "SKILL.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {plugin_name}\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"def register(ctx):\n"
        f"    ctx.register_skill({skill!r}, Path(__file__).parent / 'skills' / {skill!r} / 'SKILL.md')\n"
    )
    desc_line = f"description: {description}\n" if description else ""
    md.write_text(f"---\nname: {skill}\n{desc_line}---\n{body}\n")
    config = home / "config.yaml"
    enabled = plugin_name
    if config.exists():
        text = config.read_text()
        enabled = f"{text.rstrip().rsplit('[', 1)[0].rstrip().rstrip(',')}, {plugin_name}" if False else enabled
    config.write_text(f"plugins:\n  enabled: [{plugin_name}]\n")
    return md


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins._reset_plugin_managers_for_tests()
    try:
        yield home
    finally:
        plugins._reset_plugin_managers_for_tests()


class TestProjectionCache:
    def test_repeat_lookup_does_not_reread_plugin_skill_files(self, plugin_home):
        """B1: a cached interactive lookup must not touch plugin SKILL.md again."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "cache-probe", "guide", "Body.")
        first = sc.get_interactive_skill_commands()
        assert "/cache-probe:guide" in first
        reads = []
        real_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            if self.name == "SKILL.md" and "cache-probe" in str(self):
                reads.append(self)
            return real_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", counting_read_text):
            second = sc.get_interactive_skill_commands()
        assert second == first
        assert reads == []

    def test_invalidate_rebuilds_projection_with_fresh_frontmatter(self, plugin_home):
        """A changed description must reach the palette only after invalidation
        (the cache serves the last built projection until then)."""
        import agent.skill_commands as sc

        md = _make_plugin_skill(plugin_home, "fresh-probe", "guide", "Body.", "Old")
        assert sc.get_interactive_skill_commands()["/fresh-probe:guide"]["description"] == "Old"
        md.write_text("---\nname: guide\ndescription: New\n---\nBody.\n")
        assert sc.get_interactive_skill_commands()["/fresh-probe:guide"]["description"] == "Old"
        sc.invalidate_plugin_skill_commands()
        assert sc.get_interactive_skill_commands()["/fresh-probe:guide"]["description"] == "New"

    def test_profile_switch_self_heals_without_invalidation(self, plugin_home, tmp_path, monkeypatch):
        """The cache is keyed on the resolved home: switching profiles must never
        serve the other profile's plugin skills."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "profile-a", "guide", "Body from A.")
        assert "/profile-a:guide" in sc.get_interactive_skill_commands()

        other = tmp_path / "home-b"
        other.mkdir()
        (other / "config.yaml").write_text("plugins:\n  enabled: []\n")
        monkeypatch.setenv("HERMES_HOME", str(other))
        plugins._reset_plugin_managers_for_tests()
        try:
            commands = sc.get_interactive_skill_commands()
            assert "/profile-a:guide" not in commands
        finally:
            monkeypatch.setenv("HERMES_HOME", str(plugin_home))
            plugins._reset_plugin_managers_for_tests()
        assert "/profile-a:guide" in sc.get_interactive_skill_commands()

    def test_reload_skills_counts_plugin_skills_and_refreshes_registry(self, plugin_home):
        """/reload-skills must rebuild the plugin projection so a plugin enabled
        since the last lookup appears, and the receipt must count it."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "reload-probe", "guide", "Body.")
        config = plugin_home / "config.yaml"
        config.write_text("plugins:\n  enabled: []\n")
        first = sc.get_interactive_skill_commands()
        assert "/reload-probe:guide" not in first  # not enabled yet

        config.write_text("plugins:\n  enabled: [reload-probe]\n")
        plugins.discover_plugins(force=True)
        result = sc.reload_skills()
        assert "/reload-probe:guide" in sc.get_plugin_skill_commands()
        assert result["commands"] == len(sc.get_skill_commands()) + 1


class TestNativeStackedBoundary:
    def test_split_native_mode_never_resolves_plugin_skill_token(self, plugin_home):
        """B2: the messaging gateway's token scan must stay filesystem-only."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "boundary-probe", "guide", "Body.")
        assert "/boundary-probe:guide" in sc.get_interactive_skill_commands()
        keys, instruction = sc.split_stacked_skill_commands("/boundary-probe:guide do it")
        assert keys == []
        assert instruction == "/boundary-probe:guide do it"
        interactive_keys, _ = sc.split_stacked_skill_commands("/boundary-probe:guide do it", interactive=True)
        assert interactive_keys == ["/boundary-probe:guide"]

    def test_stacked_loader_with_native_table_refuses_plugin_skill(self, plugin_home):
        """The gateway loader is fed its native map: a leaked plugin-skill key
        must load nothing (missing), never the plugin body."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "boundary-probe", "guide", "Plugin body.")
        assert sc.build_skill_invocation_message("/boundary-probe:guide", "go") is not None

        result = sc.build_stacked_skill_invocation_message(
            ["/boundary-probe:guide"], "go", table=sc.get_skill_commands(),
        )
        assert result is None

    def test_stacked_loader_defaults_to_interactive_table(self, plugin_home):
        """CLI/TUI callers keep the interactive default: a plugin skill loads."""
        import agent.skill_commands as sc

        _make_plugin_skill(plugin_home, "boundary-probe", "guide", "Plugin body.")
        result = sc.build_stacked_skill_invocation_message(["/boundary-probe:guide"], "go")
        assert result is not None
        assert result[1] == ["boundary-probe:guide"]
