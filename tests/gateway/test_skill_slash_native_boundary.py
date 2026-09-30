"""Regression tests for review finding B2 on the plugin-skills-menu feature.

The messaging gateway's skill slash path is native/filesystem-only: the adapter
must resolve the FIRST token, scan stacked tokens, and load every stacked skill
against the same filesystem map — a ``/plugin:skill`` token must resolve as
unknown (never load a plugin skill body), and the per-platform disabled guard
must see the real names of every skill the stacked loader will load.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def _make_plugin_skill(home: Path, plugin_name: str, skill: str, body: str) -> None:
    plugin = home / "plugins" / plugin_name
    md = plugin / "skills" / skill / "SKILL.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {plugin_name}\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"def register(ctx):\n"
        f"    ctx.register_skill({skill!r}, Path(__file__).parent / 'skills' / {skill!r} / 'SKILL.md')\n"
    )
    md.write_text(f"---\nname: {skill}\ndescription: Guide\n---\n{body}\n")
    (home / "config.yaml").write_text(f"plugins:\n  enabled: [{plugin_name}]\n")


def _make_local_skill(tmp_path: Path, name: str, body: str) -> None:
    skills = tmp_path / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / name).mkdir(exist_ok=True)
    (skills / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\n{body}\n")


def _rewrite(event_text: str, tmp_path, monkeypatch, plugin_home):
    """Drive ``_hm_skill_slash_rewrite`` for a Discord event; returns its reply."""
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setenv("HERMES_HOME", str(plugin_home))
    plugins_mod._reset_plugin_managers_for_tests()
    import agent.skill_commands as sc
    try:
        monkeypatch.setattr(sc, "_skill_commands_by_key", {})
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            from gateway.run_inbound import GatewayInboundMixin

            def _unknown(self, command, source):
                return f"Unknown command: /{command}"

            runner = SimpleNamespace(
                _hm_unknown_slash_reply=_unknown.__get__(SimpleNamespace()),
                _hm_bundle_slash_rewrite=lambda *a, **k: False,
            )
            runner._hm_skill_slash_rewrite = GatewayInboundMixin._hm_skill_slash_rewrite.__get__(runner)
            source = SessionSource(platform=Platform.DISCORD, chat_id="c1")
            event = SimpleNamespace(text=event_text, get_command_args=lambda: "")
            return runner._hm_skill_slash_rewrite(event, source, "qk", event_text[1:].split()[0])
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


class TestGatewayStackedNativeBoundary:
    def test_first_token_plugin_skill_is_unknown(self, tmp_path, monkeypatch):
        """/plugin:guide as the FIRST token resolves filesystem-only: unknown."""
        from hermes_cli import plugins as plugins_mod

        plugin_home = tmp_path / "home"
        plugin_home.mkdir()
        _make_local_skill(tmp_path, "local-skill", "Local body.")
        _make_plugin_skill(plugin_home, "stack-probe", "guide", "Plugin body.")
        plugins_mod._reset_plugin_managers_for_tests()
        monkeypatch.setenv("HERMES_HOME", str(plugin_home))
        import agent.skill_commands as sc
        try:
            with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
                assert sc.resolve_skill_command_key("stack-probe:guide") is None
                assert sc.resolve_skill_command_key("stack-probe:guide", interactive=True) == "/stack-probe:guide"
        finally:
            plugins_mod._reset_plugin_managers_for_tests()

    def test_stacked_trailing_plugin_skill_is_not_loaded(self, tmp_path, monkeypatch):
        """``/local-skill /plugin:guide do it`` must load only the local skill —
        the plugin token must not reach the stacked loader (B2)."""
        plugin_home = tmp_path / "home"
        plugin_home.mkdir()
        _make_local_skill(tmp_path, "local-skill", "Local body.")
        _make_plugin_skill(plugin_home, "stack-probe", "guide", "Plugin body.")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            reply = _rewrite("/local-skill /stack-probe:guide do it", tmp_path, monkeypatch, plugin_home)
        assert reply is None  # rewrote the event, did not bounce it as unknown

    def test_plugin_only_stacked_token_bounces_unknown(self, tmp_path, monkeypatch):
        """A leading ``/plugin:guide`` (no filesystem skill) bounces unknown."""
        plugin_home = tmp_path / "home"
        plugin_home.mkdir()
        _make_plugin_skill(plugin_home, "stack-probe", "guide", "Plugin body.")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            reply = _rewrite("/stack-probe:guide do it", tmp_path, monkeypatch, plugin_home)
        assert reply is not None and "Unknown command" in reply
