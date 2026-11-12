"""The messaging gateway's skill slash path stays filesystem-only: plugin skills
(``ctx.register_skill``) are offered on interactive surfaces (CLI/TUI/Desktop) only."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource


def _make_plugin_skill(home: Path, plugin_name: str, skill: str, body: str) -> None:
    plugin = home / "plugins" / plugin_name
    md = plugin / "skills" / skill / "SKILL.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {plugin_name}\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"def register(ctx):\n"
        f"    ctx.register_skill({skill!r}, Path(__file__).parent / 'skills' / {skill!r} / 'SKILL.md')\n", encoding="utf-8"
    )
    md.write_text(f"---\nname: {skill}\ndescription: Guide\n---\n{body}\n", encoding="utf-8")
    (home / "config.yaml").write_text(f"plugins:\n  enabled: [{plugin_name}]\n", encoding="utf-8")


def _make_local_skill(tmp_path: Path, name: str, body: str) -> None:
    skills = tmp_path / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / name).mkdir(exist_ok=True)
    (skills / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\n{body}\n", encoding="utf-8")


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
            event = SimpleNamespace(
                text=event_text,
                get_command_args=lambda: event_text.split(maxsplit=1)[1] if len(event_text.split(maxsplit=1)) > 1 else "",
            )
            reply = runner._hm_skill_slash_rewrite(event, source, "qk", event_text[1:].split()[0])
            return reply, event
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


def test_messaging_stacked_skills_never_load_plugin_skill_bodies(tmp_path, monkeypatch):
    """Plugin skills are interactive-only: ``/local-skill /plugin:guide do it`` on a messaging
    platform loads only the local skill, and a leading ``/plugin:guide`` bounces as unknown."""
    plugin_home = tmp_path / "home"
    plugin_home.mkdir()
    _make_local_skill(tmp_path, "local-skill", "Local body.")
    _make_plugin_skill(plugin_home, "stack-probe", "guide", "Plugin body.")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
        reply, event = _rewrite("/local-skill /stack-probe:guide do it", tmp_path, monkeypatch, plugin_home)
        assert reply is None  # rewrote the event, did not bounce it as unknown
        assert "Local body." in event.text and "do it" in event.text
        assert "Plugin body." not in event.text
        reply, _event = _rewrite("/stack-probe:guide do it", tmp_path, monkeypatch, plugin_home)
        assert reply is not None and "Unknown command" in reply
