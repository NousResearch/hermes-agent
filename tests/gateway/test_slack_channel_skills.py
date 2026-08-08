"""Tests for Slack channel_skill_bindings auto-skill resolution."""
import builtins

from pathlib import Path
from unittest.mock import MagicMock


def _make_adapter(extra=None):
    """Create a minimal SlackAdapter stub with the given ``config.extra``."""
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


def _resolve(adapter, channel_id, parent_id=None):
    from gateway.platforms.base import resolve_channel_skills
    return resolve_channel_skills(adapter.config.extra, channel_id, parent_id)


def _write_governance_config(home: Path, *, task_class: str = "", protected: list[str] | None = None) -> None:
    (home / "governance").mkdir(parents=True, exist_ok=True)
    protected_entries = protected or []
    protected_block = "".join(f"      - {name}\n" for name in protected_entries)
    protected_yaml = protected_block or "      []\n"
    (home / "config.yaml").write_text(
        f"""\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: {task_class}
    protected_task_classes:
{protected_yaml}""",
        encoding="utf-8",
    )


def _write_registry(home: Path, entries: str) -> None:
    (home / "governance" / "skills-registry.yaml").write_text(
        f"version: 1\nskills:\n{entries}",
        encoding="utf-8",
    )


class TestSlackResolveChannelSkills:

    def test_match_by_dm_channel_id(self):
        """The primary use case: binding a skill to a Slack DM channel."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]


    def test_no_match_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") is None

    def test_single_skill_string(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skill": "german-flashcards"},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]


    def test_empty_skills_list_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skills": []},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None

    def test_protected_binding_denied_when_skill_utils_import_fails_and_governance_eval_errors(self, monkeypatch, tmp_path):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["legacy-skill"]},
            ]
        })
        home = tmp_path / "home"
        _write_governance_config(
            home,
            task_class="ardyn_engineering",
            protected=["ardyn_engineering"],
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        real_import = builtins.__import__

        def _deny_governance_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "agent.skill_utils":
                raise ImportError(f"simulated import failure: {name}")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(
            "agent.skill_governance.evaluate_skill_selection",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated governance evaluation failure")
            ),
        )
        monkeypatch.setattr(builtins, "__import__", _deny_governance_import)

        assert _resolve(adapter, "D0ATH9TQ0G6") is None

    def test_protected_binding_denied_when_config_cannot_be_parsed(self, monkeypatch, tmp_path):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["legacy-skill"]},
            ]
        })
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("skills: [\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert _resolve(adapter, "D0ATH9TQ0G6") is None

    def test_protected_binding_keeps_current_and_drops_non_current(self, monkeypatch, tmp_path):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["ToolTrust", "ModernCurrent", "PREMP"]},
            ]
        })
        home = tmp_path / "home"
        _write_governance_config(
            home,
            task_class="ardyn_engineering",
            protected=["ardyn_engineering"],
        )
        _write_registry(
            home,
            """\
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
  - name: ModernCurrent
    classification: CURRENT
  - name: PREMP
    classification: STALE
""",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert _resolve(adapter, "D0ATH9TQ0G6") == ["ModernCurrent"]

    def test_unprotected_binding_keeps_compatibility_skill(self, monkeypatch, tmp_path):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["ToolTrust"]},
            ]
        })
        home = tmp_path / "home"
        _write_governance_config(
            home,
            task_class="general_ops",
            protected=["ardyn_engineering"],
        )
        _write_registry(
            home,
            """\
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
""",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert _resolve(adapter, "D0ATH9TQ0G6") == ["ToolTrust"]


class TestSlackMessageEventAutoSkill:
    """Integration-style test: verify auto_skill propagates to MessageEvent."""

    def test_message_event_carries_auto_skill(self):
        """Simulate the handler wiring: resolve + attach to MessageEvent."""
        from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource, resolve_channel_skills

        config_extra = {
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        }
        auto_skill = resolve_channel_skills(config_extra, "D0ATH9TQ0G6", None)

        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="D0ATH9TQ0G6",
            chat_name="Mats",
            chat_type="dm",
            user_id="U0ABC",
            user_name="Mats",
        )
        event = MessageEvent(
            text="work",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="123.456",
            auto_skill=auto_skill,
        )
        assert event.auto_skill == ["german-flashcards"]
