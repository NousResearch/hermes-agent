from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.skill_commands import (
    build_skill_invocation_message,
    resolve_natural_skill_invocation,
    scan_skill_commands,
)


def _make_skill(skills_dir, name, body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Description for {name}.
---
# {name}

{body}
""",
        encoding="utf-8",
    )
    return skill_dir


class TestUltragoalNaturalSkillRouting:
    def test_exact_ultragoal_command_routes_to_ingress_skill(self):
        route = resolve_natural_skill_invocation(
            "ULTRAGOAL로 진행해",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="chris-discord-id",
            authorized_user_ids={"chris-discord-id"},
        )

        assert route is not None
        cmd_key, user_instruction, runtime_note = route
        assert cmd_key == "/kanban-ultragoal-ingress"
        assert user_instruction == "ULTRAGOAL로 진행해"
        assert "Kanban Ultragoal" in runtime_note
        assert "direct-kanban" in runtime_note
        assert "fail closed" in runtime_note.lower()
        assert "Autopilot" in runtime_note

    def test_ultragoal_with_kanban_id_preserves_target_in_instruction(self):
        route = resolve_natural_skill_invocation(
            "BO-203 ultragoal로 진행",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="chris-discord-id",
            authorized_user_ids={"chris-discord-id"},
        )

        assert route is not None
        assert route[0] == "/kanban-ultragoal-ingress"
        assert "BO-203" in route[1]
        assert "target" in route[2].lower()

    def test_similar_non_ultragoal_text_does_not_route(self):
        assert resolve_natural_skill_invocation(
            "ultragoal 얘기 다시 설명해봐",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="chris-discord-id",
            authorized_user_ids={"chris-discord-id"},
        ) is None

    def test_negated_ultragoal_phrase_does_not_route(self):
        assert resolve_natural_skill_invocation(
            "BO-203 ultragoal로 진행 말고 설명해줘",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="chris-discord-id",
            authorized_user_ids={"chris-discord-id"},
        ) is None

    def test_non_chris_sender_cannot_trigger_ultragoal_ingress(self):
        assert resolve_natural_skill_invocation(
            "ULTRAGOAL로 진행해",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="other-discord-id",
            authorized_user_ids={"chris-discord-id"},
        ) is None

    def test_missing_sender_does_not_trigger_ultragoal_ingress(self):
        assert resolve_natural_skill_invocation(
            "ULTRAGOAL로 진행해",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
        ) is None

    def test_spoof_like_sender_does_not_trigger_ultragoal_ingress(self):
        assert resolve_natural_skill_invocation(
            "ULTRAGOAL로 진행해",
            platform="discord",
            chat_type="thread",
            thread_id="1511031451796373514",
            user_id="other-discord-id",
            authorized_user_ids={"chris-discord-id"},
        ) is None

    def test_ultragoal_route_builds_ingress_skill_payload(self, tmp_path):
        _make_skill(
            tmp_path,
            "kanban-ultragoal-ingress",
            body="Load hermes-execution-routing and call kanban-ultragoal only after Kanban authority passes.",
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            route = resolve_natural_skill_invocation(
                "ULTRAGOAL로 진행해",
                platform="discord",
                chat_type="thread",
                thread_id="1511031451796373514",
                user_id="chris-discord-id",
                authorized_user_ids={"chris-discord-id"},
            )
            assert route is not None
            cmd_key, user_instruction, runtime_note = route
            msg = build_skill_invocation_message(
                cmd_key,
                user_instruction,
                runtime_note=runtime_note,
            )

        assert msg is not None
        assert "kanban-ultragoal-ingress" in msg
        assert "ULTRAGOAL로 진행해" in msg
        assert "Kanban Ultragoal" in msg
        assert "direct-kanban" in msg
