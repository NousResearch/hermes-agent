"""setup_skipped branch should still surface remote setup_note text."""

from agent.skill_commands import _build_skill_message


def test_setup_skipped_includes_remote_setup_note():
    msg = _build_skill_message(
        loaded_skill={
            "content": "Skill body.",
            "setup_skipped": True,
            "setup_note": "Remote backend needs HERMES_HOME mounted.",
        },
        skill_dir=None,
        activation_note="Activating skill.",
    )
    assert "Required environment setup was skipped" in msg
    assert "Remote backend needs HERMES_HOME mounted." in msg


def test_setup_skipped_without_extra_note():
    msg = _build_skill_message(
        loaded_skill={
            "content": "Skill body.",
            "setup_skipped": True,
        },
        skill_dir=None,
        activation_note="Activating skill.",
    )
    assert "Required environment setup was skipped" in msg
