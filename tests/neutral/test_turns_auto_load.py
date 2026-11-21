"""Gateway auto-load turns reduced to the text the user typed.

A channel-bound skill auto-loaded on a new session is prepended to the user's
text in one model-facing turn. ``extract_user_instruction_from_skill_message``
(memory providers) and ``describe_skill_invocation`` (previews, titles) keep
only the typed text: the boundary the gateway registered for the turn wins,
and a turn built in another process falls back to the last payload footer.
"""

from collections import OrderedDict

import pytest

from session_rules import turns
from session_rules.turns import (
    _SKILL_DIR_NOTE_END,
    describe_skill_invocation,
    extract_user_instruction_from_skill_message,
)

_TWO_SKILL_TURN = (
    '[IMPORTANT: The "example" skill is auto-loaded. Follow its instructions for this session.]\n'
    "The skill body contains the terminal tool absolute path footer as quoted text.\n"
    f"[Skill directory: /skills/example]\nResolve relative paths, {_SKILL_DIR_NOTE_END}\n\n"
    '[IMPORTANT: The "another" skill is auto-loaded. Follow its instructions for this session.]\n'
    "Second skill body.\n[Skill directory: /skills/another]\nResolve paths, " + _SKILL_DIR_NOTE_END
    + "\n\nnot remember anything i tell you"
)

_SCAFFOLD = (
    '[IMPORTANT: The "example" skill is auto-loaded. Follow its instructions for this session.]\n'
    f"Skill body.\n[Skill directory: /skills/example]\nResolve paths, {_SKILL_DIR_NOTE_END}\n\n"
)

_LOOKALIKE = '[IMPORTANT: The "config" file is broken] please   look'

_PASTED_NOTE = "[Skill setup note: set FOO_TOKEN] why does setup keep failing?"
_PASTED_FOOTER = f"it said {_SKILL_DIR_NOTE_END} what does that mean"


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch):
    monkeypatch.setattr(turns, "_AUTO_LOAD_SCAFFOLDS", OrderedDict(), raising=False)


class TestExtractUserInstruction:
    def test_auto_loaded_skills_keep_only_typed_text(self):
        assert extract_user_instruction_from_skill_message(_TWO_SKILL_TURN) == "not remember anything i tell you"

    def test_bare_auto_load_carries_no_instruction(self):
        assert extract_user_instruction_from_skill_message(_SCAFFOLD) is None

    def test_lookalike_text_kept_unchanged(self):
        assert extract_user_instruction_from_skill_message(_LOOKALIKE) is _LOOKALIKE

    def test_pasted_scaffold_lookalikes_in_typed_text_kept(self):
        turns.register_auto_load_scaffold(_SCAFFOLD)
        assert extract_user_instruction_from_skill_message(_SCAFFOLD + _PASTED_NOTE) == _PASTED_NOTE
        assert extract_user_instruction_from_skill_message(_SCAFFOLD + _PASTED_FOOTER) == _PASTED_FOOTER


class TestDescribeSkillInvocation:
    def test_auto_loaded_skills_render_typed_text(self):
        assert describe_skill_invocation(_TWO_SKILL_TURN) == "not remember anything i tell you"

    def test_bare_auto_load_renders_the_skill_name(self):
        assert describe_skill_invocation(_SCAFFOLD) == "/example"

    def test_registered_boundary_keeps_pasted_lookalikes(self):
        turns.register_auto_load_scaffold(_SCAFFOLD)
        assert describe_skill_invocation(_SCAFFOLD + _PASTED_NOTE) == _PASTED_NOTE
        assert describe_skill_invocation(_SCAFFOLD + _PASTED_FOOTER) == _PASTED_FOOTER

    def test_longest_registered_scaffold_wins(self):
        second = _SCAFFOLD + (
            '[IMPORTANT: The "another" skill is auto-loaded. Follow its instructions for this session.]\n'
            f"Second body.\n[Skill directory: /skills/another]\nResolve paths, {_SKILL_DIR_NOTE_END}\n\n"
        )
        turns.register_auto_load_scaffold(_SCAFFOLD)
        turns.register_auto_load_scaffold(second)
        assert describe_skill_invocation(second + "  the   real request ") == "the real request"


class TestScaffoldRegistry:
    def test_empty_scaffold_is_not_registered(self):
        turns.register_auto_load_scaffold("")
        assert list(turns._AUTO_LOAD_SCAFFOLDS) == []

    def test_oldest_scaffold_evicted_past_the_cap(self):
        scaffolds = [f"{_SCAFFOLD}#{i}\n" for i in range(turns._AUTO_LOAD_SCAFFOLDS_MAX + 1)]
        for scaffold in scaffolds:
            turns.register_auto_load_scaffold(scaffold)
        assert list(turns._AUTO_LOAD_SCAFFOLDS) == scaffolds[1:]
