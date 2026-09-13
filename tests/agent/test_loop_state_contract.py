from dataclasses import fields

from agent.conversation_loop import _LoopState


def test_loop_state_supplies_finalize_blocked_contract():
    field_names = {field.name for field in fields(_LoopState)}
    assert "blocked" in field_names
    assert _LoopState.__dataclass_fields__["blocked"].default is False
