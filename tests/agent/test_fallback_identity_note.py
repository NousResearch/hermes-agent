"""A fallback must not adopt the identity a queued ``/model`` note gave the failed primary.

``/model X`` prepends "[Note: model was just switched ... to X ...]" to the next user
message. When X then fails (401) and ``try_activate_fallback`` swaps in Y, the
rewritten ``Model:`` line alone loses to that user-turn note, so Y answers "I am X".
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import (
    append_fallback_identity_note,
    rewrite_prompt_model_identity,
)

_PROMPT = "You are a helpful assistant.\nModel: GLM-5.3-Flash\nProvider: glm-headroom"


def _agent():
    return SimpleNamespace(_cached_system_prompt=_PROMPT, ephemeral_system_prompt=None)


def _fall_back(agent):
    rewrite_prompt_model_identity(agent, "nemotron-ultra", "nvidia")
    append_fallback_identity_note(agent, "GLM-5.3-Flash", "glm-headroom", "nemotron-ultra", "nvidia")


def test_fallback_prompt_names_the_real_model_and_the_failed_one():
    agent = _agent()
    _fall_back(agent)
    sp = agent._cached_system_prompt
    assert "You are actually nemotron-ultra via nvidia" in sp
    assert "GLM-5.3-Flash (via glm-headroom) failed" in sp
    assert "[Note: model was just switched" in sp  # tells it the user-turn note is not about it


def test_restoring_the_primary_is_byte_identical():
    agent = _agent()
    _fall_back(agent)
    rewrite_prompt_model_identity(agent, "GLM-5.3-Flash", "glm-headroom")
    assert agent._cached_system_prompt == _PROMPT  # prefix cache must still match the stored row


def test_a_second_fallback_replaces_the_note_instead_of_stacking():
    agent = _agent()
    _fall_back(agent)
    rewrite_prompt_model_identity(agent, "gemma-4", "local")
    append_fallback_identity_note(agent, "nemotron-ultra", "nvidia", "gemma-4", "local")
    sp = agent._cached_system_prompt
    assert sp.count("[Fallback identity]") == 1
    assert "You are actually gemma-4 via local" in sp
