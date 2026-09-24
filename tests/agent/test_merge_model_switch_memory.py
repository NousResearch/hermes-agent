"""A model-switch row merged with the next user turn must keep that turn's memory sidecar.

``api_content`` is where the prefetch block lives. Dropping it because the
visible ``content`` was rewritten sends the first message after a switch
with no ``<memory-context>``. See #121836.
"""

from agent.agent_runtime_helpers import _merge_consecutive_users


def test_model_switch_merge_keeps_the_new_turns_memory_sidecar():
    switch = {
        "role": "user",
        "display_kind": "model_switch",
        "content": "[System: The active model for this chat has changed to X via provider Y.]",
    }
    msg = {
        "role": "user",
        "content": "hello",
        "api_content": "hello\n\n<memory-context>MEMORY</memory-context>",
    }

    out, repairs = _merge_consecutive_users(
        [{"role": "assistant", "content": "a"}, switch, msg]
    )

    assert repairs == 1
    merged = out[-1]
    assert merged["content"].startswith("[System:")
    assert "hello" in merged["content"]
    assert "<memory-context>MEMORY</memory-context>" in merged["api_content"]
    assert merged["api_content"].startswith("[System:")


def test_plain_user_merge_still_drops_a_sidecar_that_matches_content():
    first = {"role": "user", "content": "one", "api_content": "one"}
    second = {"role": "user", "content": "two", "api_content": "two"}

    out, repairs = _merge_consecutive_users([first, second])

    assert repairs == 1
    assert out[0]["content"] == "one\n\ntwo"
    assert "api_content" not in out[0]
