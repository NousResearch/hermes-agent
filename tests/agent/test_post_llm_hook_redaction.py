"""The ``post_llm_call`` payload carries masked text, and the live list is untouched.

Context-engine plugins persist their own copy of the conversation from this payload
(hermes-lcm writes ``lcm.db`` from it), so it is a second write path that transcript
redaction cannot reach. Masking happens on shallow copies: the live message dicts that
the next API call sends must keep their exact bytes or the prompt cache breaks.
"""
from __future__ import annotations

import json
import logging
import types

import pytest

from agent import turn_finalizer

FAKE_GITHUB = "ghp_" + "Q7w6E5r4T3y2U1i0O9p8A7s6D5f4G3h2J1k0"


@pytest.fixture()
def hook_on(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_llm_call")


@pytest.fixture()
def hook_off(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)


def test_hook_history_is_masked_and_leaves_the_live_list_alone(hook_on):
    messages = [
        {"role": "user", "content": f"deploy with {FAKE_GITHUB} please", "timestamp": 1.0},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"using {FAKE_GITHUB}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "timestamp": 2.0,
        },
    ]
    history = turn_finalizer._hook_history(messages)

    assert FAKE_GITHUB not in history[0]["content"]
    assert history[0]["content"].startswith("deploy with ")
    assert FAKE_GITHUB not in history[1]["content"][0]["text"]
    assert history[1]["content"][1] == messages[1]["content"][1], "the image part was rewritten"
    # Copies are shallow and the live list keeps the bytes the wire sent.
    assert history[0] is not messages[0]
    assert history[0]["timestamp"] == messages[0]["timestamp"]
    assert messages[0]["content"] == f"deploy with {FAKE_GITHUB} please"
    assert messages[1]["content"][0]["text"] == f"using {FAKE_GITHUB}"


def test_the_api_content_sidecar_is_masked_too(hook_on):
    """The sidecar is the wire copy: injected context rides in it, not in ``content``.

    A reviewer probe against this branch found ``content`` masked while ``api_content``
    still carried the credential — the exact bytes sent to the model, memory injection and
    all. Both leaves are masked now, on both row types that carry the sidecar (user rows
    with injected context, promoted assistant rows).
    """
    injected = f"<memory>\nCI token: {FAKE_GITHUB}\n</memory>\n\nsummarize the deploy"
    messages = [
        {"role": "user", "content": "summarize the deploy", "api_content": injected, "timestamp": 1.0},
        {
            "role": "assistant", "content": "", "display_kind": "hidden",
            "api_content": f"the token is {FAKE_GITHUB}", "timestamp": 2.0,
        },
    ]
    history = turn_finalizer._hook_history(messages)

    assert "api_content" in history[0], "the sidecar was dropped instead of masked"
    assert FAKE_GITHUB not in history[0]["api_content"]
    assert history[0]["api_content"].startswith("<memory>")
    assert FAKE_GITHUB not in history[1]["api_content"]
    # The live dicts keep the bytes the wire sent, or the next API call misses the cache.
    assert messages[0]["api_content"] == injected
    assert FAKE_GITHUB in messages[1]["api_content"]


def test_tool_call_arguments_are_masked_and_the_replay_copy_is_not(hook_on):
    """Arguments carry credentials too — a shell command with an exported token.

    The transcript keeps tool-call arguments raw on purpose (#43083: the model reads its own
    masked command back and copies the placeholder into the next call). That objection is
    about the conversation replayed to the model; this payload is a copy handed to a plugin
    and never sent, so the same masking the other two leaves get applies here.
    """
    args = json.dumps({"command": f"export GITHUB_TOKEN={FAKE_GITHUB} && git push origin main"})
    messages = [
        {"role": "user", "content": "push it", "timestamp": 1.0},
        {
            "role": "assistant", "content": "", "timestamp": 2.0,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "terminal", "arguments": args}}],
        },
    ]
    history = turn_finalizer._hook_history(messages)

    call = history[1]["tool_calls"][0]
    assert FAKE_GITHUB not in call["function"]["arguments"]
    assert "git push origin main" in call["function"]["arguments"], "the command was over-masked"
    # Still parseable, and the sibling keys survive: a plugin that loads the arguments
    # must not choke on the mask (#137028 is that failure on the request path).
    decoded = json.loads(call["function"]["arguments"])
    assert decoded["command"].startswith("export GITHUB_TOKEN=")
    assert decoded["command"].endswith("git push origin main")
    # Identity fields survive so a plugin can still key a call by id.
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "terminal"
    # The live dict keeps the bytes the next API request replays.
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == args
    assert messages[1]["tool_calls"] is not call


def test_no_hook_means_no_masking_work(hook_off):
    messages = [{"role": "user", "content": f"deploy with {FAKE_GITHUB}"}]
    history = turn_finalizer._hook_history(messages)
    assert history[0] is messages[0], "nothing should be copied or scanned when no hook is registered"


def _stub_agent():
    return types.SimpleNamespace(
        session_id="s-1", model="m-1", _persist_disabled=False, platform="cli",
    )


def test_the_dispatch_hands_the_hook_masked_text(monkeypatch, hook_on):
    """End of the chain: what a plugin actually receives from a finished turn."""
    captured = {}

    def fake_invoke(name, logger, **kwargs):
        if name == "post_llm_call":
            captured.update(kwargs)
        return []

    monkeypatch.setattr(turn_finalizer, "_invoke_hook_safely", fake_invoke)
    messages = [{"role": "user", "content": f"my token is {FAKE_GITHUB}"}]

    turn_finalizer._apply_output_hooks(
        _stub_agent(), "done", logging.getLogger("test"), platform="cli",
        effective_task_id="t-1", turn_id="turn-1", original_user_message="hi",
        messages=messages,
    )

    assert FAKE_GITHUB not in captured["conversation_history"][0]["content"]
    assert messages[0]["content"] == f"my token is {FAKE_GITHUB}", "the live turn was mutated"
