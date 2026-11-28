"""A held turn streams nothing, so interim commentary (model text beside a tool call) reaches the
user only through ``transform_interim_message``: rewritten, unchanged when the hook returns None, or
dropped when it returns "", when no plugin handles the hook, or when the callback fails. Unheld turns
never fire it."""

import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.redactor_plugin import COMMENTARY, REWRITE, raw_leaks, redactor_home, redactor_turn


def _run(*, hold, interim, commentary=COMMENTARY):
    from hermes_cli import plugins
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(redactor_turn(commentary=commentary)) as srv:
        redactor_home(home, srv.base_url, hold=hold, interim=interim)
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"])
            deltas: list = []
            notes: list = []
            agent.stream_delta_callback = deltas.append
            agent.interim_assistant_callback = lambda text, **kw: notes.append((text, kw.get("already_streamed")))
            result = agent.run_conversation("show me the config")
        finally:
            plugins._reset_plugin_managers_for_tests()
    return result, [d for d in deltas if d], notes


def test_held_commentary_is_shown_after_the_interim_transform():
    result, deltas, notes = _run(hold=True, interim=True)
    assert result["final_response"] == REWRITE
    assert deltas == []
    assert notes == [("Checking where [REDACTED] is stored first.", False)]


def test_held_commentary_the_hook_leaves_alone_is_shown_unchanged():
    _, deltas, notes = _run(hold=True, interim=True, commentary="Checking the config first.")
    assert deltas == []
    assert notes == [("Checking the config first.", False)]


def test_held_commentary_is_dropped_when_the_interim_transform_fails():
    result, deltas, notes = _run(hold=True, interim=True, commentary="Checking SECRET-0 first.")
    assert result["final_response"] == REWRITE
    assert deltas == [] and notes == []


def test_held_commentary_the_hook_empties_is_dropped():
    result, deltas, notes = _run(hold=True, interim=True, commentary="Checking SECRET-9 first.")
    assert result["final_response"] == REWRITE
    assert deltas == [] and notes == []


def test_held_commentary_without_an_interim_transform_is_still_dropped():
    _, deltas, notes = _run(hold=True, interim=False)
    assert deltas == [] and notes == []


@pytest.mark.parametrize("interim", [True, False])
def test_unheld_turns_never_run_the_interim_transform(interim):
    _, deltas, notes = _run(hold=False, interim=interim)
    assert [text for text, _ in notes] == [COMMENTARY]
    assert raw_leaks(deltas + [text for text, _ in notes])
