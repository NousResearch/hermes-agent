"""Interim visible text must survive content-parts lists.

After context compaction (and with multimodal providers) an assistant
message's ``content`` can be a list of parts instead of a string.
``_interim_assistant_visible_text`` passed the list straight into the
think-block regexes, and every later API call in the conversation died
with ``TypeError: expected string or bytes-like object, got 'list'``.
"""

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    return object.__new__(AIAgent)


def test_interim_visible_text_handles_content_parts_list():
    agent = _bare_agent()
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "<think>hidden</think>Risposta visibile."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            {"type": "text", "text": " Continua."},
        ],
    }

    assert agent._interim_assistant_visible_text(msg) == "Risposta visibile. Continua."


def test_interim_visible_text_ignores_non_text_only_parts_list():
    agent = _bare_agent()
    msg = {
        "role": "assistant",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}}],
    }

    assert agent._interim_assistant_visible_text(msg) == ""


def test_interim_visible_text_still_handles_plain_string():
    agent = _bare_agent()
    msg = {"role": "assistant", "content": "<think>x</think>ciao"}

    assert agent._interim_assistant_visible_text(msg) == "ciao"
