"""Structured-output mapping on the auxiliary Codex Responses route."""

from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter


_SCHEMA = {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}
_FORMAT = {"type": "json_schema", "json_schema": {"name": "reply", "schema": _SCHEMA, "strict": True}}


@pytest.mark.parametrize("surface", ["top_level", "extra_body"])
def test_json_schema_reaches_codex_responses_text_format(surface):
    adapter = _CodexCompletionsAdapter(
        SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex/"), "gpt-5.3-codex"
    )
    kwargs = {"model": "gpt-5.3-codex", "messages": [{"role": "user", "content": "say pong"}],
              "text": {"verbosity": "low"}}
    if surface == "extra_body":
        kwargs["extra_body"] = {"response_format": _FORMAT}
    else:
        kwargs["response_format"] = _FORMAT

    built, _, _ = adapter._build_responses_kwargs(kwargs)

    assert built["text"] == {"verbosity": "low", "format": {
        "type": "json_schema", "name": "reply", "schema": _SCHEMA, "strict": True,
    }}
    assert "response_format" not in built


def test_extra_body_format_takes_precedence_without_mutating_caller():
    adapter = _CodexCompletionsAdapter(
        SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex/"), "gpt-5.3-codex"
    )
    extra = {"response_format": {"type": "json_schema", "json_schema": {"schema": _SCHEMA}}}
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "response_format": _FORMAT,
              "extra_body": extra}

    built, _, _ = adapter._build_responses_kwargs(kwargs)

    assert built["text"]["format"] == {
        "type": "json_schema", "name": "response", "schema": _SCHEMA, "strict": True,
    }
    assert extra == {"response_format": {"type": "json_schema", "json_schema": {"schema": _SCHEMA}}}
