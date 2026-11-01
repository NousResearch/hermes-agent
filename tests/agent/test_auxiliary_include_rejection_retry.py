"""Encrypted-reasoning ``include`` rejection retry in the aux Responses adapter (#134796).

A self-hosted OpenAI-compatible /responses server that publishes a low/medium/high effort
ladder but never adopted encrypted-reasoning replay answers

    HTTP 400: include is not supported (reasoning.encrypted_content)

to the aux adapter's housekeeping calls (compression, background review), which used to
stall the session behind the hard 400. The recovery mirrors the temperature rungs: one
retry without ``include``, and the route remembered so the next call omits the field up
front. ``reasoning`` itself stays on the wire — the server accepts the effort field, it
only refuses the include.
"""

from types import SimpleNamespace

import pytest

import agent.auxiliary_client as auxiliary_client
from agent.auxiliary_client import _CodexCompletionsAdapter

_SELFHOSTED_400 = (
    "Error code: 400 - {'error': {'message': 'include is not supported "
    "(reasoning.encrypted_content)', 'type': 'invalid_request_error', 'param': '', 'code': None}}"
)


@pytest.fixture(autouse=True)
def _clean_route_memory():
    memory = getattr(auxiliary_client, "_ENCRYPTED_INCLUDE_REJECTED_ROUTES", None)
    if memory is not None:
        memory.clear()
    yield
    if memory is not None:
        memory.clear()


def _final():
    return SimpleNamespace(
        status="completed",
        output=[SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text="ok")],
        )],
        usage=SimpleNamespace(input_tokens=3, output_tokens=1, total_tokens=4),
    )


class _RejectingInclude:
    """responses.create of a server that 400s any request carrying ``include``."""

    def __init__(self, error_text=_SELFHOSTED_400):
        self.calls = []
        self.error_text = error_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "include" in kwargs:
            raise RuntimeError(self.error_text)
        return _final()


def _adapter(responses_api, base_url="https://tensorfold.example/v1", model="gpt-5.5"):
    return _CodexCompletionsAdapter(
        SimpleNamespace(base_url=base_url, responses=responses_api), model,
    )


def _create_kwargs():
    return dict(
        messages=[{"role": "user", "content": "Summarize the task."}],
        extra_body={"reasoning": {"effort": "low"}},
    )


def test_include_rejection_retries_once_without_include():
    responses_api = _RejectingInclude()
    result = _adapter(responses_api).create(**_create_kwargs())

    assert result.choices[0].message.content == "ok"
    assert len(responses_api.calls) == 2
    first, retry = responses_api.calls
    assert first["include"] == ["reasoning.encrypted_content"]  # still attempted on the first call
    assert "include" not in retry
    assert retry["reasoning"] == first["reasoning"]  # the effort field itself is accepted, keep it
    assert retry["model"] == first["model"]


def test_rejected_route_omits_include_up_front_on_the_next_call():
    responses_api = _RejectingInclude()
    _adapter(responses_api).create(**_create_kwargs())

    assert auxiliary_client._ENCRYPTED_INCLUDE_REJECTED_ROUTES == {("tensorfold.example", "gpt-5.5")}

    # A fresh adapter on the same route never sends the include again.
    accepting = _RejectingInclude()
    result = _adapter(accepting).create(**_create_kwargs())
    assert result.choices[0].message.content == "ok"
    assert len(accepting.calls) == 1
    assert "include" not in accepting.calls[0]


def test_unrelated_400_is_not_retried():
    responses_api = _RejectingInclude(
        "Error code: 400 - {'error': {'message': \"Invalid value: 'tool'. Supported values are: "
        "'assistant'\", 'type': 'invalid_request_error'}}"
    )

    with pytest.raises(RuntimeError, match="Invalid value"):
        _adapter(responses_api).create(**_create_kwargs())
    assert len(responses_api.calls) == 1
    assert auxiliary_client._ENCRYPTED_INCLUDE_REJECTED_ROUTES == set()


def test_variant_naming_the_value_instead_of_the_field_also_recovers():
    """A server that says ``reasoning.encrypted_content ... not supported`` without the word
    ``include`` rejects the same wire field; the retry must strip it too."""
    responses_api = _RejectingInclude(
        "Error code: 400 - {'error': {'message': 'reasoning.encrypted_content is not supported "
        "by this endpoint', 'type': 'invalid_request_error'}}"
    )
    result = _adapter(responses_api).create(**_create_kwargs())

    assert result.choices[0].message.content == "ok"
    assert "include" not in responses_api.calls[1]


def test_invalid_encrypted_content_replay_blob_is_not_an_include_rejection():
    """The main transport's ``invalid_encrypted_content`` class (server rejects the blob, not
    the field) must not collide with this rung: stripping the include would hide the real
    ask, which is dropping stale replay state."""
    class _Status400(RuntimeError):
        status_code = 400

    assert not auxiliary_client._is_encrypted_include_rejection(
        _Status400("Error code: 400 - invalid_encrypted_content")
    )
