"""`_moa_prepared_request` must never reach a client that cannot consume it.

The key is a private handshake between the conversation loop and
``MoAChatCompletions.create``. Credential rotation, provider fallback and
dead-connection cleanup rebuild ``agent.client`` from ``_client_kwargs``
between attempts — and a prepared request survives that boundary via
``pending_moa_prepared_request`` — while ``agent.provider`` stays ``"moa"``.
Handing the key to the rebuilt native client raises a non-retryable
``TypeError`` that kills every remaining turn on the session.
"""

import types

from agent.conversation_loop import _moa_client_consumes_prepared_request
from agent.moa_loop import MoAChatCompletions


def _client_with(completions):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )


class _NativeCompletions:
    """openai.resources.chat.Completions — an explicit keyword signature."""

    def create(self, *, model=None, messages=None, tools=None, stream=None):
        return "native ok"


def test_native_client_does_not_consume_the_prepared_request():
    assert _moa_client_consumes_prepared_request(_client_with(_NativeCompletions())) is False


def test_real_moa_facade_consumes_the_prepared_request():
    facade = MoAChatCompletions.__new__(MoAChatCompletions)
    assert _moa_client_consumes_prepared_request(_client_with(facade)) is True


def test_client_without_a_chat_attribute_is_not_a_facade():
    assert _moa_client_consumes_prepared_request(object()) is False
    assert _moa_client_consumes_prepared_request(None) is False


def test_native_client_rejects_the_key_it_must_not_receive():
    """Why the guard exists: the kwarg is fatal to a native client."""
    native = _NativeCompletions()
    try:
        native.create(
            model="2 model",
            messages=[{"role": "user", "content": "hi"}],
            _moa_prepared_request={"messages": [], "layers": 3},
        )
    except TypeError as exc:
        assert "_moa_prepared_request" in str(exc)
    else:  # pragma: no cover - would mean the premise no longer holds
        raise AssertionError("native client accepted the private MoA kwarg")

    # Without the key it is an ordinary call the rebuilt client can serve.
    assert native.create(
        model="2 model", messages=[{"role": "user", "content": "hi"}]
    ) == "native ok"
