"""Public names for two turn-path decisions plugins need to agree with.

``sanitize_surrogates`` is the scrub every outbound message gets; a plugin that
writes user text into a session outside the normal turn must apply the same one
or the provider rejects the request. ``should_stream`` is how a turn decides
between the streaming and buffered provider call; a plugin that reports on the
dispatch has to label it the way the turn made it. Each public name is the SAME
object as the private spelling, so the turn path is unchanged.
"""

import pytest

from agent import message_sanitization, turn_api_call


@pytest.mark.parametrize(
    ("module", "public", "private"),
    [
        (message_sanitization, "sanitize_surrogates", "_sanitize_surrogates"),
        (turn_api_call, "should_stream", "_should_stream"),
    ],
)
def test_public_name_is_the_private_helper(module, public, private):
    assert getattr(module, public) is getattr(module, private)


@pytest.mark.parametrize(
    ("module", "public"),
    [(message_sanitization, "sanitize_surrogates"), (turn_api_call, "should_stream")],
)
def test_public_name_is_exported(module, public):
    assert public in module.__all__


def test_turn_api_call_exports_resolve():
    for name in turn_api_call.__all__:
        assert hasattr(turn_api_call, name), name


def test_sanitize_surrogates_replaces_a_lone_surrogate():
    assert message_sanitization.sanitize_surrogates("a\ud800b") == "a�b"
    assert message_sanitization.sanitize_surrogates("plain") == "plain"


def test_should_stream_honours_the_disable_flag():
    class _Agent:
        _disable_streaming = True

    assert turn_api_call.should_stream(_Agent()) is False
