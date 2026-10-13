"""The host passes the same-prefix request only to engines whose compress() accepts it."""

from agent.conversation_compression import _supported_compression_kwargs


def _old(messages, current_tokens=None, focus_topic=None, force=False):
    return messages


def _new(messages, current_tokens=None, focus_topic=None, force=False, prefix_request=None):
    return messages


def test_an_engine_without_the_parameter_does_not_get_it():
    kwargs = _supported_compression_kwargs(_old, current_tokens=1, focus_topic=None, force=True, memory_context="",
                                           prefix_request=object())
    assert "prefix_request" not in kwargs


def test_an_engine_with_the_parameter_gets_it():
    seam = object()
    kwargs = _supported_compression_kwargs(_new, current_tokens=1, focus_topic=None, force=True, memory_context="",
                                           prefix_request=seam)
    assert kwargs["prefix_request"] is seam


def test_no_seam_means_no_keyword():
    kwargs = _supported_compression_kwargs(_new, current_tokens=1, focus_topic=None, force=True, memory_context="")
    assert "prefix_request" not in kwargs
