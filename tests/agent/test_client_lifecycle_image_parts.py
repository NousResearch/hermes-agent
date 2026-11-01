"""Regression tests for ``ClientLifecycleMixin._api_kwargs_have_image_parts``.

Pins the image-part detection contract and guards against the unhashable-type
crash: a message part whose ``type`` field holds a ``dict``/``list``/``set``
used to raise ``TypeError: unhashable type`` because the value was tested for
membership in the image-type set without an ``isinstance(str)`` check. The
detector now only compares string types and lets non-string parts fall through
to the recursive traversal.
"""

from __future__ import annotations

import pytest

from agent.client_lifecycle import ClientLifecycleMixin


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url"},
        {"type": "input_image"},
    ],
)
def test_detects_string_image_types(part):
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [part]}) is True


@pytest.mark.parametrize(
    "part",
    [
        {"type": {"foo": "bar"}},
        {"type": ["image_url", "input_image"]},
        {"type": {"image_url", "input_image"}},
    ],
)
def test_unhashable_type_field_does_not_raise(part):
    # Regression: dict/list/set type fields previously raised TypeError.
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [part]}) is False


def test_detects_nested_image_part():
    api_kwargs = {"messages": [{"content": [{"type": "text"}, {"type": "image_url"}]}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True


def test_input_field_is_scanned():
    api_kwargs = {"input": [{"type": "input_image"}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True


def test_no_image_part_returns_false():
    api_kwargs = {"messages": [{"type": "text", "content": "hello"}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False


@pytest.mark.parametrize("api_kwargs", [None, [], "not-a-dict"])
def test_non_dict_api_kwargs_returns_false(api_kwargs):
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False


def test_non_list_messages_field_returns_false():
    # A dict-valued "messages" field is not scanned (only list fields are).
    api_kwargs = {"messages": {"type": "image_url"}}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False