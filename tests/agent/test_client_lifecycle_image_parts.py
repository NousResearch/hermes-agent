"""``ClientLifecycleMixin._api_kwargs_have_image_parts`` on payloads whose ``type`` is not a string.

Tool schemas and tool-call arguments carry arbitrary JSON, so a dict can hold ``"type": {...}`` or a
JSON-Schema union ``"type": ["string", "null"]``. Testing that value for set membership raised
``TypeError: unhashable type`` before the request was built, and the retry loop then switched the
session to the fallback model (#149107, #148586).
"""

import pytest

from hermes_agent.agent.client_lifecycle import ClientLifecycleMixin


@pytest.mark.parametrize("type_value", [{"slug": "full-time"}, ["string", "null"]])
def test_a_non_string_type_neither_raises_nor_stops_the_scan(type_value):
    plain = {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": {"type": type_value}}]}]}
    nested = {"input": [{"type": type_value, "nested": {"type": "input_image", "input_image": "https://x.test/a.png"}}]}

    assert ClientLifecycleMixin._api_kwargs_have_image_parts(plain) is False
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(nested) is True
