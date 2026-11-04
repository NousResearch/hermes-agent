"""``_api_kwargs_have_image_parts`` must not hash arbitrary ``type`` values (#148586).

The recursive scan looks for native image parts in outbound ``messages`` / ``input``. Structured
tool results legitimately carry JSON Schema unions such as ``{"type": ["string", "null"]}``; a set
membership test raises ``TypeError: unhashable type`` on those, before the provider request opens.
"""

import pytest

from agent.client_lifecycle import ClientLifecycleMixin


def _tool_result(payload: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "content": {"schema": payload}}]},
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"type": ["string", "null"]}, id="list"),
        pytest.param({"type": []}, id="empty-list"),
        pytest.param({"type": {"nested": 1}}, id="dict"),
        pytest.param({"type": None}, id="null"),
        pytest.param({"type": 3}, id="number"),
        pytest.param({"type": "string"}, id="text"),
    ],
)
def test_non_hashable_schema_type_is_not_an_image(payload):
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(_tool_result(payload)) is False


@pytest.mark.parametrize("field", ["messages", "input"])
@pytest.mark.parametrize("image_type", ["image_url", "input_image"])
def test_image_parts_are_still_found(field, image_type):
    payload = {
        field: [
            {"role": "user", "content": [
                {"type": "tool_result", "content": {"schema": {"type": ["string", "null"]}}},
                {"type": image_type},
            ]},
        ],
    }

    assert ClientLifecycleMixin._api_kwargs_have_image_parts(payload) is True
