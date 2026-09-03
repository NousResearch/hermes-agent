"""Focused content-visibility contracts for transcript repair."""

import pytest

from agent.transcript_repair import _has_visible_repair_content


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("answer", True, id="string"),
        pytest.param("  ", False, id="whitespace-string"),
        pytest.param([{"type": "text", "text": "answer"}], True, id="text-part"),
        pytest.param([{"type": "text", "text": "  "}], False, id="blank-text"),
        pytest.param(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                }
            ],
            True,
            id="image-url",
        ),
        pytest.param(
            [{"type": "image_url", "image_url": {"url": ""}}],
            False,
            id="empty-image-url",
        ),
        pytest.param(
            [{"type": "input_audio", "input_audio": {"data": "audio-data"}}],
            True,
            id="input-audio",
        ),
        pytest.param(
            [{"type": "input_audio", "input_audio": {"data": ""}}],
            False,
            id="empty-input-audio",
        ),
        pytest.param([], False, id="empty-list"),
        pytest.param({}, False, id="empty-dict"),
        pytest.param([{}], False, id="empty-part"),
        pytest.param([{"type": "unknown"}], False, id="unknown-placeholder"),
    ],
)
def test_has_visible_repair_content(content, expected):
    assert _has_visible_repair_content(content) is expected
