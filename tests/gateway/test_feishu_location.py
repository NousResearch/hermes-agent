"""Feishu ``location`` messages must reach the agent as readable text.

Feishu returns a location message's ``body.content`` as
``{"name": "<POI>", "longitude": "<lng>", "latitude": "<lat>"}``.  Before this
fix ``normalize_feishu_message`` had no ``location`` branch, so the message fell
through to the empty-text fallback and the whole message was dropped as
"no content" — the user saw nothing happen.

All coordinates below are synthetic; never put a real user location in a test.
"""

import json

import pytest

from plugins.platforms.feishu.adapter import normalize_feishu_message

NAME = "合成测试地点"
LNG = "120.100000"
LAT = "30.200000"


def _marker_url(lng: str, lat: str, name: str) -> str:
    from urllib.parse import quote

    return f"https://uri.amap.com/marker?position={lng},{lat}&name={quote(name, safe='')}"


def test_full_payload_renders_name_coordinates_and_marker_link():
    normalized = normalize_feishu_message(
        message_type="location",
        raw_content=json.dumps({"name": NAME, "longitude": LNG, "latitude": LAT}),
    )

    text = normalized.text_content
    assert normalized.raw_type == "location"
    assert NAME in text
    # Coordinates keep the payload's original string precision (a float round
    # trip would turn "120.100000" into "120.1").
    assert f"{LNG},{LAT}" in text
    assert _marker_url(LNG, LAT, NAME) in text


def test_coordinates_without_name_still_deliver_a_position():
    normalized = normalize_feishu_message(
        message_type="location",
        raw_content=json.dumps({"longitude": LNG, "latitude": LAT}),
    )

    text = normalized.text_content
    assert text, "an empty text_content gets the whole message dropped"
    assert f"{LNG},{LAT}" in text
    assert f"https://uri.amap.com/marker?position={LNG},{LAT}" in text


@pytest.mark.parametrize("bad", ["", "abc", "nan", "inf", "-inf", None])
def test_name_with_invalid_coordinates_degrades_to_name_only(bad):
    normalized = normalize_feishu_message(
        message_type="location",
        raw_content=json.dumps({"name": NAME, "longitude": bad, "latitude": LAT}),
    )

    text = normalized.text_content
    assert NAME in text
    assert "uri.amap.com" not in text


def test_empty_payload_returns_placeholder_text():
    normalized = normalize_feishu_message(message_type="location", raw_content="{}")

    # Anything non-empty keeps the message alive; an empty text_content is what
    # gets the whole event dropped before the agent ever sees it.
    assert normalized.text_content
    assert normalized.text_content.startswith("[")
