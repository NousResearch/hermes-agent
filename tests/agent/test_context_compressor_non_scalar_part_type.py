"""Regression tests for #107628: a content part whose ``type`` is not a scalar must not crash compression.

Tool results carry arbitrary JSON, so a part can be shaped like a JSON-Schema union
(``{"type": ["string", "null"]}``) or a nested object. ``x in frozenset`` hashes ``x``, so the image-type
membership tests raised ``TypeError: unhashable type`` for those values. The raise escaped compression's
own error handling, which left an over-limit session permanently uncompressed.
"""

import pytest

from agent.context_compressor import (
    _content_has_images,
    _is_image_part,
    _replace_image_parts,
    _summary_part_text,
)

NON_SCALAR_TYPES = [["string", "null"], [], {"type": "string"}, {"a": 1}]
SCALAR_NON_IMAGE_TYPES = [None, 0, 1, True, ""]


@pytest.mark.parametrize("ptype", NON_SCALAR_TYPES + SCALAR_NON_IMAGE_TYPES)
def test_non_image_part_type_is_reported_as_no_image(ptype):
    assert _is_image_part({"type": ptype}) is False


@pytest.mark.parametrize("ptype", NON_SCALAR_TYPES + SCALAR_NON_IMAGE_TYPES)
def test_non_image_part_type_does_not_break_image_detection(ptype):
    assert _content_has_images([{"type": ptype}]) is False
    assert _content_has_images([{"type": "text", "text": "hi"}, {"type": ptype}]) is False


@pytest.mark.parametrize("ptype", NON_SCALAR_TYPES)
def test_strip_leaves_non_scalar_parts_alone(ptype):
    part = {"type": ptype, "text": "keep me"}
    assert _replace_image_parts([part], "[stripped]") is None


@pytest.mark.parametrize("ptype", NON_SCALAR_TYPES)
def test_summary_text_tolerates_non_scalar_type(ptype):
    assert isinstance(_summary_part_text({"type": ptype}), str)


@pytest.mark.parametrize("ptype", ["image_url", "input_image", "image"])
def test_known_image_types_are_still_detected(ptype):
    assert _is_image_part({"type": ptype}) is True
    assert _content_has_images([{"type": "text", "text": "hi"}, {"type": ptype}]) is True
    assert _replace_image_parts([{"type": ptype}], "[stripped]") == [{"type": "text", "text": "[stripped]"}]


def test_text_parts_still_summarized():
    assert _summary_part_text({"type": "text", "text": "hello"}) == "hello"
    assert _summary_part_text("raw string") == "raw string"


def test_non_image_part_marker_preserved():
    assert _summary_part_text({"type": "tool_result"}) == "[tool_result]"
    assert _summary_part_text({"text": "no type"}) == "[attachment]"
