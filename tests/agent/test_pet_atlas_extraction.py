"""Sprite-strip extraction: context-aware line erase, vertical box merge, lenient rows."""

import pytest
from PIL import Image, ImageDraw

from agent.pet.generate import atlas


def _rgba(w, h):
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def test_vertically_stacked_parts_merge():
    head, body = (10, 0, 50, 30), (8, 35, 52, 90)
    assert atlas._merge_related_boxes([head, body]) == [(8, 0, 52, 90)]


def test_separate_poses_stay_separate():
    """Positive control: two poses far apart on one row are never bridged."""
    a, b = (0, 0, 40, 90), (200, 0, 240, 90)
    assert sorted(atlas._merge_related_boxes([a, b])) == [a, b]


def test_floor_is_erased_between_poses_but_kept_under_a_pose():
    img = _rgba(60, 20)
    draw = ImageDraw.Draw(img)
    draw.rectangle((5, 2, 15, 15), fill=(255, 0, 0, 255))  # the pose
    draw.rectangle((0, 16, 59, 17), fill=(0, 255, 0, 255))  # a drawn floor
    out = atlas._erase_long_axis_lines(img)
    assert out.getpixel((30, 16))[3] == 0  # floor across background: erased
    assert out.getpixel((10, 16))[3] == 255  # stub under the pose's feet: kept
    assert out.getpixel((10, 10))[3] == 255  # the pose itself: untouched


def _frame(width):
    img = _rgba(320, 40)
    ImageDraw.Draw(img).rectangle((5, 5, 5 + width - 1, 24), fill=(255, 255, 255, 255))
    return img


def test_lenient_validation_accepts_a_suspect_row():
    frames = [_frame(20), _frame(20), _frame(300)]
    with pytest.raises(ValueError, match="multi-pose width outlier"):
        atlas._validate_extracted_frames(frames, 3)
    atlas._validate_extracted_frames(frames, 3, strict=False)
