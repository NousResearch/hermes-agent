"""Public names for the pet sprite pipeline's frame and prompt primitives.

A plugin that generates its own sprite sheets (a different frame grid, extra
rows) wants to reuse the same row slicing, cell fitting and chroma-key prompt
text the built-in pet generator uses, instead of copying them. Each public
name is the SAME object as the private spelling, so the generator is unchanged.
"""

import pytest
from PIL import Image

from agent.pet.generate import atlas, prompts


@pytest.mark.parametrize(
    ("module", "public", "private"),
    [
        (atlas, "erase_long_axis_lines", "_erase_long_axis_lines"),
        (atlas, "frame_x_ranges", "_frame_x_ranges"),
        (atlas, "sever_expected_gutters", "_sever_expected_gutters"),
        (atlas, "fit_to_cell", "_fit_to_cell"),
        (atlas, "clear_transparent_rgb", "_clear_transparent_rgb"),
        (prompts, "spacing_spec", "_spacing_spec"),
        (prompts, "ASSUMED_STRIP_WIDTH", "_ASSUMED_STRIP_WIDTH"),
        (prompts, "BACKGROUND", "_BACKGROUND"),
    ],
)
def test_public_name_is_the_private_spelling(module, public, private):
    assert getattr(module, public) is getattr(module, private)


def test_fit_to_cell_returns_a_cell_sized_sprite():
    sprite = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    assert atlas.fit_to_cell(sprite).size == (atlas.CELL_WIDTH, atlas.CELL_HEIGHT)


def test_clear_transparent_rgb_zeroes_colour_under_zero_alpha():
    image = Image.new("RGBA", (1, 1), (200, 100, 50, 0))
    assert atlas.clear_transparent_rgb(image).getpixel((0, 0)) == (0, 0, 0, 0)


def test_frame_x_ranges_finds_one_span_per_pose():
    strip = Image.new("RGBA", (300, 50), (0, 0, 0, 0))
    for left in (10, 110, 210):
        strip.paste(Image.new("RGBA", (60, 30), (0, 255, 0, 255)), (left, 10))
    assert atlas.frame_x_ranges(strip, 3) == [(10, 70), (110, 170), (210, 270)]


def test_spacing_spec_splits_each_slot_into_pose_and_gap():
    pose, gap = prompts.spacing_spec(4)
    assert pose > gap > 0
    assert pose < prompts.ASSUMED_STRIP_WIDTH // 4
