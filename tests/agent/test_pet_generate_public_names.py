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


def _four_pose_strip():
    strip = Image.new("RGBA", (800, 200), (0, 0, 0, 0))
    for left in (25, 225, 425, 625):
        strip.paste(Image.new("RGBA", (150, 150), (255, 0, 0, 255)), (left, 25))
    return strip


def test_overriding_public_fit_to_cell_reaches_the_pipeline(monkeypatch):
    stub = Image.new("RGBA", (atlas.CELL_WIDTH, atlas.CELL_HEIGHT), (1, 2, 3, 255))
    monkeypatch.setattr(atlas, "fit_to_cell", lambda image: stub)
    frames = atlas.extract_strip_frames(_four_pose_strip(), 4)
    assert all(frame is stub for frame in frames)
    assert atlas.single_frame(Image.new("RGBA", (40, 40), (255, 0, 0, 255))) is stub


def test_overriding_public_clear_transparent_rgb_reaches_compose_atlas(monkeypatch):
    marker = Image.new("RGBA", (1, 1))
    monkeypatch.setattr(atlas, "clear_transparent_rgb", lambda image: marker)
    assert atlas.compose_atlas({}) is marker


def test_overriding_public_assumed_strip_width_reaches_spacing_and_prompt(monkeypatch):
    base = prompts.spacing_spec(4)
    monkeypatch.setattr(prompts, "ASSUMED_STRIP_WIDTH", 3072)
    assert prompts.spacing_spec(4) == (base[0] * 2, base[1] * 2)
    assert "3072px" in prompts.build_row_prompt("idle", 4, "a fox")


def test_overriding_public_spacing_spec_and_background_reach_the_prompts(monkeypatch):
    monkeypatch.setattr(prompts, "spacing_spec", lambda frame_count: (777, 333))
    monkeypatch.setattr(prompts, "BACKGROUND", "PLUGIN-BACKGROUND")
    row = prompts.build_row_prompt("idle", 4, "a fox")
    assert "777px" in row and "PLUGIN-BACKGROUND" in row
    assert "PLUGIN-BACKGROUND" in prompts.build_base_prompt("a fox")
