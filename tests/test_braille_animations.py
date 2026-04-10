"""Tests for hermes_cli.braille_animations — ported unicode-animations frame data."""

import pytest

from hermes_cli.braille_animations import (
    ANIMATIONS,
    grid_to_braille,
    make_grid,
    get_animation,
    get_animation_names,
    get_frame,
)


class TestAnimationData:
    def test_all_18_animations_present(self):
        assert len(ANIMATIONS) == 18

    def test_animation_names_sorted(self):
        names = get_animation_names()
        assert names == tuple(sorted(names))
        assert len(names) == 18

    def test_each_animation_has_required_keys(self):
        for name, anim in ANIMATIONS.items():
            assert "frames" in anim, f"{name} missing 'frames'"
            assert "interval_ms" in anim, f"{name} missing 'interval_ms'"
            assert isinstance(anim["frames"], tuple), f"{name} frames should be tuple"
            assert len(anim["frames"]) > 0, f"{name} has no frames"
            assert anim["interval_ms"] > 0, f"{name} has non-positive interval"

    def test_frames_are_strings(self):
        for name, anim in ANIMATIONS.items():
            for i, frame in enumerate(anim["frames"]):
                assert isinstance(frame, str), f"{name} frame {i} is not a string"

    def test_known_animation_frame_counts(self):
        assert len(ANIMATIONS["braille"]["frames"]) == 10
        assert len(ANIMATIONS["breathe"]["frames"]) == 17
        assert len(ANIMATIONS["columns"]["frames"]) == 26
        assert len(ANIMATIONS["orbit"]["frames"]) == 8
        assert len(ANIMATIONS["dna"]["frames"]) == 12

    def test_braille_is_classic_spinner(self):
        frames = ANIMATIONS["braille"]["frames"]
        assert frames[0] == "⠋"
        assert frames[-1] == "⠏"


class TestGetAnimation:
    def test_valid_name(self):
        anim = get_animation("breathe")
        assert anim["interval_ms"] == 100
        assert len(anim["frames"]) == 17

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            get_animation("nonexistent")


class TestGetFrame:
    def test_returns_first_frame_at_zero(self):
        frame = get_frame("braille", 0)
        assert frame == "⠋"

    def test_wraps_around(self):
        anim = ANIMATIONS["braille"]
        total_ms = anim["interval_ms"] * len(anim["frames"])
        frame_at_0 = get_frame("braille", 0)
        frame_at_wrap = get_frame("braille", total_ms)
        assert frame_at_0 == frame_at_wrap

    def test_advances_with_time(self):
        frame_0 = get_frame("braille", 0)
        frame_1 = get_frame("braille", 80)
        assert frame_0 != frame_1

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            get_frame("nonexistent", 0)


class TestGridToBraille:
    def test_empty_grid(self):
        assert grid_to_braille([]) == ""

    def test_single_cell_all_false(self):
        grid = make_grid(4, 2)
        result = grid_to_braille(grid)
        assert result == "⠀"  # U+2800 (braille blank)

    def test_single_cell_all_true(self):
        grid = make_grid(4, 2)
        for r in range(4):
            for c in range(2):
                grid[r][c] = True
        result = grid_to_braille(grid)
        assert result == "⣿"  # U+28FF (all dots)

    def test_top_left_dot(self):
        grid = make_grid(4, 2)
        grid[0][0] = True
        result = grid_to_braille(grid)
        assert result == "⠁"  # U+2801 (dot 1)


class TestMakeGrid:
    def test_dimensions(self):
        grid = make_grid(4, 6)
        assert len(grid) == 4
        assert all(len(row) == 6 for row in grid)

    def test_all_false(self):
        grid = make_grid(4, 2)
        assert all(not cell for row in grid for cell in row)
