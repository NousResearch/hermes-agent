"""Display-width helpers for the session list tables (#44199).

CJK characters occupy two terminal columns, so ``str`` slices and
format-spec padding (which count characters) misalign every column to
the right of CJK content. ``_clip_to_width`` / ``_pad_to_width`` count
display columns instead.
"""

from hermes_cli.main import _clip_to_width, _disp_width, _pad_to_width


class TestDispWidth:
    def test_ascii_counts_one_column_per_char(self):
        assert _disp_width("hello") == 5

    def test_cjk_counts_two_columns_per_char(self):
        assert _disp_width("整理论文") == 8

    def test_mixed_content(self):
        assert _disp_width("更新README") == 10  # 2*2 + 6

    def test_control_chars_clamp_to_zero(self):
        # wcswidth() returns -1 for control chars; must not go negative.
        assert _disp_width("a\x07b") == 0

    def test_empty_string(self):
        assert _disp_width("") == 0


class TestClipToWidth:
    def test_ascii_clips_like_slice(self):
        assert _clip_to_width("abcdefgh", 5) == "abcde"

    def test_short_text_unchanged(self):
        assert _clip_to_width("abc", 10) == "abc"

    def test_cjk_clips_by_columns_not_chars(self):
        # 4 chars = 8 columns; only 3 chars (6 columns) fit in 6.
        assert _clip_to_width("整理论文", 6) == "整理论"

    def test_never_splits_a_double_width_char(self):
        # 5 columns can hold 2 CJK chars (4 cols); the 3rd (2 cols)
        # would straddle the boundary and must be dropped entirely.
        assert _clip_to_width("整理论文", 5) == "整理"

    def test_clipped_result_fits_width(self):
        for width in range(0, 12):
            assert _disp_width(_clip_to_width("整理论文和更新", width)) <= width


class TestPadToWidth:
    def test_ascii_matches_ljust(self):
        assert _pad_to_width("abc", 8) == "abc".ljust(8)

    def test_cjk_pads_by_display_columns(self):
        # 2 CJK chars = 4 columns -> 4 spaces to reach 8 columns.
        assert _pad_to_width("整理", 8) == "整理    "
        assert _disp_width(_pad_to_width("整理", 8)) == 8

    def test_overlong_text_not_truncated(self):
        assert _pad_to_width("abcdef", 3) == "abcdef"
