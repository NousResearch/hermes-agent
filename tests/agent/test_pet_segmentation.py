"""Segmentation contracts for pet row strips.

Fast, hermetic, no generation mocks — these are the structural contracts the
hatch retry policy keys off, so they belong in the default suite. The
image-processing suite in ``tests/agent/test_pet_generate.py`` is opt-in behind
``HERMES_RUN_SLOW_PET_TESTS``, which CI does not set; contracts that guard a
paid-retry decision should not live only there.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from agent.pet.generate import atlas

SLOT = 208
HEIGHT = 208


def _strip_of(widths: list[int]) -> Image.Image:
    """One row strip with one opaque ellipse per slot, each *widths[i]* px wide."""
    img = Image.new("RGBA", (SLOT * len(widths), HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, width in enumerate(widths):
        cx = i * SLOT + SLOT // 2
        draw.ellipse((cx - width // 2, 34, cx + width // 2, 174), fill=(60, 80, 200, 255))
    return img


def _merged_pair_strip(count: int = 6) -> Image.Image:
    """Slots 3 and 4 drawn as ONE connected blob straddling the gutter."""
    img = _strip_of([140] * count)
    draw = ImageDraw.Draw(img)
    draw.ellipse((3 * SLOT - 140, 24, 5 * SLOT - 76, 184), fill=(200, 80, 80, 255))
    return img


def _two_poses_in_one_slot_strip(count: int = 6) -> Image.Image:
    """One slot holds two poses merged side by side — wide, not tall.

    This is the reported "split into two" frame: padded segmentation succeeds
    (the blob sits inside its slot with margins), so only frame validation can
    reject it.
    """
    return _strip_of([60] * 3 + [300] + [60] * (count - 4))


def _frame(width: int, height: int, *, opaque: bool = True) -> Image.Image:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if opaque:
        ImageDraw.Draw(img).rectangle((0, 0, width - 1, height - 1), fill=(60, 80, 200, 255))
    return img


def test_merged_poses_raise_the_structural_error_not_a_bare_valueerror():
    # Strict mode must fail with the STRUCTURAL error type: the orchestrator
    # keys off this class to skip the remaining strict (paid) retries instead of
    # substring-matching error text (#87739).
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(_merged_pair_strip(), 6, method="components")


def test_strict_mode_surfaces_validation_failures_structurally_too():
    # Extraction can succeed while the ART is still unusable — a slot holding
    # two merged poses is one connected subject with margins, so padded slicing
    # accepts it and only validation rejects it. That is the same defect class
    # as an unsegmentable strip (the model drew two characters), so it must
    # reach the orchestrator as the same structural error, not a bare ValueError
    # that would trigger more paid strict re-rolls.
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(_two_poses_in_one_slot_strip(), 6, method="components")


def test_auto_still_salvages_a_strip_strict_mode_rejects():
    # ``auto`` is the lenient path: the new exception must not change its
    # behavior. Uses a strip strict mode genuinely rejects, so this exercises
    # the salvage path rather than a strip both methods accept.
    strip = _merged_pair_strip()
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(strip, 6, method="components")
    assert len(atlas.extract_strip_frames(strip, 6, method="auto")) == 6


def test_row_frames_collapsed_flags_slivers_and_passes_whole_poses():
    # A lenient salvage can "succeed" with thin fragments of a body that match
    # each other, so nothing marks them as outliers until compose rejects the
    # whole atlas. Compare the row's median width against what the character's
    # silhouette implies at that height.
    reference = (92, 145)
    assert atlas.row_frames_collapsed([_frame(92, 145) for _ in range(6)], reference) is None
    reason = atlas.row_frames_collapsed([_frame(32, 145) for _ in range(6)], reference)
    assert reason is not None and "sliver" in reason


def test_row_frames_collapsed_abstains_without_a_reference():
    # No identity anchor decoded (or an empty one) means no judgement: the gate
    # must not guess, and must not reject a row on a missing reference.
    frames = [_frame(32, 145) for _ in range(6)]
    assert atlas.row_frames_collapsed(frames, None) is None
    assert atlas.row_frames_collapsed(frames, (0, 0)) is None


def test_row_frames_collapsed_reports_a_row_with_no_art():
    assert atlas.row_frames_collapsed([_frame(92, 145, opaque=False) for _ in range(6)], (92, 145)) == "row has no visible frames"
