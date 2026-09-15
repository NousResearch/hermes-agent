"""Assembles the playbook's story from the two content modules."""

from __future__ import annotations

import content_a as a
import content_b as b


def build_story():
    story = []
    story += a.front_matter()
    for part in (a.part_1, a.part_2, a.part_3, a.part_4, a.part_5, a.part_6, a.part_7,
                 a.part_8, a.part_9, a.part_10, a.part_11, a.part_12,
                 b.part_13, b.part_14, b.part_15, b.part_16, b.part_17, b.part_18,
                 b.part_19, b.part_20, b.part_21, b.part_22, b.part_23, b.part_24,
                 b.part_25):
        story += part()
    return story
