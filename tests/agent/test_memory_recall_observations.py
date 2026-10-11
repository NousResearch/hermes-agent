"""Shared recall deduplication must remove only self-contained identical observations (#133976)."""

import pytest

from agent.memory_manager import build_memory_context_block


OBSERVATIONS = [
    "- CaseABC quota=10.",
    "[2026-10-06 12:00:00] CaseABC quota=10.",
    "[2026-10-06T12:00:00Z] CaseABC quota=10.",
    "[2026-10-06T12:00:00+08:00] CaseABC quota=10.",
    "[2026-10-06T12:00:00.123Z] CaseABC quota=10.",
]


@pytest.mark.parametrize("line", OBSERVATIONS)
@pytest.mark.parametrize("boundary", ["", "\n## Next peer\n", "\n---\n"])
def test_standalone_identical_observations_are_stated_once(line, boundary):
    raw = f"## Peer A\n{line}\n{line}{boundary}"
    expected = f"## Peer A\n{line}{boundary}"
    block = build_memory_context_block(raw)
    assert block.count(line) == 1
    assert expected in block
    # A fresh composition has its own scope; no previous block can suppress it.
    assert build_memory_context_block(line).count(line) == 1


@pytest.mark.parametrize("line", OBSERVATIONS)
def test_record_boundaries_and_provenance_are_lossless(line):
    cases = [
        f"## Peer A\n{line}\n## Peer B\n{line}",
        f"Profile: A\n{line}\nProfile: B\n{line}",
        f"{line}\n---\n{line}",
        f"{line}\n**Other section**\n{line}",
        f"{line}\n{line.replace('CaseABC', 'caseABC')}",
        f"{line}\n{line.replace('quota=10', 'quota=11')}",
        f"{line} [fact=a]\n{line} [fact=b]",
        f"{line}\n  source A\n{line}\n  source B",
        f"{line}\n\n  source A\n{line}\n\n  source B",
        f"{line}\n  source A\n{line}",
        f"```text\n{line}\n{line}\n```",
        f"~~~text\n{line}\n{line}\n~~~",
        "1. Same numbered item\n1. Same numbered item",
        "Ordinary repeated prose.\nOrdinary repeated prose.",
        "[2026-02-30 12:00:00] invalid date.\n[2026-02-30 12:00:00] invalid date.",
        "[2026-10-06T12:00:00+24:00] invalid zone.\n[2026-10-06T12:00:00+24:00] invalid zone.",
        "[2026-10-06] date only.\n[2026-10-06] date only.",
        f"    {line}\n    {line}",
        f"````text\n{line}\n```\n{line}\n````",
        f"```text\n{line}\n~~~\n{line}\n```",
        f"{line}\n```text\n{line}\n```\n{line}",

    ]
    if line.startswith("["):
        cases.extend([
            f"{line}\n{line.replace('12:00:00', '12:00:01')}",
            f"{line}\n{line}\nUnindented continuation belonging to the second record.",
        ])
    for raw in cases:
        block = build_memory_context_block(raw)
        assert raw in block, raw
