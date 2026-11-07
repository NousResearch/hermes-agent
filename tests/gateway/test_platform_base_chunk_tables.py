"""truncate_message keeps pipe tables readable when a table is larger than one chunk."""

import re

from gateway.platforms.base import BasePlatformAdapter

HEADER = "| name | value |\n|:---|---:|"
ROWS = [f"| row{i} | {'y' * 40} |" for i in range(120)]


def _body(chunk: str) -> str:
    return re.sub(r" \(\d+/\d+\)$", "", chunk)


def test_oversized_table_repeats_header_in_every_chunk_and_keeps_rows_once():
    chunks = BasePlatformAdapter.truncate_message("Results:\n\n" + HEADER + "\n" + "\n".join(ROWS), 2000)

    assert len(chunks) > 2
    assert all(len(c) <= 2000 for c in chunks)
    for chunk in chunks:
        lines = _body(chunk).split("\n")
        first_row = next(i for i, line in enumerate(lines) if line.startswith("| row"))
        assert lines[first_row - 2:first_row] == HEADER.split("\n"), chunk[:120]
    rows = [line for c in chunks for line in _body(c).split("\n") if line.startswith("| row")]
    assert rows == ROWS

    # Pipes inside a code fence are code, not a table: nothing is injected.
    fenced = "```text\n" + HEADER + "\n" + "\n".join(ROWS) + "\n```"
    chunks = BasePlatformAdapter.truncate_message(fenced, 2000)
    assert sum(c.count("| name | value |") for c in chunks) == 1


def test_split_never_strands_a_header_without_its_rows():
    rows = "\n".join(f"| r{i} | {'v' * 30} |" for i in range(80))
    for pad in range(0, 400, 3):
        content = "intro" + " word" * pad + "\n" + HEADER + "\n" + rows + "\nend"
        for max_length in (300, 500, 900):
            for chunk in BasePlatformAdapter.truncate_message(content, max_length):
                last = _body(chunk).rstrip().split("\n")[-1]
                assert last not in HEADER.split("\n"), (pad, max_length, chunk[-80:])
