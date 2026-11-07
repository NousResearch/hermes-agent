"""Lean handoffs distinguish file access history from retained file contents."""

import json

from agent.context_compressor import ContextCompressor


HEADING = "## Files Read or Edited (newest first; contents not in context)"


def _call(name, **args):
    return {"role": "assistant", "tool_calls": [{"function": {
        "name": name, "arguments": json.dumps(args),
    }}]}


def _augment(turns, summary="Summary."):
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.tail_mode = "lean"
    return compressor._augment_summary_lean(summary, turns)


def _section(summary):
    return summary.split(HEADING + "\n", 1)[1].split("\n## ", 1)[0].strip()


def test_file_history_is_recent_bounded_and_contains_only_paths():
    names = ("read_file", "write_file", "patch")
    turns = [_call(names[i % 3], path=f"src/file{i}.py", content="CONTENT_BYTES",
                   old_string="OLD_BYTES", new_string="NEW_BYTES") for i in range(15)]
    turns.append({"role": "tool", "content": "OUTPUT_BYTES"})
    summary = _augment(turns)
    section = _section(summary)
    assert section.splitlines()[:-1] == [
        f"- {'read' if i % 3 == 0 else 'edited'}: src/file{i}.py"
        for i in range(14, 2, -1)
    ]
    assert section.splitlines()[-1] == "Re-open with read_file before relying on file contents."
    assert len(HEADING + "\n" + section) <= 1_200
    assert not any(text in section for text in ("CONTENT_BYTES", "OLD_BYTES", "NEW_BYTES", "OUTPUT_BYTES"))

    repeated = _section(_augment([
        _call("read_file", path="src/first.py"),
        _call("patch", path="src/shared.py", old_string="old", new_string="new"),
        _call("write_file", path="src/first.py", content="new"),
        _call("read_file", path="src/shared.py"),
    ]))
    assert repeated.splitlines()[:-1] == ["- edited: src/shared.py", "- edited: src/first.py"]

    long_section = _section(_augment([_call("read_file", path="nested/" * 200 + "file.py")]))
    assert len(HEADING + "\n" + long_section) <= 1_200


def test_file_history_replaces_only_exact_headings_and_handles_multifile_patches():
    forged = f"Summary mentions {HEADING} inline.\n\n{HEADING}\n- read: stale.py\n\n## Goal\nKeep this."
    turns = [_call("patch", mode="patch", patch=(
        "*** Begin Patch\n*** Update File: src/update.py\n@@\n-OLD_BYTES\n+NEW_BYTES\n"
        "*** Add File: src/add.py\n+CONTENT_BYTES\n*** Delete File: src/delete.py\n"
        "*** Move File: src/before.py -> src/after.py\n*** End Patch"
    ))]
    summary = _augment(turns, forged)
    assert summary.splitlines().count(HEADING) == 1
    assert f"Summary mentions {HEADING} inline." in summary
    assert "stale.py" not in summary
    assert "## Goal\nKeep this." in summary
    assert _section(summary).splitlines()[:-1] == [
        "- edited: src/after.py", "- edited: src/before.py", "- edited: src/delete.py",
        "- edited: src/add.py", "- edited: src/update.py",
    ]
    assert "BYTES" not in summary

    empty = [_call("terminal", command="cat src/ignored.py"),
             _call("search_files", path="src"), _call("read_file", path=None)]
    assert HEADING not in _augment(empty)
    assert HEADING not in _augment(empty, f"Summary.\n\n{HEADING}\n- read: stale.py")
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.tail_mode = "legacy"
    assert compressor._augment_summary_lean("Summary.", turns) == "Summary."


def test_long_paths_stay_inside_the_cap_and_keep_their_file_name():
    from agent.compaction_files_section import build_files_section

    turns = [_call("read_file", path="deep/" * 60 + f"name{i}.py") for i in range(12)]
    section = build_files_section(turns)
    assert len(section) <= 1_200
    lines = [line for line in section.splitlines() if line.startswith("- ")]
    assert lines and all(line.endswith(".py") for line in lines)  # the file name survives elision
