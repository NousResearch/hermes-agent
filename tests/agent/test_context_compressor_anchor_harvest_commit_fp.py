"""Anchor Index ``commits:`` harvest must not fabricate commits (#141267).

The commits line is published as a verbatim session_search anchor, so every
harvested value must look like a real hash: pure-digit runs (build numbers,
timestamps) and fragments of hyphenated ids (memory UUIDs split at ``-`` by
``\\b``) are false anchors that can never hit.
"""
import agent.context_compressor as cc


def _commits_harvested(text: str) -> set[str]:
    index = cc._build_anchor_index([{"role": "user", "content": text}])
    for line in index.splitlines():
        if line.startswith("commits: "):
            return {v.split("(x", 1)[0] for v in line[len("commits: "):].split(", ")}
    return set()


def test_commits_harvest_rejects_pure_digits_and_uuid_segments_keeps_real_hashes():
    text = (
        "archived memory d4e74cab-c16c-4e5b-a434-5aad31e0988f and "
        "d3a24c89-3983-4943-96ec-ccc8db9e44c1; ESET thread reports build "
        "262209022 from https://forum.eset.com/topic/build-262209022; "
        "fix landed in abc123def45"
    )
    got = _commits_harvested(text)
    uuids = "d4e74cab-c16c-4e5b-a434-5aad31e0988f d3a24c89-3983-4943-96ec-ccc8db9e44c1"
    # False-anchor legs from the report: bare build number + UUID pieces.
    assert "262209022" not in got
    assert "5aad31e0988f" not in got
    assert "ccc8db9e44c1" not in got
    assert all(v not in uuids for v in got)
    # Behavior guard: a real hash is still harvested.
    assert "abc123def45" in got


def test_commits_harvest_uuid_only_text_yields_no_fabricated_commits():
    # Id mid-message: the hyphen lookbehind must veto every segment.
    assert not _commits_harvested("memory id d4e74cab-c16c-4e5b-a434-5aad31e0988f archived")
    # Id at absolute message start (no preceding char): the lookbehind must
    # still veto the 12-hex tail, unlike the old \b (pos-0 boundary is free).
    assert not _commits_harvested("d4e74cab-c16c-4e5b-a434-5aad31e0988f archived")
