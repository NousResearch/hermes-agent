#!/usr/bin/env python3
"""Guard against drift between the embedded prompt-quality-canon copies.

The canon is embedded in three places — workflow-builder references,
agent-builder references, and the agent-builder asset emitted into built
agents — with no in-file sync note, because the loaded files are LLM-facing
and a maintenance comment there is paid on every load. This test is the
sync mechanism instead: all three copies must be byte-identical.
Run with: python3 -m pytest test_canon_sync.py
(or plain `python3 test_canon_sync.py` for a lightweight self-check).
"""
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[3]

CANON_COPIES = [
    SKILLS_DIR / "bmad-workflow-builder" / "references" / "prompt-quality-canon.md",
    SKILLS_DIR / "bmad-agent-builder" / "references" / "prompt-quality-canon.md",
    SKILLS_DIR / "bmad-agent-builder" / "assets" / "prompt-quality-canon.md",
]


def test_all_copies_exist():
    missing = [str(p) for p in CANON_COPIES if not p.is_file()]
    assert not missing, f"canon copy missing: {missing}"


def test_all_copies_identical():
    contents = {p: p.read_bytes() for p in CANON_COPIES if p.is_file()}
    reference = CANON_COPIES[0]
    diverged = [
        str(p)
        for p, body in contents.items()
        if body != contents.get(reference)
    ]
    assert not diverged, (
        "canon copies have drifted from "
        f"{reference}: {diverged} — sync all copies together"
    )


if __name__ == "__main__":
    test_all_copies_exist()
    test_all_copies_identical()
    print(f"ok: {len(CANON_COPIES)} canon copies present and identical")
