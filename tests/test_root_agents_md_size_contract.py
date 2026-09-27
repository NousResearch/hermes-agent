"""Root AGENTS.md must stay under the subdirectory-hint truncation ceiling.

agent/subdirectory_hints.py loads the root AGENTS.md as a hint on the first
tool touch of the repo root and truncates it at ``_MAX_HINT_CHARS = 32_000``
(head 70% + tail 20% — the middle ~30% of a 36k file is silently dropped from
what the agent sees). The file grew past the ceiling in Sept 2026, so the
Dependency Pinning Policy, Commits/Merges/PRs and the back half of Testing
disappeared from every agent's context while the log only whispered a
WARNING. This contract keeps the file under the ceiling so the routing-table
pattern (root stays small, detail lives in per-area AGENTS.md files) holds.

This is a size contract between two pieces of data (the hint ceiling and the
root file), not a change-detector: it pins the *relationship* the hint loader
depends on.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hint_ceiling() -> int:
    """Read the ceiling from the loader itself — the contract's other side."""
    import re
    hints_py = REPO_ROOT / "agent" / "subdirectory_hints.py"
    text = hints_py.read_text(encoding="utf-8")
    match = re.search(r"_MAX_HINT_CHARS\s*=\s*([\d_]+)", text)
    assert match, "subdirectory_hints.py no longer defines _MAX_HINT_CHARS"
    return int(match.group(1).replace("_", ""))


def test_root_agents_md_stays_under_the_hint_truncation_ceiling():
    ceiling = _hint_ceiling()
    size = len((REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    assert size < ceiling, (
        f"AGENTS.md is {size} chars, over the {ceiling} subdirectory-hint "
        "ceiling: agent/subdirectory_hints.py truncates it head+tail and the "
        "middle sections never reach the agent's context. Move detail to a "
        "per-area AGENTS.md file and link it from the routing table."
    )


def test_root_agents_md_keeps_headroom_under_the_ceiling():
    """Not just 'under' — keep 10% headroom so routine doc edits don't tip it
    over between CI runs (the ceiling check alone would gate exactly at the
    cliff edge)."""
    ceiling = _hint_ceiling()
    size = len((REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    assert size < ceiling * 0.9, (
        f"AGENTS.md is {size} chars, within 10% of the {ceiling} hint "
        "ceiling — move detail out before the next growth spurt puts it over."
    )


def test_root_agents_md_routing_table_links_tests_area_file():
    """The moved Testing detail must remain reachable: the routing table (the
    file's own discovery mechanism) links tests/AGENTS.md."""
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "tests/AGENTS.md" in text, (
        "Routing table should point contributors at tests/AGENTS.md for the "
        "detailed testing rules."
    )
    assert (REPO_ROOT / "tests" / "AGENTS.md").is_file(), (
        "tests/AGENTS.md should exist once Testing detail moves there."
    )


def test_moved_sections_live_in_tests_agents_md():
    """The four detailed rules moved out of the root must survive in the
    tests area file (checked by heading, not body text: content stays free to
    evolve there)."""
    moved = (REPO_ROOT / "tests" / "AGENTS.md").read_text(encoding="utf-8")
    for heading in (
        "Don't fake the host OS",
        "Don't write change-detector tests",
        "Never read source code in tests",
    ):
        assert heading in moved, f"'{heading}' moved out of root AGENTS.md must live in tests/AGENTS.md"
