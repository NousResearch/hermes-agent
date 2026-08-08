"""Seam identity for context_compressor_skill_prune extract (LB2).

Part of #78645 + #78647.
"""

from agent import context_compressor as cc
from agent import context_compressor_skill_prune as sp


def test_all_members_resolve_is_identical_through_godfile():
    members = [
        "SKILL_PRUNED_MARKER_PREFIX",
        "_SKILL_VIEW_PRUNE_MIN_CHARS",
        "_MAX_PRUNED_SKILL_MARKERS",
        "_SKILL_PRUNED_MARKER_RE",
        "_PRUNED_SKILLS_SECTION_HEADING",
        "_SKILL_PRUNE_RECENT_WINDOW",
        "_skill_pruned_marker",
        "_extract_pruned_skill_names",
        "_collect_ghosted_skill_names",
        "_reinject_pruned_skill_markers",
        "_skill_view_call_sites",
        "_collect_protected_skill_names",
    ]
    for m in members:
        assert getattr(cc, m) is getattr(sp, m), f"{m} not is-identical"


def test_no_duplicate_defs_in_godfile():
    from pathlib import Path

    src = Path(cc.__file__).read_text(encoding="utf-8")
    for name in [
        "_skill_pruned_marker",
        "_extract_pruned_skill_names",
        "_collect_ghosted_skill_names",
        "_reinject_pruned_skill_markers",
        "_skill_view_call_sites",
        "_collect_protected_skill_names",
    ]:
        assert src.count(f"def {name}") == 0, f"duplicate def {name} left in godfile"
    assert "context_compressor_skill_prune" in src


def test_behavior_smoke():
    # marker build + extract round trip
    marker = cc._skill_pruned_marker("my-skill")
    assert marker.startswith("[SKILL_PRUNED:")
    assert "my-skill" in marker
    names = cc._extract_pruned_skill_names(marker)
    assert "my-skill" in names
    # ghosted collection on a turn list
    turns = [{"role": "user", "content": f"loaded {marker}"}]
    ghosted = cc._collect_ghosted_skill_names(turns)
    assert "my-skill" in ghosted
    # reinject restores marker
    out = cc._reinject_pruned_skill_markers("summary text", ["my-skill"])
    assert "[SKILL_PRUNED:" in out


def test_import_orders_no_cycle():
    import importlib

    import agent.context_compressor_skill_prune as a
    import agent.context_compressor as b

    importlib.reload(a)
    importlib.reload(b)
    assert b._skill_pruned_marker is a._skill_pruned_marker
