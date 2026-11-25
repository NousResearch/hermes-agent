"""The idle scratch pruner honors an operator-installed protection list and fails closed.

Regression for the 2026-11-22 incident: the built-in idle pruner removed
979,371,788 bytes of unique archive content (``cache/scratch/arc_verify``)
and a 4,053,192,704-byte probe database (``cache/scratch/costprobe.db``).
Both matched the operator's ``protected_patterns``; the rule that deletes
never consulted them, so protected, unique content was removable — and was
removed.

The pruner now loads the same protection source the quota/age rules use
(``<home>/scripts/scratch_governance.py``, by file path) and keeps every
top-level entry whose subtree matches a protected pattern. A stock install
without the source prunes exactly as before; a source that exists but
cannot be read deletes NOTHING that pass (fail closed).
"""

import os
import time

import pytest

import hermes_constants_scratch as scratch

OLD = time.time() - 48 * 3600

# A minimal stand-in for the operator's protection source: the same three
# call sites the pruner consults (load_quota_config / protected_patterns /
# is_protected), protecting the two prefixes named on 2026-11-22.
PROTECTION_SOURCE = '''
import fnmatch
import os


def load_quota_config(write_if_missing=False):
    return {"protected_patterns": ["arc_verify*", "outbox_archives*"]}


def protected_patterns(cfg):
    return list(cfg.get("protected_patterns") or [])


def is_protected(path, patterns, root=None):
    rel = os.path.relpath(path, root) if root else str(path)
    return any(
        fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(os.path.basename(str(path)), p)
        for p in patterns
    )
'''


def _home(tmp_path):
    home = tmp_path / "home"
    (home / "cache" / "scratch").mkdir(parents=True)
    (home / "scripts").mkdir()
    return home


def _idle_entry(scratch_root, name, filename, text):
    entry = scratch_root / name
    entry.mkdir()
    target = entry / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    p = target
    while True:
        os.utime(p, (OLD, OLD))
        if p == entry:
            break
        p = p.parent
    return entry


def _install_source(home, text=PROTECTION_SOURCE):
    (home / "scripts" / "scratch_governance.py").write_text(text, encoding="utf-8")


def test_protected_entry_survives_unprotected_entry_dies(tmp_path):
    home = _home(tmp_path)
    _install_source(home)
    root = home / "cache" / "scratch"
    protected = _idle_entry(root, "arc_verify", "outbox_archives/blob.jsonl", "x" * 64)
    unprotected = _idle_entry(root, "junk_entry", "big.bin", "y" * 64)

    removed = scratch.prune_idle_entries(root, 24, frozenset(), None)

    assert protected.exists(), (
        "the protected entry was deleted although 'arc_verify*' is a "
        "protected pattern")
    assert not unprotected.exists(), (
        "the guard is protecting too much: the unprotected idle entry "
        "should have been pruned")
    assert removed == 1


def test_negative_control_the_entry_was_really_prunable(tmp_path):
    """Without the protection source the SAME entry IS pruned.

    Without this control, `protected.exists()` above proves nothing — the
    entry might have survived for an unrelated reason.
    """
    home = _home(tmp_path)
    root = home / "cache" / "scratch"
    entry = _idle_entry(root, "arc_verify", "outbox_archives/blob.jsonl", "x" * 64)

    # no protection source installed: stock behaviour, prune as before
    assert scratch.prune_idle_entries(root, 24, frozenset(), None) == 1
    assert not entry.exists(), (
        "the entry should be prunable without the protection source — "
        "otherwise the survival test above proves nothing")

    # source installed + entry restored: it survives again
    _install_source(home)
    restored = _idle_entry(root, "arc_verify", "outbox_archives/blob.jsonl", "x" * 64)
    scratch.prune_idle_entries(root, 24, frozenset(), None)
    assert restored.exists()


def test_nested_protected_path_protects_its_top_level_entry(tmp_path):
    """A protected path nested under an entry protects the whole entry:
    deleting the entry would delete the protected content."""
    home = _home(tmp_path)
    _install_source(home)
    root = home / "cache" / "scratch"
    entry = _idle_entry(root, "carry", "outbox_archives/deep.jsonl", "x" * 64)

    scratch.prune_idle_entries(root, 24, frozenset(), None)
    assert entry.exists(), (
        "a protected path nested under a top-level entry must keep that "
        "entry (deleting the entry deletes the protected content)")


def test_unreadable_protection_source_fails_closed(tmp_path):
    home = _home(tmp_path)
    _install_source(home, "raise RuntimeError('boom')\n")
    root = home / "cache" / "scratch"
    entry = _idle_entry(root, "arc_verify", "f.bin", "z" * 64)

    removed = scratch.prune_idle_entries(root, 24, frozenset(), None)

    assert removed == 0
    assert entry.exists(), (
        "a protection source that exists but cannot be read must delete "
        "NOTHING (fail closed), not prune everything")
