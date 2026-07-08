"""Regression test for #60258: skills prompt index never refreshes for
external_dirs changes in a long-running process.

Bug: ``build_skills_system_prompt`` has two caching layers that both fail
to invalidate when external-dir file contents change:

1. **Layer 1 in-process LRU** -- cache_key at prompt_builder.py:1458
   uses ``tuple(str(d) for d in external_dirs)`` (paths only, not contents).
   A new file in an external dir produces the same key, so the LRU
   serves stale text forever.

2. **Layer 2 disk snapshot** -- ``_build_skills_manifest`` only walks
   ``skills_dir``, not ``external_dirs``. The snapshot's manifest
   never sees external-dir file changes, so the snapshot persists
   stale external-dir state across process restarts.

This test exercises Layer 1. Layer 2 is fixed in the same PR by
extending ``_build_skills_manifest`` to also walk external_dirs (or by
adding a sibling helper).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_lru_cache_key_changes_when_external_dir_file_added(tmp_path, monkeypatch):
    """Adding a SKILL.md to an external skill dir must produce a new
    cache key. Without the fix, both states hash the same key and the
    LRU serves the pre-add snapshot forever in the same process.
    """
    from agent import prompt_builder

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    external_dir = tmp_path / "external_a"
    external_dir.mkdir()

    # We patch both ``get_skills_dir`` and ``get_all_skills_dirs`` so
    # ``build_skills_system_prompt`` sees our test fixtures rather
    # than reading from HERMES_HOME-driven config.
    monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: local_dir)
    monkeypatch.setattr(
        prompt_builder,
        "get_all_skills_dirs",
        lambda: [local_dir, external_dir],
    )

    # Clear the cache between calls.
    prompt_builder._SKILLS_PROMPT_CACHE.clear()

    # First call: external dir empty.
    prompt_builder.build_skills_system_prompt(
        available_tools=set(),
        available_toolsets=set(),
    )
    keys_before = list(prompt_builder._SKILLS_PROMPT_CACHE.keys())
    assert len(keys_before) == 1, (
        f"expected exactly one cache entry after first build, "
        f"got {keys_before!r}"
    )

    # Second call: external dir has a SKILL.md.
    (external_dir / "SKILL.md").write_text(
        "---\nname: fresh\ndescription: just added\n---\n",
        encoding="utf-8",
    )

    prompt_builder.build_skills_system_prompt(
        available_tools=set(),
        available_toolsets=set(),
    )
    keys_after = list(prompt_builder._SKILLS_PROMPT_CACHE.keys())

    # After the fix: a new key is created (LRU has 2 entries).
    # Without the fix: keys_before == keys_after (1 entry, the old one).
    assert len(keys_after) == 2, (
        f"LRU cache key did not change after adding external-dir "
        f"SKILL.md. keys_before={keys_before!r} keys_after={keys_after!r}. "
        f"Without a new key, the LRU serves stale text forever "
        f"(#60258)."
    )

    # Sanity: the new cache entry's value must differ from the old one
    # -- otherwise the new state was hashed to the same string as the
    # old state, which would be a different bug.
    new_key = (set(keys_after) - set(keys_before)).pop()
    old_value = prompt_builder._SKILLS_PROMPT_CACHE[keys_before[0]]
    new_value = prompt_builder._SKILLS_PROMPT_CACHE[new_key]
    assert new_value != old_value, (
        f"Cache values for distinct keys are identical -- the rendered "
        f"prompt did not reflect the new SKILL.md (#60258)."
    )


def test_disk_snapshot_manifest_includes_external_dirs(tmp_path, monkeypatch):
    """Layer 2 disk snapshot manifest must include external-dir
    fingerprints, not just the local skills_dir.
    """
    from agent import prompt_builder

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    external_dir = tmp_path / "external_a"
    external_dir.mkdir()

    monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: local_dir)
    monkeypatch.setattr(
        prompt_builder,
        "get_all_skills_dirs",
        lambda: [local_dir, external_dir],
    )

    # Build manifest with empty external dir.
    manifest_empty = prompt_builder._build_skills_manifest(local_dir)

    # Add a SKILL.md to the external dir.
    ext_skill = external_dir / "SKILL.md"
    ext_skill.write_text(
        "---\nname: another\ndescription: external\n---\n",
        encoding="utf-8",
    )

    # After the fix, _build_skills_manifest (or its replacement) must
    # walk external_dirs too, so the manifest reflects the new file.
    manifest_with_file = prompt_builder._build_skills_manifest(local_dir)
    # The manifest should reference the external file. We assert
    # at least one key contains the external-dir path or its parent.
    ext_mentioned = any(
        str(ext_skill) in k or str(external_dir) in k
        for k in manifest_with_file
    )
    assert ext_mentioned, (
        f"_build_skills_manifest does not include external-dir "
        f"fingerprints (manifest keys: {sorted(manifest_with_file)!r}). "
        f"External dir file changes are invisible to the manifest, so "
        f"the disk snapshot never invalidates (#60258)."
    )