#!/usr/bin/env python3
"""Reconcile the bundled-skills sync manifest against disk state.

A ``sync_skills()`` crash mid-loop (e.g. the #48200 scope-guard ValueError
fixed alongside this script) can replace a skill's on-disk content with the
new bundled version but skip the manifest write that records the new origin
hash — the manifest is only persisted *after* the loop completes. The next
sync then falsely reads that skill as ``user_modified`` (because the stale
origin hash no longer matches the now-updated disk) and skips it forever,
even though the disk copy is byte-identical to the bundled source.

This script finds those false positives and re-baselines the manifest entry
so future upstream changes flow in again. It is strictly safe:

  - A skill is re-baselined ONLY when its on-disk content is byte-identical
    to the bundled source. Genuine user modifications (disk != bundled) are
    never touched and are reported separately.
  - Skills with no bundled source (removed upstream, hub-installed, or
    locally authored) are ignored — not our concern.
  - Dry-run by default; pass ``--apply`` to write the manifest.

Concurrent runs: ``--apply`` does a read-modify-write on the manifest. A
``hermes update`` running at the same time could have its manifest write
overwritten (the file itself stays atomic/uncorrupt; last-writer-wins).
The reconcile pass is idempotent, so the worst case is a redundant entry
rather than data loss — but avoid running ``--apply`` while an update is
in progress.

Scopes to the active profile's HERMES_HOME. To reconcile a named profile,
point HERMES_HOME at it first:

    HERMES_HOME=~/.hermes/profiles/<name> \\
        python scripts/reconcile_skills_manifest.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap repo root onto sys.path so this runs as a standalone script
# regardless of the caller's CWD (``tools.skills_sync`` lives at repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.skills_sync import (  # noqa: E402  (path bootstrap above)
    _compute_relative_dest,
    _dir_hash,
    _discover_bundled_skills,
    _get_bundled_dir,
    _read_manifest,
    _write_manifest,
    SKILLS_DIR,
)


def _find_drift() -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Return ``(drift, genuine_edits, missing)`` for the active profile.

    - ``drift``: ``(name, stale_origin_hash, correct_bundled_hash)`` tuples to
      re-baseline. Disk == bundled, but the manifest origin is stale.
    - ``genuine_edits``: names where disk != bundled (left untouched).
    - ``missing``: manifest entries not on disk (user-deleted; ignored).
    """
    manifest = _read_manifest()
    bundled_dir = _get_bundled_dir()
    bundled_by_name = dict(_discover_bundled_skills(bundled_dir))

    drift: list[tuple[str, str, str]] = []
    genuine_edits: list[str] = []
    missing: list[str] = []

    for name, origin_hash in sorted(manifest.items()):
        src = bundled_by_name.get(name)
        if src is None:
            continue  # no bundled source upstream — skip
        dest = _compute_relative_dest(src, bundled_dir)
        if not dest.exists():
            missing.append(name)
            continue
        # A file where a skill directory is expected (e.g. a stray file
        # colliding with a skill name) hashes to the empty digest because
        # ``_dir_hash``'s rglob finds no files inside it. Guard against the
        # theoretical case where an empty bundled dir could then match it.
        if not dest.is_dir():
            continue
        disk_hash = _dir_hash(dest)
        bundled_hash = _dir_hash(src)
        if disk_hash == bundled_hash:
            if origin_hash != bundled_hash:
                drift.append((name, origin_hash, bundled_hash))
            # else: already correct — nothing to do
        else:
            genuine_edits.append(name)

    return drift, genuine_edits, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the reconciled manifest (default: dry-run, report only).",
    )
    args = parser.parse_args()

    print(f"Skills dir: {SKILLS_DIR}")
    drift, genuine, missing = _find_drift()

    if not drift:
        print("✓ No manifest drift detected — nothing to reconcile.")
        if genuine:
            print(f"  ({len(genuine)} genuine user-modified skill(s) left untouched)")
        return 0

    print(f"\nFound {len(drift)} drifted manifest entr{'y' if len(drift) == 1 else 'ies'} "
          f"(disk == bundled, origin hash stale):")
    for name, old, new in drift:
        print(f"  {name}: {old[:12]} -> {new[:12]}")

    if genuine:
        print(f"\nLeaving {len(genuine)} genuine user modification(s) untouched "
              f"(disk != bundled):")
        for name in genuine:
            print(f"  {name}")

    if missing:
        print(f"\n{len(missing)} manifest entr{'y' if len(missing) == 1 else 'ies'} "
              f"not on disk (user-deleted; ignored).")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write the reconciled manifest.")
        return 0

    manifest = _read_manifest()
    for name, _old, new in drift:
        manifest[name] = new
    _write_manifest(manifest)
    print(f"\n✓ Re-baselined {len(drift)} manifest entr{'y' if len(drift) == 1 else 'ies'}. "
          "Future `hermes update` runs will resume updating these skills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
