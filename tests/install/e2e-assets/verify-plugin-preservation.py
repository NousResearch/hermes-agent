#!/usr/bin/env python3
"""Plugin upgrade-preservation verifier (Hermes release-harness hook).

Standalone and stdlib-only. Snapshot/verify are read-only against the home;
the explicit seed command creates controlled fixtures in a disposable home.

  snapshot  walk every plugin tree of a HERMES_HOME (the active home's
            ``plugins/**`` plus each ``profiles/<name>/plugins/**`` tree,
            overridable with --profiles-dir) and record, per entry —
            including EMPTY DIRECTORIES and the tree roots themselves —
            kind (file/dir/symlink/junction), byte size + sha256 for
            regular files, link target, and a recursive fingerprint of a
            symlink's external target (so an externally-owned sidecar
            witness file cannot be tampered with unnoticed) -> JSON file.
  verify    re-walk the same home and diff against a snapshot. FAILS
            (exit 1) on any deleted or modified entry. New entries are
            reported but do not fail: an upgrade may add files; it may
            not take yours away or alter them.

Unreadable paths are a hard error (exit 2), never a silent skip: a
scanner that cannot see a file cannot defend it. The verifier never
creates, deletes or writes anything under the scanned home.

Usage:
  python verify-plugin-preservation.py snapshot --home <HERMES_HOME> --out snap.json
  python verify-plugin-preservation.py verify --home <HERMES_HOME> --snapshot snap.json [--report r.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

SCHEMA_VERSION = 2
PROFILES_DIR = "profiles"
PLUGIN_ROOT = "plugins"
CHUNK = 1 << 20
# Upper bound for the recursive fingerprint of a symlink's external target:
# the harness points links at small controlled fixtures, but a stray link at
# a huge tree must not explode the snapshot.
MAX_EXTERNAL_ENTRIES = 10000


class ScanError(Exception):
    pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _is_link(path: str) -> bool:
    """True for symlinks everywhere, and for NTFS junctions on Windows
    (st.islink() is False for junctions, but readlink() works). POSIX
    behavior is untouched: FILE_ATTRIBUTE_REPARSE_POINT only exists on NT."""
    if os.path.islink(path):
        return True
    st = os.lstat(path)
    reparse = getattr(st, "st_file_attributes", 0)
    return bool(reparse & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _lstat_exists(path: str) -> bool:
    """Exists including dangling symlinks/junctions (os.path.exists hides
    those by resolving)."""
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False


def plugin_roots(home: str, profiles_dir: str | None = None) -> list[str]:
    """Every plugin tree that must survive an upgrade: the active home's
    plugins/ root plus each profile's plugins/ root (found under
    profiles_dir, default <home>/profiles). Roots are detected with lstat so
    a root that is itself a (possibly dangling) symlink is still scanned and
    fingerprinted. Returns stable "<home>/..." labels for existing roots."""
    roots: list[str] = []
    if _lstat_exists(os.path.join(home, PLUGIN_ROOT)):
        roots.append("<home>/" + PLUGIN_ROOT)
    profiles = profiles_dir or os.path.join(home, PROFILES_DIR)
    if os.path.isdir(profiles):
        for name in sorted(os.listdir(profiles)):
            p_root = os.path.join(profiles, name, PLUGIN_ROOT)
            if _lstat_exists(p_root):
                label = os.path.relpath(p_root, home).replace(os.sep, "/")
                roots.append("<home>/" + label)
    return roots


def _fingerprint_tree(root: str, budget: list[int]) -> dict:
    """Recursive content fingerprint of a directory (used for the EXTERNAL
    target of a plugin-runtime symlink: the sidecar witness is owned outside
    the home, so its content must be hashed here, not just listed)."""
    out: dict[str, dict] = {}
    def _on_walk_error(exc: OSError) -> None:
        raise ScanError(f"unreadable under {root}: {exc}") from exc
    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            if budget[0] <= 0:
                raise ScanError(f"external target too large to fingerprint: {root}")
            budget[0] -= 1
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            if _is_link(abs_path):
                out[rel] = {"kind": "symlink", "target": os.readlink(abs_path)}
            elif os.path.isdir(abs_path):
                out[rel] = {"kind": "dir"}
            elif os.path.isfile(abs_path):
                st = os.lstat(abs_path)
                out[rel] = {"kind": "file", "size": st.st_size,
                            "sha256": _sha256_file(abs_path)}
            else:
                out[rel] = {"kind": "other"}
    return out


def _entry_record(abs_path: str) -> dict:
    if _is_link(abs_path):
        rec: dict = {"kind": "symlink", "target": os.readlink(abs_path)}
        resolved = os.path.realpath(abs_path)
        rec["target_resolves"] = os.path.exists(resolved)
        if rec["target_resolves"]:
            if os.path.isfile(resolved):
                rec["target_kind"] = "file"
                rec["target_sha256"] = _sha256_file(resolved)
            elif os.path.isdir(resolved):
                rec["target_kind"] = "dir"
                rec["target_tree"] = _fingerprint_tree(resolved, [MAX_EXTERNAL_ENTRIES])
        return rec
    if os.path.isdir(abs_path):
        return {"kind": "dir"}
    if os.path.isfile(abs_path):
        st = os.lstat(abs_path)
        return {"kind": "file", "size": st.st_size, "sha256": _sha256_file(abs_path)}
    return {"kind": "other"}


def snapshot_home(home: str, profiles_dir: str | None = None) -> dict:
    home = os.path.abspath(home)
    roots = plugin_roots(home, profiles_dir)
    entries: dict[str, dict] = {}
    for root_label in roots:
        root = os.path.join(home, root_label[len("<home>/"):].replace("/", os.sep))
        # The root itself is an entry: if it is a symlink its identity and
        # target are recorded; if a plain dir it anchors empty-dir coverage.
        entries[root_label[len("<home>/"):]] = _entry_record(root)
        if _is_link(root):
            continue  # walking through it would double-record its target
        def _on_walk_error(exc: OSError) -> None:
            raise ScanError(f"unreadable under {root}: {exc}") from exc
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False,
                                                    onerror=_on_walk_error):
            dirnames.sort()
            for name in sorted(dirnames) + sorted(filenames):
                abs_path = os.path.join(dirpath, name)
                rel = os.path.relpath(abs_path, home).replace(os.sep, "/")
                entries[rel] = _entry_record(abs_path)
            dirnames[:] = [name for name in dirnames if not _is_link(os.path.join(dirpath, name))]
    return {
        "schema": SCHEMA_VERSION,
        "home": home,
        "roots": roots,
        "entries": entries,
    }


def verify_home(home: str, snap: dict) -> dict:
    home = os.path.abspath(home)
    now = snapshot_home(home, snap.get("_profiles_dir"))["entries"]
    before = snap["entries"]
    deleted = sorted(set(before) - set(now))
    added = sorted(set(now) - set(before))
    modified = {}
    for key in sorted(set(before) & set(now)):
        if before[key] != now[key]:
            modified[key] = {"before": before[key], "after": now[key]}
    return {
        "schema": SCHEMA_VERSION,
        "home": home,
        "snapshot_roots": snap["roots"],
        "counts": {
            "before": len(before),
            "after": len(now),
            "deleted": len(deleted),
            "modified": len(modified),
            "added": len(added),
        },
        "deleted": deleted,
        "modified": modified,
        "added": added,
        "ok": not deleted and not modified,
    }


def seed_fixtures(home: Path, external: Path) -> None:
    """Only create fresh fixtures; retries must not repair damaged witnesses."""
    wrapper = home / "plugins/mnemosyne-wrapper"
    profile = home / "profiles/e2e-preserve/plugins/second-plugin"
    for path in (wrapper, profile, external):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to replace preservation fixture: {path}")
    for path in (wrapper, profile, external):
        path.mkdir(parents=True)
    (wrapper / "mnemosyne-wrapper.json").write_text(
        '{"wrapper":true,"marker":"mnemosyne-wrapper","owner":"e2e-preservation"}\n', encoding="utf-8")
    (wrapper / "plugin.py").write_text('# directory wrapper fixture: no dependencies\n', encoding="utf-8")
    (profile / "marker.json").write_text('{"plugin":"second-plugin"}\n', encoding="utf-8")
    (profile / "data.bin").write_bytes(b"profile-plugin-bytes\n")
    (external / "sidecar-witness.txt").write_text("external-sidecar-witness-v1\n", encoding="utf-8")
    (external / "engine.bin").write_bytes(b"\x00\x01\x02external-engine\n")
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(external.resolve()), str(wrapper / "runtime"))
    else:
        (wrapper / "runtime").symlink_to(external.resolve(), target_is_directory=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    p_seed = sub.add_parser("seed", help="create fresh controlled fixtures in a disposable home")
    p_seed.add_argument("--home", required=True)
    p_seed.add_argument("--external", required=True)

    p_snap = sub.add_parser("snapshot", help="record plugin-tree state to JSON")
    p_snap.add_argument("--home", required=True)
    p_snap.add_argument("--out", required=True)
    p_snap.add_argument("--profiles-dir",
                        help="override the profiles root (default <home>/profiles)")

    p_ver = sub.add_parser("verify", help="compare current state to a snapshot")
    p_ver.add_argument("--home", required=True)
    p_ver.add_argument("--snapshot", required=True)
    p_ver.add_argument("--profiles-dir")
    p_ver.add_argument("--report")

    args = ap.parse_args(argv)

    if args.mode == "seed":
        seed_fixtures(Path(args.home), Path(args.external))
        return 0

    if not os.path.isdir(os.path.abspath(args.home)):
        print(f"error: --home is not a directory: {args.home}", file=sys.stderr)
        return 2

    if args.mode == "snapshot":
        try:
            snap = snapshot_home(args.home, args.profiles_dir)
        except (OSError, ScanError) as exc:
            print(f"snapshot failed: {exc}", file=sys.stderr)
            return 2
        if not snap["entries"]:
            print("snapshot INCONCLUSIVE: ZERO entries", file=sys.stderr)
            return 3
        if args.profiles_dir:
            snap["_profiles_dir"] = args.profiles_dir
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
        print(
            f"snapshot: {len(snap['entries'])} entries across "
            f"{len(snap['roots'])} plugin root(s) -> {args.out}"
        )
        return 0

    with open(args.snapshot, "r", encoding="utf-8") as fh:
        snap = json.load(fh)
    if args.profiles_dir:
        snap["_profiles_dir"] = args.profiles_dir
    try:
        report = verify_home(args.home, snap)
    except (OSError, ScanError) as exc:
        print(f"verify failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    if not snap["entries"]:
        print(
            "PLUGIN PRESERVATION INCONCLUSIVE: the snapshot is empty; "
            "refusing to claim preservation over zero recorded entries.",
            file=sys.stderr,
        )
        return 3
    if report["ok"]:
        c = report["counts"]
        print(
            f"plugin preservation OK: {c['before']} entries intact "
            f"({c['added']} added, 0 deleted, 0 modified)"
        )
        return 0
    print(rendered, file=sys.stderr)
    print(
        "PLUGIN PRESERVATION FAILED: "
        f"{report['counts']['deleted']} deleted, "
        f"{report['counts']['modified']} modified",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
