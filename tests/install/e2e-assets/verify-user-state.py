#!/usr/bin/env python3
"""User-state upgrade-preservation verifier (Hermes install/update E2E hook).

Answers the question the install/update E2E legs only answered indirectly:
after an upgrade, is the user's *own* state still there?

Standalone and stdlib-only. Both modes are read-only against the home; nothing
is ever created, deleted or written under it. There is deliberately NO `seed`
mode: the state this verifies must be produced by the product through the
ordinary user path (a real chat turn, `hermes auth add`, `hermes profile
create`), never hand-written by the harness. Assertions over fixtures we wrote
ourselves would only prove the harness can write files.

  snapshot  walk the durable state of a HERMES_HOME and record, per entry —
            kind (file/dir/symlink), byte size + sha256 for regular files, link
            target, and, for every state.db, its ROW COUNTS — into JSON.
  verify    re-walk and diff against a snapshot. FAILS (exit 1) on any deleted
            or modified entry. New entries are reported but tolerated: an
            upgrade may add files, it may not take yours away or alter them.

Two things are deliberately NOT failed on:

* ``config.yaml`` (root and per-profile) is rewritten by config migration. The
  rewrite is additive by design, so it is reported, never judged.
* ``skills/**`` is seeded and re-synced from the bundled library at startup and
  after every update (``tools.skills_sync.sync_skills``) at BOTH roots -- the
  home's own tree and every profile's. It changes on an ordinary release, so it
  is advisory-only. The exception is ``skills/.archive/**``, which holds
  restorable *user* skills and therefore is judged like everything else.

Ownership: the ``plugins/**`` trees belong to verify-plugin-preservation.py —
that contract (nothing deleted or modified, additions tolerated) is already
enforced separately, so this tool does not double-own them.

Unreadable paths are a hard error (exit 2), never a silent skip: a scanner that
cannot see a file cannot defend it. An empty snapshot is refused as
inconclusive (exit 3) rather than reported as a pass.

Usage:
  python verify-user-state.py snapshot --home <HERMES_HOME> --out snap.json
  python verify-user-state.py verify --home <HERMES_HOME> --snapshot snap.json [--report r.json]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

SCHEMA_VERSION = 1
PROFILES_DIR = "profiles"
PLUGIN_ROOT = "plugins"
SKILLS_ROOT = "skills"
SKILL_ARCHIVE = ".archive"
CHUNK = 1 << 20

# Top-level files whose content must survive verbatim.
JUDGED_FILES = ("config.yaml", ".env", "auth.json", "state.db", "gateway_state.json")
# Trees that must survive: every entry under them is judged.
JUDGED_ROOTS = ("memories", "cron", "sessions", "profiles", "photon",
                "desktop-plugins", "tui-widgets", "skins", "pets",
                "skills/" + SKILL_ARCHIVE)
# Trees recorded for the report but never judged (see the module docstring).
ADVISORY_ROOTS = (SKILLS_ROOT,)

# The tree's own migrations clear dead provider vars out of .env (the old setup
# wizard wrote LLM_MODEL/OPENAI_MODEL; config.yaml is the source of truth now).
# A key the CURRENT tree retires is not user state, so the upgrade clearing it
# is reported, not failed. Derived from the migration source so a newly retired
# var cannot drift out of this set; unreadable source retires nothing, which
# keeps every .env change fatal.
MIGRATION_SOURCE = Path(__file__).resolve().parents[3] / "hermes_cli" / "config_migrations.py"
EMPTY_VALUE_DIGEST = hashlib.sha256(b"").hexdigest()[:12]
# state.db tables whose row counts stand in for "the user's data is still here".
# Counting rows rather than hashing bytes: a live SQLite file changes for
# benign reasons (WAL checkpoint, migration, a later turn).
COUNTED_TABLES = ("sessions", "messages", "usage", "cron_jobs")


class ScanError(Exception):
    pass


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _is_link(path: str) -> bool:
    """True for symlinks everywhere, and for NTFS junctions on Windows
    (st.islink() is False for junctions, but readlink() works)."""
    if os.path.islink(path):
        return True
    st = os.lstat(path)
    reparse = getattr(st, "st_file_attributes", 0)
    return bool(reparse & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _lexists(path: str) -> bool:
    """Exists including dangling symlinks/junctions (os.path.exists resolves)."""
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _db_counts(path: str) -> dict | None:
    """Row counts for COUNTED_TABLES present in the db, or None if unreadable."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        names = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for table in COUNTED_TABLES:
            if table in names:
                counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts
    except sqlite3.Error:
        return None
    finally:
        con.close()


def _owns_plugins(rel: str, *, profiles: bool) -> bool:
    """plugins/** belongs to verify-plugin-preservation.py, at both roots."""
    parts = rel.split("/")
    if not profiles:
        return parts[0] == PLUGIN_ROOT
    # profiles/<name>/plugins/...
    return len(parts) >= 3 and parts[2] == PLUGIN_ROOT


def _is_bundled_skill(rel: str, *, profiles: bool) -> bool:
    """skills/** except skills/.archive/**, at both roots (see module docstring)."""
    parts = rel.split("/")
    if not profiles:
        return parts[0] == SKILLS_ROOT and not (
            len(parts) > 1 and parts[1] == SKILL_ARCHIVE)
    if len(parts) >= 3 and parts[2] == SKILLS_ROOT:
        return not (len(parts) > 3 and parts[3] == SKILL_ARCHIVE)
    return False


def _env_key_hashes(path: str) -> dict[str, str]:
    """Per-key VALUE digests for a dotenv-style file: names and equality only.

    ``.env`` is a secrets file, so the verifier must never carry its content --
    but "an upgrade rewrote this file" is useless without knowing which variable
    moved, and an equal-size rewrite is invisible in the whole-file hash. Key
    names plus 12-char value digests name the writer and leak nothing.
    """
    digests: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return digests
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export "):].lstrip()
        key, sep, value = text.partition("=")
        key = key.strip()
        if sep and key:
            digests[key] = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:12]
    return digests


def _env_key_diff(before: dict, after: dict) -> dict:
    left = before.get("env_keys") or {}
    right = after.get("env_keys") or {}
    return {
        "keys_added": sorted(set(right) - set(left)),
        "keys_removed": sorted(set(left) - set(right)),
        "keys_changed": sorted(key for key in set(left) & set(right) if left[key] != right[key]),
    }


def _clears_loop_var(node: ast.AST, loop_var: str) -> bool:
    """``save_env_value(loop_var, "")`` -- the migration's retire-this-key write."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_env_value"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == loop_var
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == ""
    )


def retired_env_vars(source: Path | None = None) -> frozenset[str]:
    """Provider vars the tree's own migrations clear to empty.

    Matches the loop form the 12 -> 13 migration uses
    (``for dead in ("X", "Y"): save_env_value(dead, "")``) -- the only shape
    that both names the keys and acts on them. A migration written another way
    is simply not derived, which fails a leg loudly instead of silently
    tolerating a real loss.
    """
    path = MIGRATION_SOURCE if source is None else source
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, UnicodeError):
        return frozenset()
    retired: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not any(_clears_loop_var(call, node.target.id) for call in ast.walk(node)):
            continue
        retired.update(
            element.value for element in ast.walk(node.iter)
            if isinstance(element, ast.Constant) and isinstance(element.value, str))
    return frozenset(retired)


def retired_env_clear(rel: str, pair: dict, retired: frozenset[str] | None = None) -> list[str]:
    """Names a .env change may carry: the tree's retired vars, and emptied.

    Everything else stays fatal -- an added or removed key, a live key's value,
    or a retired key the migration did NOT clear. Values are compared as the
    same digests the snapshot records, so "emptied" is exact.
    """
    known = retired_env_vars() if retired is None else retired
    if os.path.basename(rel) != ".env" or not known:
        return []
    diff = _env_key_diff(pair["before"], pair["after"])
    if diff["keys_added"] or diff["keys_removed"] or not diff["keys_changed"]:
        return []
    if any(key not in known for key in diff["keys_changed"]):
        return []
    after_keys = pair["after"].get("env_keys") or {}
    if any(after_keys.get(key) != EMPTY_VALUE_DIGEST for key in diff["keys_changed"]):
        return []
    return diff["keys_changed"]


def _env_line_summary(path: str) -> dict:
    """Line-level shape of a .env without any content: totals, blanks, key order.

    The per-key digests deliberately ignore comments, blank lines and ordering, so
    a file rewritten only in those ways reports as modified with nothing named --
    which is exactly the "0 deleted, 1 modified" with no `variables ...` line.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return {"lines": 0, "comments": 0, "blanks": 0, "key_order": []}
    comments = sum(1 for line in lines if line.strip().startswith("#"))
    blanks = sum(1 for line in lines if not line.strip())
    order = [line.split("=", 1)[0].strip() for line in lines
             if "=" in line and line.strip() and not line.strip().startswith("#")]
    return {"lines": len(lines), "comments": comments, "blanks": blanks, "key_order": order}


def _entry_record(abs_path: str) -> dict:
    if _is_link(abs_path):
        record: dict = {"kind": "symlink", "target": os.readlink(abs_path)}
        record["target_resolves"] = os.path.exists(os.path.realpath(abs_path))
        return record
    if os.path.isdir(abs_path):
        return {"kind": "dir"}
    if os.path.isfile(abs_path):
        st = os.lstat(abs_path)
        record = {"kind": "file", "size": st.st_size}
        if os.path.basename(abs_path).endswith(".db"):
            counts = _db_counts(abs_path)
            if counts is not None:
                # Row counts, not bytes: a byte hash of a live db is noise.
                record["rows"] = counts
            else:
                record["sha256"] = _sha256_file(abs_path)
        else:
            record["sha256"] = _sha256_file(abs_path)
        if os.path.basename(abs_path) == ".env":
            record["env_keys"] = _env_key_hashes(abs_path)
            record["env_lines"] = _env_line_summary(abs_path)
        return record
    return {"kind": "other"}


def _walk_root(home: str, root_rel: str, *, profiles: bool, judged: bool):
    """Yield (rel_path, abs_path, judged) for every entry under a root, pruning
    the trees owned elsewhere (plugins).

    The third field is per ENTRY, not per root: a bundled skills tree is seeded
    and re-synced from the library at startup and after every update at BOTH
    roots (``tools.skills_sync``), so it is advisory-only -- except
    ``<skills>/.archive/**``, which holds restorable USER skills and therefore
    stays judged.
    """
    root_abs = os.path.join(home, root_rel.replace("/", os.sep))
    if not _lexists(root_abs):
        return
    yield root_rel, root_abs, judged and not _is_bundled_skill(root_rel, profiles=profiles)
    # Top-level entries include plain FILES (config.yaml, state.db): only a real
    # directory can be descended, and a symlink must never be followed.
    if _is_link(root_abs) or not os.path.isdir(root_abs):
        return
    stack = [root_abs]
    while stack:
        current = stack.pop()
        try:
            names = sorted(os.listdir(current))
        except OSError as exc:
            raise ScanError(f"unreadable under {home}: {exc}") from exc
        for name in names:
            abs_path = os.path.join(current, name)
            rel = os.path.relpath(abs_path, home).replace(os.sep, "/")
            if _owns_plugins(rel, profiles=profiles):
                continue
            # The judged skills/.archive tree is walked as its own root, so the
            # advisory walk of skills/ must not claim it.
            if not judged and (rel.split("/")[-1] == SKILL_ARCHIVE or "/" + SKILL_ARCHIVE + "/" in rel):
                continue
            entry_judged = judged and not _is_bundled_skill(rel, profiles=profiles)
            if _is_link(abs_path):
                yield rel, abs_path, entry_judged
                continue
            if os.path.isdir(abs_path):
                yield rel, abs_path, entry_judged
                stack.append(abs_path)
            else:
                yield rel, abs_path, entry_judged


def snapshot_home(home: str, profiles_dir: str | None = None) -> dict:
    home = os.path.abspath(home)
    entries: dict[str, dict] = {}
    advisory: dict[str, dict] = {}

    targets: list[tuple[str, bool, bool]] = []   # (rel, profiles_style, judged)
    for name in JUDGED_FILES:
        targets.append((name, False, True))
    for name in JUDGED_ROOTS:
        targets.append((name, name.split("/")[0] == PROFILES_DIR, True))
    for name in ADVISORY_ROOTS:
        targets.append((name, False, False))

    for rel_root, profiles, judged in targets:
        for rel, abs_path, entry_judged in _walk_root(home, rel_root, profiles=profiles, judged=judged):
            (entries if entry_judged else advisory)[rel] = _entry_record(abs_path)

    # Profiles may live somewhere else entirely (a test harness override).
    if profiles_dir and os.path.abspath(profiles_dir) != os.path.join(home, PROFILES_DIR):
        override_abs = os.path.abspath(profiles_dir)
        if _lexists(override_abs):
            entries["profiles"] = {"kind": "dir"}
            for name in sorted(os.listdir(override_abs)):
                abs_path = os.path.join(override_abs, name)
                rel = f"{PROFILES_DIR}/{name}"
                entries[rel] = _entry_record(abs_path)
                if os.path.isdir(abs_path) and not _is_link(abs_path):
                    for sub_rel, sub_abs, sub_judged in _walk_root(
                            override_abs, name, profiles=True, judged=True):
                        if sub_rel == name:
                            continue
                        (entries if sub_judged else advisory)[f"{PROFILES_DIR}/{sub_rel}"] = _entry_record(sub_abs)

    return {
        "schema": SCHEMA_VERSION,
        "home": home,
        "entries": entries,
        "advisory": advisory,
    }


def _diff(before: dict, after: dict) -> dict:
    return {
        "deleted": sorted(set(before) - set(after)),
        "added": sorted(set(after) - set(before)),
        "modified": {key: {"before": before[key], "after": after[key]}
                     for key in sorted(set(before) & set(after))
                     if before[key] != after[key]},
    }


def _modification_allowed(rel: str) -> bool:
    """Entries whose *bytes* may legitimately change during an upgrade.

    config.yaml is rewritten by config migration: additive, by design.
    state.db is a live SQLite file — its size moves whenever rows are written,
    so its real contract is the row-count check below, not byte equality.
    The cron ticker stamps its own liveness on a schedule of its own, inside the
    verified window or not: they are never user state. On a cold home they show up
    as ordinary additions (tolerated); once the ticker exists, the same file moves.
    """
    name = rel.rsplit("/", 1)[-1]
    if name in ("config.yaml", "state.db") or name.endswith(".db"):
        return True
    parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
    is_cron = parent == "cron" or parent.endswith("/cron")
    return is_cron and name in ("ticker_heartbeat", "ticker_last_success")


def _volatile_sidecar(rel: str) -> bool:
    """SQLite's own sidecars (-wal/-shm/-journal): hardware, not user state.

    A WAL/shm pair exists only while a connection is open. A snapshot taken with one
    live sees them; they disappear when it closes, and the report then says the
    upgrade DELETED user state (observed: cron/executions.db-shm and -wal). The
    database itself stays judged -- state.db by row counts -- so real loss still fails.
    """
    return rel.endswith(("-wal", "-shm", "-journal"))


def _rows_shrank(rel: str, before: dict, after: dict) -> bool:
    if not rel.rsplit("/", 1)[-1].endswith(".db"):
        return False
    old, new = before.get("rows"), after.get("rows")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    return any(isinstance(old.get(k), int) and isinstance(new.get(k), int)
               and new[k] < old[k] for k in old)


def verify_home(home: str, snap: dict) -> dict:
    home = os.path.abspath(home)
    fresh = snapshot_home(home, snap.get("_profiles_dir"))
    judged = _diff(snap["entries"], fresh["entries"])
    advisory = _diff(snap.get("advisory", {}), fresh["advisory"])

    failing_modified = {k: v for k, v in judged["modified"].items()
                        if not _modification_allowed(k)}
    failing_deleted = sorted(k for k in judged["deleted"] if not _volatile_sidecar(k))
    tolerated_deleted = sorted(k for k in judged["deleted"] if _volatile_sidecar(k))
    shrank = sorted(k for k, v in judged["modified"].items()
                    if _rows_shrank(k, v["before"], v["after"]))
    tolerated_modified = sorted(set(judged["modified"]) - set(failing_modified))
    # A rewritten .env is fatal by design -- except when the tree's own migration
    # is what rewrote it, clearing a key that tree no longer reads.
    retired_env: dict[str, list[str]] = {}
    for rel, pair in list(failing_modified.items()):
        cleared = retired_env_clear(rel, pair)
        if cleared:
            retired_env[rel] = cleared
            del failing_modified[rel]
            tolerated_modified = sorted(set(tolerated_modified) | {rel})
    # The rest still names the variables that moved, or the caller is left
    # holding two hashes of a secrets file and no lead.
    for rel, pair in failing_modified.items():
        if os.path.basename(rel) == ".env":
            pair["key_diff"] = _env_key_diff(pair["before"], pair["after"])

    return {
        "schema": SCHEMA_VERSION,
        "home": home,
        "counts": {
            "before": len(snap["entries"]),
            "after": len(fresh["entries"]),
            "deleted": len(failing_deleted),
            "modified": len(failing_modified),
            "added": len(judged["added"]),
            "tolerated_modified": len(tolerated_modified),
            "tolerated_deleted": len(tolerated_deleted),
            "advisory_changed": len(advisory["deleted"]) + len(advisory["modified"])
                                + len(advisory["added"]),
        },
        "deleted": failing_deleted,
        "tolerated_deleted": tolerated_deleted,
        "modified": failing_modified,
        "added": judged["added"],
        "tolerated_modified": tolerated_modified,
        "retired_env_cleared": retired_env,
        "rows_shrank": shrank,
        "advisory": advisory,
        "ok": not failing_deleted and not failing_modified and not shrank,
    }


def _render(report: dict) -> str:
    lines = []
    counts = report["counts"]
    lines.append(
        f"user state: {counts['before']} -> {counts['after']} entries "
        f"({counts['deleted']} deleted, {counts['modified']} modified, "
        f"{counts['added']} added, {counts['tolerated_modified']} tolerated, "
        f"{counts['advisory_changed']} advisory)")
    for label, key in (("DELETED", "deleted"), ("MODIFIED", "modified")):
        for rel in report[key]:
            lines.append(f"  {label} {rel}")
    for rel, pair in report["modified"].items():
        detail = pair.get("key_diff")
        if not detail:
            continue
        named = ", ".join(
            f"{kind}={','.join(detail[field])}"
            for kind, field in (("added", "keys_added"), ("removed", "keys_removed"),
                                ("changed", "keys_changed"))
            if detail.get(field))
        if named:
            lines.append(f"    {rel}: variables {named} (names only; values are never recorded)")
        else:
            lines.append(f"    {rel}: no key differs -- comments/order/blanks only "
                         f"(before {pair['before'].get('env_lines')}, after {pair['after'].get('env_lines')}; "
                         f"counts and names, never content)")
    for rel in report["rows_shrank"]:
        lines.append(f"  ROWS SHRANK {rel}")
    for rel, names in report.get("retired_env_cleared", {}).items():
        lines.append(f"  tolerated (retired var cleared: {','.join(names)}) {rel}")
    for rel in report["tolerated_modified"]:
        if rel in report.get("retired_env_cleared", {}):
            continue
        lines.append(f"  tolerated (config rewrite) {rel}")
    for rel in report.get("tolerated_deleted", []):
        lines.append(f"  tolerated (sqlite sidecar) {rel}")
    advisory = report["advisory"]
    for label, key in (("advisory deleted", "deleted"), ("advisory modified", "modified")):
        for rel in advisory[key]:
            lines.append(f"  {label} (skills sync) {rel}")
    return "\n".join(lines)


def env_key_report(home: str, profiles_dir: str | None = None) -> list[str]:
    """Every .env this verifier judges, with its key names -- names only.

    Exists to be compared against the driver's own view of ``$HERMES_HOME/.env``.
    The install e2e's probe showed OPENAI_BASE_URL present immediately before AND
    after the snapshot while the report called it an ADDITION, which can only mean
    the snapshot read a different file than the run wrote. Printing the paths the
    verifier actually enumerated settles that in one run instead of three.
    """
    snap = snapshot_home(home, profiles_dir)
    lines: list[str] = []
    for label, entries in (("judged", snap["entries"]), ("advisory", snap.get("advisory", {}))):
        for rel in sorted(entries):
            if not rel.endswith(".env"):
                continue
            keys = sorted((entries[rel].get("env_keys") or {}).keys())
            path = os.path.join(os.path.abspath(home), rel.replace("/", os.sep))
            lines.append(f"{label} {path}: {' '.join(keys) if keys else '(no keys)'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    p_snap = sub.add_parser("snapshot", help="record durable user state to JSON")
    p_snap.add_argument("--home", required=True)
    p_snap.add_argument("--out", required=True)
    p_snap.add_argument("--profiles-dir",
                        help="override the profiles root (default <home>/profiles)")

    p_ver = sub.add_parser("verify", help="compare current state to a snapshot")
    p_ver.add_argument("--home", required=True)
    p_ver.add_argument("--snapshot", required=True)
    p_ver.add_argument("--report")
    p_ver.add_argument("--profiles-dir")

    p_env = sub.add_parser("env-keys",
                           help="print every judged .env and its key names (names only)")
    p_env.add_argument("--home", required=True)
    p_env.add_argument("--profiles-dir")

    args = ap.parse_args(argv)

    if not os.path.isdir(os.path.abspath(args.home)):
        print(f"error: --home is not a directory: {args.home}", file=sys.stderr)
        return 2

    if args.mode == "env-keys":
        try:
            for line in env_key_report(args.home, args.profiles_dir):
                print(line)
        except (OSError, ScanError) as exc:
            print(f"env-keys failed: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.mode == "snapshot":
        try:
            snap = snapshot_home(args.home, args.profiles_dir)
        except (OSError, ScanError) as exc:
            print(f"snapshot failed: {exc}", file=sys.stderr)
            return 2
        if not snap["entries"]:
            print("snapshot INCONCLUSIVE: zero judged entries — was this home "
                  "ever used? Refusing to record an empty contract.",
                  file=sys.stderr)
            return 3
        if args.profiles_dir:
            snap["_profiles_dir"] = args.profiles_dir
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(snap, handle, indent=2, sort_keys=True)
        print(f"snapshot: {len(snap['entries'])} judged entries "
              f"({len(snap['advisory'])} advisory) -> {args.out}")
        return 0

    with open(args.snapshot, "r", encoding="utf-8") as handle:
        snap = json.load(handle)
    if args.profiles_dir:
        snap["_profiles_dir"] = args.profiles_dir
    try:
        report = verify_home(args.home, snap)
    except (OSError, ScanError) as exc:
        print(f"verify failed: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered)

    if not snap.get("entries"):
        print("USER-STATE PRESERVATION INCONCLUSIVE: the snapshot is empty; "
              "refusing to claim preservation over zero recorded entries.",
              file=sys.stderr)
        return 3

    if report["ok"]:
        print("user-state preservation OK: " + _render(report))
        return 0
    print(_render(report), file=sys.stderr)
    counts = report["counts"]
    print("USER-STATE PRESERVATION FAILED: "
          f"{counts['deleted']} deleted, {counts['modified']} modified, "
          f"{len(report['rows_shrank'])} with fewer rows", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
