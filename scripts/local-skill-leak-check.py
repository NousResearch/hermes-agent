#!/usr/bin/env python3
"""local-skill-leak-check — detect skills authored in the GITIGNORED local tree,
and provide the OFFICIAL/BUNDLED veto the leak-relocator MUST honor before it
deletes anything (A3 / G3).

WHY THIS SCRIPT
---------------
The fleet shares skills via the git-tracked ``skills-shared/<group>/`` tree. The
local ``skills/`` tree and every ``profiles/<p>/skills/`` overlay are blanket
gitignored, so a skill authored there is **single-copy-on-disk: not backed up,
not shareable**. This checker reports those leaks (read-only; exit 0 = clean,
1 = leak(s) found).

THE G3 SAFETY THE RELOCATOR NEEDS
---------------------------------
An hourly sweep (``pending-queue-sweep.py``) auto-relocates leaked non-queue
skills out of the local tree and then ``rmtree``s the original. It decides
"this is a leak" purely from THIS checker's output. **Bundled/official skills
legitimately live in the local tree** (e.g. ``petdex`` ships inside the package
at ``hermes-agent/skills/<group>/petdex`` and is *supposed* to be local). It
once got eaten, and the only scar is a hand-added allowlist entry. That is a
manually-maintained denylist-of-exceptions, not a structural rule.

``is_official_skill()`` / ``relocate_is_safe()`` make officialness a **computed
property** so the relocator can never ``rmtree`` an official/shared skill again,
with NO hand-maintenance. A skill is official/protected when ANY of:

  1. its ``<profile>/<name>`` key is in ``local-skills-allowlist.txt`` (the
     existing allowlist — the relocator RE-CHECKS it independently, because the
     relocator's job is deletion and it must not trust an upstream filter alone),
  2. a **bundled twin** exists — a same-named ``SKILL.md`` under the installed
     package's bundled tree ``hermes-agent/skills/*/<name>/`` (this is what makes
     ``petdex`` official *by construction*), or
  3. a **git-tracked shared twin** exists — a same-named skill already under
     ``skills-shared/*/<name>/`` (relocating would be redundant/destructive; an
     applier should FOLD, the sweep must not rmtree).

Zero Hermes deps (runs under a bare ``/usr/bin/python3``, like the sweep
scripts). Prefix/exclusion parity with ``agent/skill_utils`` is asserted by the
test suite so drift is caught in CI, never at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Kept in sync with agent/skill_utils.EXCLUDED_SKILL_DIRS via the parity test.
# (Hardcoded so this script imports zero Hermes deps and runs under bare python3.)
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
# Hermes-local vendoring dir (holds third-party SKILL.md under deploy venvs);
# not in the index set because the index never scans it.
EXTRA_EXCLUDED_DIR_NAMES = frozenset({"_lib"})

_ALL_EXCLUDED = EXCLUDED_DIR_NAMES | EXTRA_EXCLUDED_DIR_NAMES

# Work-queue note prefixes — MUST stay byte-identical to
# agent/skill_utils.QUEUE_NOTE_PREFIXES (asserted by the parity test). Queue
# notes are relocated wholesale by the sweep; they are never "official".
QUEUE_NOTE_PREFIXES = (
    "pending-shared-skill-patches",
    "pending-shared-patches",
    "pending-patch-",
)


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))


def _allowlist_path() -> Path:
    # Canonical location is the skill ROOT (one level up from scripts/), which is
    # what SKILL.md documents. Fall back to the legacy script-adjacent path.
    root = Path(__file__).resolve().parent.parent / "local-skills-allowlist.txt"
    legacy = Path(__file__).resolve().parent / "local-skills-allowlist.txt"
    if root.exists():
        return root
    if legacy.exists():
        return legacy
    return root


def load_allowlist(path: Path | None = None) -> set[str]:
    """Return the set of allowed ``<profile>/<name>`` keys (reasons stripped)."""
    p = path or _allowlist_path()
    allowed: set[str] = set()
    if not p.exists():
        return allowed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split("#", 1)[0].strip()
        if key:
            allowed.add(key)
    return allowed


def _is_excluded(rel_parts: tuple[str, ...]) -> bool:
    return any(part in _ALL_EXCLUDED for part in rel_parts)


def is_queue_note_name(name: str) -> bool:
    """True if *name* (a skill-dir basename) is a reserved work-queue note prefix."""
    return bool(name) and name.startswith(QUEUE_NOTE_PREFIXES)


def _scan_tree(root: Path, profile: str) -> list[tuple[str, Path]]:
    """Return [(key, skill_dir)] for every real SKILL.md under ``root``.

    ``key`` is ``<profile>/<skill-name>``. Excluded/venv/archive dirs skipped.
    """
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _ALL_EXCLUDED]
        if "SKILL.md" in filenames:
            skill_dir = Path(dirpath)
            rel = skill_dir.relative_to(root)
            if _is_excluded(rel.parts):
                continue
            name = skill_dir.name
            found.append((f"{profile}/{name}", skill_dir))
    return found


def find_leaks(home: Path | None = None, allowlist: set[str] | None = None) -> list[tuple[str, Path]]:
    """Return [(key, skill_dir)] of un-allowlisted skills in gitignored local trees."""
    h = home or _hermes_home()
    allowed = allowlist if allowlist is not None else load_allowlist()
    leaks: list[tuple[str, Path]] = []

    # 1. default-profile local tree: ~/.hermes/skills/
    leaks += _scan_tree(h / "skills", "default")

    # 2. every profile overlay: ~/.hermes/profiles/<p>/skills/
    profiles_dir = h / "profiles"
    if profiles_dir.is_dir():
        for prof in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
            leaks += _scan_tree(prof / "skills", prof.name)

    return [(k, d) for (k, d) in leaks if k not in allowed]


# ── G3 — official/bundled/shared-twin veto the relocator MUST honor ───────────


def _bundled_skills_root(bundled_root: Path | str | None = None) -> Path | None:
    """Resolve the installed package's bundled skills tree (hermes-agent/skills).

    Precedence: explicit arg → this repo's own ``skills/`` (this file lives at
    ``<repo>/scripts/local-skill-leak-check.py``, so ``../skills``). Returns None
    only when nothing resolvable exists — never a hardcoded ``~/.hermes`` path
    (profile-sandbox-safe by construction).
    """
    if bundled_root:
        p = Path(bundled_root)
        return p if p.is_dir() else None
    repo_bundled = Path(__file__).resolve().parent.parent / "skills"
    if repo_bundled.is_dir():
        return repo_bundled
    return None


def _has_named_twin(root: Path | None, name: str) -> bool:
    """True if a ``<root>/*/<name>/SKILL.md`` (one level of grouping) exists."""
    if root is None or not root.is_dir():
        return False
    try:
        for group in root.iterdir():
            if not group.is_dir() or group.name in _ALL_EXCLUDED:
                continue
            if (group / name / "SKILL.md").is_file():
                return True
    except OSError:
        return False
    return False


def is_official_skill(
    name: str,
    *,
    profile: str = "default",
    home: Path | None = None,
    allowlist: set[str] | None = None,
    bundled_root: Path | str | None = None,
) -> tuple[bool, str]:
    """Is ``name`` an official/protected skill the relocator must NOT delete?

    Returns ``(is_official, reason)``. Officialness is a COMPUTED property (no
    hand-maintenance): allowlisted, OR a bundled twin ships in the package, OR a
    git-tracked shared twin already lives under skills-shared/. This is the veto
    a relocator re-checks INDEPENDENTLY before any ``rmtree`` — the eaten-skill
    (petdex) class is closed without needing a hand-added allowlist entry.
    """
    h = home or _hermes_home()
    allowed = allowlist if allowlist is not None else load_allowlist()

    key = f"{profile}/{name}"
    if key in allowed:
        return True, f"allowlisted ({key})"

    if _has_named_twin(_bundled_skills_root(bundled_root), name):
        return True, f"bundled twin exists (package skills/*/{name})"

    if _has_named_twin(h / "skills-shared", name):
        return True, f"shared twin exists (skills-shared/*/{name})"

    return False, ""


def relocate_is_safe(
    name: str,
    skill_dir: Path,
    *,
    profile: str = "default",
    home: Path | None = None,
    allowlist: set[str] | None = None,
    bundled_root: Path | str | None = None,
) -> tuple[bool, str]:
    """Gate a relocator MUST call before ``rmtree(skill_dir)``.

    Returns ``(safe_to_relocate, reason)``. Unsafe when the skill is official
    (any veto in ``is_official_skill``). A queue note is always safe (it is
    relocated wholesale, never treated as official). Fails SAFE: any doubt →
    return official/unsafe so a genuine skill is kept, never eaten.
    """
    if is_queue_note_name(name):
        return True, "queue-note (relocated wholesale)"
    official, reason = is_official_skill(
        name, profile=profile, home=home, allowlist=allowlist,
        bundled_root=bundled_root,
    )
    if official:
        return False, f"skip official/bundled '{name}' — {reason}; belongs local by design"
    return True, "no official/bundled/shared twin — genuine leak, safe to relocate"


def _remediation(key: str, skill_dir: Path) -> str:
    name = skill_dir.name
    return (
        f"  • {key}\n"
        f"      at: {skill_dir}\n"
        f"      fix: move to the git-tracked shared tree, e.g.\n"
        f"           mv '{skill_dir}' ~/.hermes/skills-shared/<group>/{name}\n"
        f"           git add ~/.hermes/skills-shared/<group>/{name}/   "
        f"# specific dir — NEVER 'git add -f <group>/' (sweeps __pycache__/.ruff_cache junk)\n"
        f"           git diff --cached --name-only   # verify ONLY the skill's files staged\n"
        f"      or, if it is a genuine single-agent local skill, add '{key}' (with a reason)\n"
        f"           to local-skills-allowlist.txt"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--home", help="override HERMES_HOME root")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report each leak's relocate SAFETY (official-veto) without touching disk",
    )
    args = ap.parse_args(argv)

    home = Path(args.home) if args.home else None
    leaks = find_leaks(home=home)

    if args.dry_run:
        rows = []
        for key, skill_dir in leaks:
            profile = key.split("/", 1)[0]
            safe, reason = relocate_is_safe(
                skill_dir.name, skill_dir, profile=profile, home=home
            )
            rows.append({"key": key, "path": str(skill_dir), "would_relocate": safe, "reason": reason})
        if args.json:
            print(json.dumps({"dry_run": rows}, indent=2))
        else:
            if not rows:
                print("✓ no local skill leaks — nothing to relocate")
            for r in rows:
                verb = "RELOCATE" if r["would_relocate"] else "SKIP"
                print(f"  [{verb}] {r['key']}  ({r['reason']})")
        return 1 if leaks else 0

    if args.json:
        print(json.dumps({"leaks": [{"key": k, "path": str(d)} for k, d in leaks]}, indent=2))
        return 1 if leaks else 0

    if not leaks:
        print("✓ no local skill leaks (all skills live in the git-backed shared tree)")
        return 0

    print(
        f"🔴 {len(leaks)} skill(s) authored in the GITIGNORED local tree "
        f"(single-copy-on-disk, NOT backed up):"
    )
    for key, skill_dir in leaks:
        print(_remediation(key, skill_dir))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
