#!/usr/bin/env python3
"""Distribution spine — single source of truth for Hub mirrors and package.

Reads distribution.yaml. Commands:

  verify   Fail if mirrors drift, SKILL.md support list wrong, or package parity fails
  write    Refresh mirrors, regenerate SKILL.md hub list, rsync hub package
  report   JSON report of current state

Error codes: NG-PKG-* (see docs/HERMES_DISTRIBUTION.md)

Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DIST_FILE = SKILL_ROOT / "distribution.yaml"

# Hermes Hub path extraction (must match hermes-agent tools/skills_hub.py)
_HUB_LINK_RE = re.compile(
    r"(?:\]\(|`|(?:^|[\s\"']))((?:references|templates|scripts|assets|examples)/[^\s)`\"'<>]+)",
    re.MULTILINE,
)

MARK_BEGIN = "<!-- BEGIN HUB_SUPPORT_FILES (generated; do not edit) -->"
MARK_END = "<!-- END HUB_SUPPORT_FILES -->"


# ---------------------------------------------------------------------------
# Minimal YAML subset (enough for distribution.yaml; stdlib only)
# ---------------------------------------------------------------------------


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s or s.startswith("#"):
        return None
    if "#" in s and not (s.startswith('"') or s.startswith("'")):
        s = s.split("#", 1)[0].rstrip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in {"true", "True"}:
        return True
    if s in {"false", "False"}:
        return False
    if s in {"null", "Null", "~"}:
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a constrained YAML subset: nested maps + lists of scalars/maps."""
    lines = text.splitlines()
    root: dict[str, Any] = {}
    # stack of (indent, container)
    stack: list[tuple[int, Any]] = [(-1, root)]

    def current_container(indent: int) -> Any:
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        return stack[-1][1]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        cont = current_container(indent)

        if content.startswith("- "):
            item_raw = content[2:].strip()
            if isinstance(cont, list):
                if ":" in item_raw and not item_raw.startswith(("{", "[")):
                    # list of maps: - key: value
                    key, _, val = item_raw.partition(":")
                    obj: dict[str, Any] = {key.strip(): _parse_scalar(val)}
                    cont.append(obj)
                    stack.append((indent, obj))
                else:
                    cont.append(_parse_scalar(item_raw))
            else:
                raise ValueError(f"list item under non-list at line {i + 1}: {line}")
            i += 1
            continue

        if ":" in content:
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            if not isinstance(cont, dict):
                # continuing a list-of-maps entry
                if stack and isinstance(stack[-1][1], dict) and stack[-1][0] == indent:
                    cont = stack[-1][1]
                else:
                    raise ValueError(f"map key under non-map at line {i + 1}: {line}")
            if val == "":
                # peek next non-empty
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("#")):
                    j += 1
                if j < len(lines):
                    nxt = lines[j]
                    nindent = len(nxt) - len(nxt.lstrip(" "))
                    if nindent > indent and nxt.strip().startswith("- "):
                        child: list[Any] = []
                        cont[key] = child
                        stack.append((indent, child))
                    elif nindent > indent:
                        child_m: dict[str, Any] = {}
                        cont[key] = child_m
                        stack.append((indent, child_m))
                    else:
                        cont[key] = {}
                else:
                    cont[key] = {}
            else:
                cont[key] = _parse_scalar(val)
            i += 1
            continue

        raise ValueError(f"unparsed YAML line {i + 1}: {line}")
    return root


def load_distribution(root: Path = SKILL_ROOT) -> dict[str, Any]:
    path = root / "distribution.yaml"
    if not path.is_file():
        raise FileNotFoundError("NG-PKG-001: distribution.yaml missing")
    return load_simple_yaml(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    out: list[Path] = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out.append(p)
    return out


def expand_globs(root: Path, patterns: list[str], excludes: list[str] | None = None) -> list[str]:
    excludes = excludes or []
    found: set[str] = set()
    for pat in patterns:
        # support ** and *
        if "**" in pat or "*" in pat:
            # manual walk
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if any(fnmatch.fnmatch(rel, ex) or fnmatch.fnmatch(p.name, ex) for ex in excludes):
                    continue
                if fnmatch.fnmatch(rel, pat):
                    found.add(rel)
        else:
            p = root / pat
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if not any(fnmatch.fnmatch(rel, ex) for ex in excludes):
                    found.add(rel)
    return sorted(found)


def extract_hub_refs(skill_md: str) -> set[str]:
    paths: set[str] = set()
    for m in _HUB_LINK_RE.finditer(skill_md.replace("\\", "/")):
        from urllib.parse import unquote, urlsplit

        raw = unquote(urlsplit(m.group(1).rstrip(".,;:")).path)
        paths.add(raw)
    return paths


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def sync_mirrors(root: Path, dist: dict[str, Any]) -> list[str]:
    """Copy canonical sources → destinations. Returns list of updated rel paths."""
    updated: list[str] = []
    for m in dist.get("mirrors") or []:
        kind = m.get("kind") or "dir"
        src = root / str(m["source"])
        dst = root / str(m["destination"])
        if kind == "file":
            if not src.is_file():
                raise FileNotFoundError(f"NG-PKG-002: mirror source missing: {m['source']}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            updated.append(str(m["destination"]))
        else:
            if not src.is_dir():
                raise FileNotFoundError(f"NG-PKG-002: mirror source missing: {m['source']}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            updated.append(str(m["destination"]) + "/")
    return updated


def verify_mirrors(root: Path, dist: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for m in dist.get("mirrors") or []:
        kind = m.get("kind") or "dir"
        src = root / str(m["source"])
        dst = root / str(m["destination"])
        if kind == "file":
            if not src.is_file():
                errors.append(f"NG-PKG-002: source missing: {m['source']}")
                continue
            if not dst.is_file():
                errors.append(f"NG-PKG-003: mirror missing: {m['destination']}")
                continue
            if file_sha256(src) != file_sha256(dst):
                errors.append(
                    f"NG-PKG-004: mirror drift {m['source']} != {m['destination']}"
                )
        else:
            if not src.is_dir():
                errors.append(f"NG-PKG-002: source missing: {m['source']}")
                continue
            if not dst.is_dir():
                errors.append(f"NG-PKG-003: mirror dir missing: {m['destination']}")
                continue
            src_files = {
                p.relative_to(src).as_posix(): file_sha256(p) for p in iter_files(src)
            }
            dst_files = {
                p.relative_to(dst).as_posix(): file_sha256(p) for p in iter_files(dst)
            }
            for rel, digest in src_files.items():
                if rel not in dst_files:
                    errors.append(
                        f"NG-PKG-003: mirror missing file: {m['destination']}/{rel}"
                    )
                elif dst_files[rel] != digest:
                    errors.append(
                        f"NG-PKG-004: mirror drift: {m['destination']}/{rel}"
                    )
            for rel in dst_files:
                if rel not in src_files:
                    errors.append(
                        f"NG-PKG-005: orphan mirror file: {m['destination']}/{rel}"
                    )
    return errors


def expected_support_files(root: Path, dist: dict[str, Any]) -> list[str]:
    globs = list(dist.get("hub_support_globs") or [])
    excludes = list(dist.get("hub_support_exclude") or [])
    return expand_globs(root, globs, excludes)


def render_support_section(files: list[str]) -> str:
    bullets = "\n".join(f"- `{f}`" for f in files)
    intro = (
        "Hermes Hub installs only `SKILL.md` plus **explicitly path-referenced** "
        "files under allowlisted dirs (`references/`, `templates/`, `scripts/`, "
        "`assets/`, `examples/`). The list below is **generated** from "
        "`distribution.yaml` — run `python scripts/distribution_spine.py write` "
        "after adding packaging files:\n\n"
    )
    return (
        f"{intro}"
        f"{MARK_BEGIN}\n"
        f"{bullets}\n"
        f"{MARK_END}\n"
    )


def update_skill_md_support_list(root: Path, dist: dict[str, Any]) -> bool:
    """Rewrite the hub support section. Returns True if file changed."""
    skill_cfg = dist.get("skill_md") or {}
    start = skill_cfg.get("section_start") or "### Hermes Hub support files"
    end = skill_cfg.get("section_end") or "Full tree also keeps"
    path = root / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    start_idx = text.find(start)
    if start_idx < 0:
        raise ValueError(f"NG-PKG-006: SKILL.md missing section start: {start!r}")
    # content after the heading line
    after_heading = text.find("\n", start_idx)
    if after_heading < 0:
        raise ValueError("NG-PKG-006: SKILL.md section start has no newline")
    end_idx = text.find(end, after_heading)
    if end_idx < 0:
        raise ValueError(f"NG-PKG-006: SKILL.md missing section end: {end!r}")

    files = expected_support_files(root, dist)
    new_body = "\n\n" + render_support_section(files) + "\n"
    new_text = text[: after_heading + 1] + new_body + text[end_idx:]
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def verify_skill_md_support(root: Path, dist: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = root / "SKILL.md"
    if not path.is_file():
        return ["NG-PKG-006: SKILL.md missing"]
    text = path.read_text(encoding="utf-8")
    expected = set(expected_support_files(root, dist))
    # Extract bullets inside markers if present, else all hub refs that look like support list
    if MARK_BEGIN in text and MARK_END in text:
        block = text.split(MARK_BEGIN, 1)[1].split(MARK_END, 1)[0]
        listed = set(re.findall(r"`((?:references|templates|scripts|assets|examples)/[^`]+)`", block))
    else:
        listed = set()
        errors.append(
            "NG-PKG-007: SKILL.md missing generated markers; run distribution_spine.py write"
        )

    missing_in_list = sorted(expected - listed)
    extra_in_list = sorted(listed - expected)
    for rel in missing_in_list:
        errors.append(f"NG-PKG-008: support list missing {rel}")
    for rel in extra_in_list:
        errors.append(f"NG-PKG-009: support list extra {rel}")

    # Every listed/expected path must exist as a file
    for rel in sorted(expected | listed):
        if not (root / rel).is_file():
            errors.append(f"NG-PKG-010: referenced support file missing: {rel}")

    # All hub-extracted refs from entire SKILL.md must be real files (no dirs)
    for rel in sorted(extract_hub_refs(text)):
        p = root / rel
        if not p.is_file():
            errors.append(
                f"NG-PKG-011: SKILL.md hub path is not a file (breaks hub install): {rel}"
            )

    # Unreferenced files under allowlisted dirs that are in expected set are ok;
    # flag packaging scripts that are required but not listed already covered.
    # Flag: expected file not present in any hub ref across whole document
    hub_refs = extract_hub_refs(text)
    for rel in sorted(expected):
        if rel not in hub_refs:
            errors.append(f"NG-PKG-012: expected hub file never path-referenced: {rel}")

    return errors


def sync_hub_package(root: Path, dist: dict[str, Any]) -> Path:
    pkg = dist.get("hub_package") or {}
    dest = root / str(pkg.get("root") or "skills/neon-genie")
    excludes = set(pkg.get("exclude") or [])
    dest.mkdir(parents=True, exist_ok=True)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        dir_path = Path(directory)
        for name in names:
            rel = (dir_path / name).relative_to(root).as_posix()
            if name in {".git", "out", "skills", ".superpowers", "__pycache__", ".venv", "venv"}:
                ignored.add(name)
                continue
            if name.endswith(".pyc"):
                ignored.add(name)
                continue
            if rel in {"docs/superpowers", "docs/assets"} or rel.startswith("docs/superpowers/") or rel.startswith("docs/assets/"):
                ignored.add(name)
            for ex in excludes:
                if fnmatch.fnmatch(name, ex) or fnmatch.fnmatch(rel, ex):
                    ignored.add(name)
        return ignored

    # Copy top-level entries selectively
    for child in sorted(root.iterdir()):
        name = child.name
        if name in {".git", "out", "skills", ".superpowers", "__pycache__", ".venv", "venv"}:
            continue
        if name.endswith(".pyc"):
            continue
        target = dest / name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target, ignore=ignore)
        else:
            shutil.copy2(child, target)

    # Ensure key docs
    docs_dest = dest / "docs"
    docs_dest.mkdir(parents=True, exist_ok=True)
    for f in ("PREMIERE.md", "DEMO.md", "ROADMAP.md", "HERMES_DISTRIBUTION.md", "README.md"):
        src = root / "docs" / f
        if src.is_file():
            shutil.copy2(src, docs_dest / f)
    if (root / "skills.sh.json").is_file():
        shutil.copy2(root / "skills.sh.json", dest / "skills.sh.json")
    (dest / ".gitignore").write_text(
        "out/\n__pycache__/\n*.pyc\n.superpowers/\n.venv/\nvenv/\n", encoding="utf-8"
    )
    return dest


def verify_package_parity(root: Path, dist: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pkg_rel = str((dist.get("hub_package") or {}).get("root") or "skills/neon-genie")
    pkg_root = root / pkg_rel
    if not pkg_root.is_dir():
        # Standalone leaf install (Hub allowlist tree or optional-skills/<cat>/neon-genie):
        # no nested monorepo package is expected.
        skill_md = root / "SKILL.md"
        if skill_md.is_file() and root.name in {"neon-genie", "skills"}:
            return []
        return [f"NG-PKG-013: hub package missing: {pkg_rel}"]
    patterns = list(dist.get("package_parity_globs") or [])
    for rel in expand_globs(root, patterns):
        # skip nested skills path
        if rel.startswith("skills/"):
            continue
        src = root / rel
        dst = pkg_root / rel
        if not src.is_file():
            continue
        if not dst.is_file():
            errors.append(f"NG-PKG-014: package missing {rel}")
            continue
        if file_sha256(src) != file_sha256(dst):
            errors.append(f"NG-PKG-015: package drift {rel}")
    return errors


def cmd_verify(root: Path, dist: dict[str, Any]) -> int:
    errors: list[str] = []
    errors.extend(verify_mirrors(root, dist))
    errors.extend(verify_skill_md_support(root, dist))
    errors.extend(verify_package_parity(root, dist))
    if errors:
        print("FAIL: distribution spine", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nRemediation: python scripts/distribution_spine.py write",
            file=sys.stderr,
        )
        return 1
    n = len(expected_support_files(root, dist))
    print("PASS: distribution spine")
    print(f"  mirrors: ok")
    print(f"  hub support files: {n}")
    print(f"  package parity: ok")
    return 0


def cmd_write(root: Path, dist: dict[str, Any]) -> int:
    updated = sync_mirrors(root, dist)
    skill_changed = update_skill_md_support_list(root, dist)
    dest = sync_hub_package(root, dist)
    print("WRITE: distribution spine")
    for u in updated:
        print(f"  mirror: {u}")
    print(f"  SKILL.md support list: {'updated' if skill_changed else 'unchanged'}")
    print(f"  package: {dest.relative_to(root)}")
    # re-verify
    return cmd_verify(root, dist)


def cmd_report(root: Path, dist: dict[str, Any]) -> int:
    mirror_errors = verify_mirrors(root, dist)
    skill_errors = verify_skill_md_support(root, dist)
    pkg_errors = verify_package_parity(root, dist)
    report = {
        "skill": dist.get("skill"),
        "schema_version": dist.get("schema_version"),
        "hub_support_files": expected_support_files(root, dist),
        "mirror_errors": mirror_errors,
        "skill_md_errors": skill_errors,
        "package_errors": pkg_errors,
        "ok": not (mirror_errors or skill_errors or pkg_errors),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neon Genie distribution spine")
    parser.add_argument(
        "command",
        nargs="?",
        default="verify",
        choices=["verify", "write", "report"],
        help="verify (default) | write | report",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SKILL_ROOT,
        help="Skill root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        dist = load_distribution(root)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.command == "verify":
        return cmd_verify(root, dist)
    if args.command == "write":
        return cmd_write(root, dist)
    if args.command == "report":
        return cmd_report(root, dist)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
