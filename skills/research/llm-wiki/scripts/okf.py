#!/usr/bin/env python3
"""OKF v0.2 conformance checker and migrator for an llm-wiki bundle.

The wiki this skill builds is *shaped* like an Open Knowledge Format bundle
but is not conformant out of the box: ``SCHEMA.md`` and the ``raw/`` sources
carry no ``type``, ``log.md`` headings are ``## [DATE] action | subject``
rather than ISO date headings, and ``sources:`` is a list of bare paths where
OKF requires a ``resource`` per entry.  This script reports those gaps and,
on request, closes them.

Subcommands
-----------
  check WIKI     report OKF v0.2 conformance; exit 1 on a hard failure
  migrate WIKI   bring a Karpathy-style wiki to OKF v0.2 (the opt-in)
  upgrade WIKI   bring an existing OKF v0.1 bundle to v0.2

Every subcommand is a dry run until ``--write`` is passed, and every one is
idempotent: running it twice changes nothing the second time.  Edits are
applied as targeted text surgery on the frontmatter block, never as a YAML
reserialize, so comments, key order, and formatting survive untouched.

Spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml ships with hermes
    sys.exit("okf.py requires pyyaml (pip install pyyaml)")

OKF_VERSION = "0.2"
RESERVED = {"index.md", "log.md"}
TYPE_MAP = {
    "entity": "Entity",
    "concept": "Concept",
    "comparison": "Comparison",
    "query": "Query",
    "summary": "Summary",
}
SKIP_DIRS = {".git", ".obsidian", "_archive", "node_modules"}
FM = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?", re.DOTALL)
LOG_ENTRY = re.compile(r"^##\s*\[(\d{4}-\d{2}-\d{2})\]\s*(\w+)\s*\|\s*(.*?)\s*$")
ISO_DATE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")


@dataclass
class Report:
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "okf_version": OKF_VERSION,
            "conformant": not self.hard,
            "hard_failures": self.hard,
            "soft_issues": self.soft,
            "changes": self.changes,
        }


# ---------------------------------------------------------------------------
# frontmatter surgery
# ---------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return (frontmatter_text, body). frontmatter_text is None when absent."""
    m = FM.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end() :]


def parse_frontmatter(fm_text: str | None) -> dict | None:
    if fm_text is None:
        return None
    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def rebuild(fm_text: str, body: str) -> str:
    return f"---\n{fm_text}\n---\n{body}"


def insert_key(fm_text: str, key: str, value: str) -> str:
    """Insert ``key: value`` at the top of a frontmatter block."""
    return f"{key}: {value}\n{fm_text}" if fm_text.strip() else f"{key}: {value}"


def replace_scalar(fm_text: str, key: str, new_line: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:[^\n]*$", re.MULTILINE)
    return pattern.sub(lambda _m: new_line, fm_text, count=1)


def md_files(wiki: Path):
    for path in sorted(wiki.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.relative_to(wiki).parts):
            continue
        yield path


def concept_files(wiki: Path):
    for path in md_files(wiki):
        if path.name not in RESERVED:
            yield path


def default_type(wiki: Path, path: Path) -> str:
    rel = path.relative_to(wiki)
    if rel.parts[0] == "raw":
        return "Source"
    if path.name == "SCHEMA.md":
        return "Reference"
    return TYPE_MAP.get(rel.parts[0].rstrip("s").lower(), "Concept")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def opted_in(wiki: Path) -> bool:
    index = wiki / "index.md"
    if not index.is_file():
        return False
    fm = parse_frontmatter(split_frontmatter(index.read_text(encoding="utf-8"))[0])
    return bool(fm and fm.get("okf_version"))


def check(wiki: Path) -> Report:
    rep = Report()
    for path in concept_files(wiki):
        rel = path.relative_to(wiki)
        fm_text, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        if fm_text is None:
            rep.hard.append(f"{rel}: no YAML frontmatter block (§11.1)")
            continue
        fm = parse_frontmatter(fm_text)
        if fm is None:
            rep.hard.append(f"{rel}: frontmatter does not parse as a YAML mapping (§11.1)")
            continue
        if not str(fm.get("type") or "").strip():
            rep.hard.append(f"{rel}: missing non-empty 'type' (§11.2)")
        for entry in fm.get("sources") or []:
            if isinstance(entry, str):
                rep.soft.append(f"{rel}: sources entry {entry!r} has no 'resource' key (§5.1)")
        if "timestamp" in fm and "generated" not in fm:
            rep.soft.append(f"{rel}: legacy 'timestamp'; v0.2 uses generated.at (§13.1)")

    for path in md_files(wiki):
        if path.name != "index.md":
            continue
        rel = path.relative_to(wiki)
        fm = parse_frontmatter(split_frontmatter(path.read_text(encoding="utf-8"))[0])
        if fm is None:
            continue
        extra = set(fm) - {"okf_version"}
        if extra or path.parent != wiki:
            rep.hard.append(f"{rel}: index.md carries frontmatter beyond okf_version (§8)")

    log = wiki / "log.md"
    if log.is_file():
        legacy = [
            line
            for line in log.read_text(encoding="utf-8").splitlines()
            if LOG_ENTRY.match(line)
        ]
        if legacy:
            rep.hard.append(
                f"log.md: {len(legacy)} heading(s) are not ISO date headings (§9)"
            )
    return rep


# ---------------------------------------------------------------------------
# migrate: karpathy wiki -> OKF v0.2
# ---------------------------------------------------------------------------


def migrate_log(text: str) -> str:
    """Rewrite ``## [DATE] action | subject`` into date-grouped §9 sections.

    Legacy entries become one bullet per action with their detail lines nested
    beneath, newest-first order preserved. Already-migrated logs are returned
    byte-identical, so the rewrite is a one-pass fixpoint.
    """
    head: list[str] = []
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    current: str | None = None
    legacy = False

    for line in text.splitlines():
        m = LOG_ENTRY.match(line)
        iso = ISO_DATE.match(line)
        if m or iso:
            date = (m or iso).group(1)  # type: ignore[union-attr]
            if date not in groups:
                groups[date] = []
                order.append(date)
            current = date
            if m:
                legacy = True
                action, subject = m.group(2), m.group(3)
                bullet = f"* **{action.capitalize()}**"
                groups[date].append(f"{bullet}: {subject}" if subject else bullet)
            continue
        if current is None:
            head.append(line)
        elif legacy and line.lstrip().startswith(("-", "*")) and groups[current]:
            groups[current].append(f"  {line.lstrip()}")
        else:
            groups[current].append(line)

    if not order or not legacy:
        return text

    out = "\n".join(head).rstrip()
    for date in order:
        body = "\n".join(groups[date]).strip("\n")
        out += f"\n\n## {date}\n{body}" if body else f"\n\n## {date}"
    return out.lstrip("\n") + "\n"


def migrate(wiki: Path, rep: Report) -> dict[Path, str]:
    edits: dict[Path, str] = {}

    for path in concept_files(wiki):
        rel = path.relative_to(wiki)
        original = path.read_text(encoding="utf-8")
        raw_fm, body = split_frontmatter(original)
        created = raw_fm is None
        fm_text = raw_fm or ""
        fm = parse_frontmatter(fm_text) or {}

        new_fm = fm_text
        current = str(fm.get("type") or "").strip()
        if not current:
            new_fm = insert_key(new_fm, "type", default_type(wiki, path))
            rep.changes.append(f"{rel}: + type: {default_type(wiki, path)}")
        elif current in TYPE_MAP:
            new_fm = replace_scalar(new_fm, "type", f"type: {TYPE_MAP[current]}")
            rep.changes.append(f"{rel}: type {current} -> {TYPE_MAP[current]}")

        sources = fm.get("sources")
        if isinstance(sources, list) and any(isinstance(s, str) for s in sources):
            rendered = ["sources:"]
            for entry in sources:
                if isinstance(entry, str):
                    ref = entry if entry.startswith(("/", "http")) else f"/{entry}"
                    rendered.append(f"  - id: {Path(entry).stem}")
                    rendered.append(f"    resource: {ref}")
                elif isinstance(entry, dict):
                    rendered.append(f"  - {json.dumps(entry, default=str)}")
            new_fm = re.sub(
                r"^sources:[^\n]*(?:\n[ \t]+[^\n]*)*$",
                lambda _m, block="\n".join(rendered): block,
                new_fm,
                count=1,
                flags=re.MULTILINE,
            )
            rep.changes.append(f"{rel}: sources -> OKF entries with 'resource'")

        updated = rebuild(new_fm, body) if (new_fm != fm_text or created) else original
        if updated != original:
            edits[path] = updated

    index = wiki / "index.md"
    if index.is_file():
        original = index.read_text(encoding="utf-8")
        raw_fm, body = split_frontmatter(original)
        fm = parse_frontmatter(raw_fm) or {}
        if str(fm.get("okf_version") or "") != OKF_VERSION:
            declared = f'okf_version: "{OKF_VERSION}"'
            new_fm = (
                replace_scalar(raw_fm, "okf_version", declared)
                if raw_fm is not None and "okf_version" in fm
                else insert_key(raw_fm or "", "okf_version", f'"{OKF_VERSION}"')
            )
            edits[index] = rebuild(new_fm, body)
            rep.changes.append(f"index.md: {declared}")

    log = wiki / "log.md"
    if log.is_file():
        original = log.read_text(encoding="utf-8")
        migrated = migrate_log(original)
        if migrated != original:
            edits[log] = migrated
            rep.changes.append("log.md: headings -> ISO date groups (§9)")
    return edits


# ---------------------------------------------------------------------------
# upgrade: OKF v0.1 bundle -> v0.2
# ---------------------------------------------------------------------------


def extract_citations(body: str) -> tuple[list[str], str]:
    m = re.search(r"^#+\s*Citations\s*$", body, re.MULTILINE)
    if not m:
        return [], body
    rest = body[m.end() :]
    end = re.search(r"^#{1,6}\s+\S", rest, re.MULTILINE)
    block = rest[: end.start()] if end else rest
    items = re.findall(r"^\s*[-*]\s+(.*\S)\s*$", block, re.MULTILINE)
    if not items:
        return [], body
    return items, body[: m.start()] + (rest[end.start() :] if end else "")


def citation_entries(items: list[str]) -> list[str]:
    out = ["sources:"]
    for i, raw in enumerate(items, 1):
        link = re.search(r"\[([^\]]+)\]\(([^)]+)\)", raw)
        url = re.search(r"https?://\S+", raw)
        if link:
            title, resource = link.group(1), link.group(2)
        elif url:
            title, resource = raw.replace(url.group(0), "").strip(" -—:"), url.group(0)
        else:
            title, resource = raw, raw
        out.append(f"  - id: source-{i}")
        out.append(f"    resource: {resource}")
        if title and title != resource:
            out.append(f"    title: {title}")
    return out


def upgrade(wiki: Path, rep: Report) -> dict[Path, str]:
    edits: dict[Path, str] = {}
    for path in concept_files(wiki):
        rel = path.relative_to(wiki)
        original = path.read_text(encoding="utf-8")
        raw_fm, body = split_frontmatter(original)
        if raw_fm is None:
            continue
        fm = parse_frontmatter(raw_fm)
        if fm is None:
            continue
        new_fm, new_body = raw_fm, body

        if "timestamp" in fm and "generated" not in fm:
            actor = str(fm.get("generated_by") or "human:unknown")
            # Read the raw scalar, not fm["timestamp"]: PyYAML resolves an ISO
            # 8601 timestamp into a datetime whose str() is not ISO 8601.
            raw_ts = re.search(r"^timestamp:[ \t]*(.+?)[ \t]*$", new_fm, re.MULTILINE)
            stamp = raw_ts.group(1).strip("\"'") if raw_ts else str(fm["timestamp"])
            new_fm = replace_scalar(
                new_fm, "timestamp", f"generated: {{ by: {actor}, at: {stamp} }}"
            )
            rep.changes.append(f"{rel}: timestamp -> generated.at (§13.1)")

        if "sources" not in fm:
            items, stripped = extract_citations(new_body)
            if items:
                new_body = stripped
                new_fm = new_fm.rstrip("\n") + "\n" + "\n".join(citation_entries(items))
                rep.changes.append(f"{rel}: # Citations -> sources ({len(items)}) (§13.1)")

        if new_fm != raw_fm or new_body != body:
            edits[path] = rebuild(new_fm, new_body)

    index = wiki / "index.md"
    if index.is_file():
        original = index.read_text(encoding="utf-8")
        raw_fm, body = split_frontmatter(original)
        fm = parse_frontmatter(raw_fm) or {}
        if raw_fm is not None and fm.get("okf_version") and str(fm["okf_version"]) != OKF_VERSION:
            edits[index] = rebuild(
                replace_scalar(raw_fm, "okf_version", f'okf_version: "{OKF_VERSION}"'),
                body,
            )
            rep.changes.append(f'index.md: okf_version -> "{OKF_VERSION}"')
    return edits


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def apply(edits: dict[Path, str], write: bool) -> None:
    if not write:
        return
    for path, text in edits.items():
        path.write_text(text, encoding="utf-8")


def emit(rep: Report, as_json: bool, write: bool, wiki: Path) -> int:
    if as_json:
        print(json.dumps(rep.as_dict(), indent=2))
        return 1 if rep.hard else 0

    if rep.changes:
        verb = "applied" if write else "would apply"
        print(f"{len(rep.changes)} change(s) {verb} in {wiki}:")
        for line in rep.changes:
            print(f"  {line}")
        if not write:
            print("\nDry run. Re-run with --write to apply.")
    if rep.hard:
        print(f"\nHARD failures ({len(rep.hard)}) — bundle is NOT OKF {OKF_VERSION} conformant:")
        for line in rep.hard:
            print(f"  {line}")
    if rep.soft:
        print(f"\nSoft issues ({len(rep.soft)}) — conformant, but recommended to fix:")
        for line in rep.soft:
            print(f"  {line}")
    if not rep.changes and not rep.hard and not rep.soft:
        print(f"{wiki}: OKF {OKF_VERSION} conformant, nothing to do.")
    return 1 if rep.hard else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OKF v0.2 checker/migrator for an llm-wiki bundle.")
    parser.add_argument("command", choices=("check", "migrate", "upgrade"))
    parser.add_argument("wiki", type=Path)
    parser.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    wiki: Path = args.wiki.expanduser()
    if not wiki.is_dir():
        print(f"not a directory: {wiki}", file=sys.stderr)
        return 2

    rep = Report()
    if args.command == "check":
        rep = check(wiki)
    else:
        runner = migrate if args.command == "migrate" else upgrade
        apply(runner(wiki, rep), args.write)
        if args.write:
            post = check(wiki)
            rep.hard, rep.soft = post.hard, post.soft
    return emit(rep, args.as_json, args.write, wiki)


if __name__ == "__main__":
    raise SystemExit(main())
