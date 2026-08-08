#!/usr/bin/env python3
"""Idempotently merge extracted entries into People/Project entity files.

Reads a batch of structured entries for ONE source note (JSON) and writes them
into per-entity markdown files inside managed-region fences. Only content
between the fences is ever rewritten; anything a human types outside them is
preserved byte-for-byte. Entries are addressed by a claim-keyed stable id, so
rewording a fact does not duplicate it and re-running is convergent.

Usage (invoke through the `terminal` tool):
  python upsert_entry.py --vault "/path/to/vault" \
      --source-path "/path/to/note.md" --source-link "note" --source-sha SHA \
      --entries-json '[{entry}, ...]'      # or --entries-file PATH, or stdin

Entry shape:
  {"entity": {"kind": "person|project", "name": "...", "aliases": ["..."]},
   "section": "facts|commitments|topics|ideas|decisions|blockers|todos",
   "claim": {"subject": "...", "predicate": "...", "object": "..."},
   "text": "human-readable bullet", "op": "assert|retract"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _state import (  # noqa: E402
    PERSON_SECTIONS,
    PROJECT_SECTIONS,
    atomic_write_text,
    entity_id as make_entity_id,
    entry_id as make_entry_id,
    load_state,
    norm_key,
    now_iso,
    nfc,
    save_state,
    slugify,
)

BULLET_ID_RE = re.compile(r"\s\^(nx-[0-9a-f]{8})\s*$")
BULLET_DATE_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
OUTPUT_DIRS = {"person": "People", "project": "Ideas-Projects"}


# --------------------------------------------------------------------------- #
# Frontmatter (minimal, deterministic — we own these files)
# --------------------------------------------------------------------------- #
def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return ({key: rawvalue}, body). Empty dict + full text if no frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fm_block = text[4:end]
    body = text[end + 5:]
    fm: dict = {}
    for line in fm_block.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip()
    return fm, body


def render_frontmatter(fm: dict, kind: str) -> str:
    keys = ["type", "id", "name", "aliases"]
    if kind == "project":
        keys.append("status")
    keys.append("updated")
    lines = ["---"]
    for k in keys:
        if k in fm:
            lines.append(f"{k}: {fm[k]}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def aliases_to_field(aliases: list[str]) -> str:
    return "[" + ", ".join(sorted(set(a for a in aliases if a))) + "]"


def field_to_aliases(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [a.strip() for a in value.split(",") if a.strip()]


# --------------------------------------------------------------------------- #
# Managed-region engine (pure functions on body text)
# --------------------------------------------------------------------------- #
def markers(region: str) -> tuple[str, str]:
    return (f"<!-- notes-extract:begin {region} -->",
            f"<!-- notes-extract:end {region} -->")


def heading_for(section: str) -> str:
    return section.capitalize()


def region_bounds(body: str, region: str) -> tuple[int, int] | None:
    begin, end = markers(region)
    b = body.find(begin)
    if b == -1:
        return None
    e = body.find(end, b)
    if e == -1:
        return None
    return (b, e + len(end))


def read_region_bullets(body: str, region: str) -> list[tuple[str, str]]:
    """Return [(entry_id, full_bullet_line)] inside the region, in order."""
    bounds = region_bounds(body, region)
    if bounds is None:
        return []
    begin, end = markers(region)
    inner = body[bounds[0] + len(begin):body.find(end, bounds[0])]
    out = []
    for line in inner.splitlines():
        m = BULLET_ID_RE.search(line)
        if m and line.lstrip().startswith("-"):
            out.append((m.group(1), line.rstrip()))
    return out


def write_region(body: str, region: str, bullets: list[str], heading: str) -> str:
    """Replace (or create) the region with exactly these bullet lines."""
    begin, end = markers(region)
    block = begin + "\n" + ("\n".join(bullets) + "\n" if bullets else "") + end
    bounds = region_bounds(body, region)
    if bounds is not None:
        return body[:bounds[0]] + block + body[bounds[1]:]
    # No region yet — attach under its heading if present, else append.
    head_re = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    m = head_re.search(body)
    insertion = "\n" + block + "\n"
    if m:
        at = m.end()
        return body[:at] + "\n" + block + "\n" + body[at:]
    sep = "" if body.endswith("\n\n") or body == "" else ("\n" if body.endswith("\n") else "\n\n")
    return body + sep + f"## {heading}\n" + insertion


# --------------------------------------------------------------------------- #
# Entity file resolution (slug + collision)
# --------------------------------------------------------------------------- #
def existing_id(path: Path) -> str | None:
    if not path.exists():
        return None
    fm, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    return fm.get("id")


def resolve_path(vault: Path, kind: str, slug: str, eid: str,
                 claimed: "dict[Path, str] | None" = None) -> tuple[Path, bool]:
    """Return (path, collided). Suffix -2.. if the slug is taken by a different id.

    `claimed` maps paths already assigned in THIS batch → their entity id, so two
    new same-slug entities in one run don't both land on <slug>.md.
    """
    claimed = claimed or {}
    base = vault / OUTPUT_DIRS[kind]

    def taken_by_other(path: Path) -> bool:
        disk = existing_id(path)
        if disk not in (None, eid):
            return True
        return claimed.get(path) not in (None, eid)

    candidate = base / f"{slug}.md"
    if not taken_by_other(candidate):
        return candidate, False
    n = 2
    while True:
        candidate = base / f"{slug}-{n}.md"
        if not taken_by_other(candidate):
            return candidate, True
        n += 1


# --------------------------------------------------------------------------- #
# Core upsert
# --------------------------------------------------------------------------- #
def bullet_line(text: str, link: str, date: str, eid: str) -> str:
    link_part = f" [[{link}]]" if link else ""
    return f"- {text.strip()}{link_part} ({date}) ^{eid}"


def resolve_entity(state: dict, kind: str, name: str, aliases: list[str]) -> str:
    """Find an existing entity by name/alias, else mint a deterministic id."""
    keys = [norm_key(name)] + [norm_key(a) for a in aliases]
    for k in keys:
        if k in state["alias_index"]:
            return state["alias_index"][k]
    return make_entity_id(kind, name)


def upsert_source(vault: Path, source_path: str, source_link: str, source_sha: str,
                  entries: list[dict], state: dict, clock=None) -> dict:
    """Apply all entries from one source; reconcile against prior run. Mutates state."""
    from _state import source_id as make_source_id

    date = now_iso(clock)
    sid = make_source_id(vault, Path(source_path))
    valid_sections = {"person": PERSON_SECTIONS, "project": PROJECT_SECTIONS}

    # New id set for this source, with where each lives.
    new_records: list[dict] = []
    # Group desired edits by target file.
    per_file: dict[Path, dict] = {}

    def file_ctx(kind, name, aliases):
        eid = resolve_entity(state, kind, name, aliases)
        claimed = {p: c["eid"] for p, c in per_file.items()}
        ent = state["entities"].get(eid)
        if ent is None:
            slug = slugify(name)
            path, collided = resolve_path(vault, kind, slug, eid, claimed)
            ent = {"kind": kind, "slug": path.stem, "name": nfc(name),
                   "aliases": sorted(set(aliases))}
            state["entities"][eid] = ent
        else:
            path, collided = resolve_path(vault, kind, ent["slug"], eid, claimed)
            for a in aliases:
                if a and a not in ent["aliases"]:
                    ent["aliases"].append(a)
        for k in [norm_key(name)] + [norm_key(a) for a in aliases]:
            state["alias_index"].setdefault(k, eid)
        if path not in per_file:
            per_file[path] = {"eid": eid, "kind": kind, "ent": ent,
                              "assert": {}, "retract": set(), "collided": collided}
        return eid, path

    report = {"entries": [], "needs_confirm": [], "removed": []}

    for e in entries:
        ent_meta = e.get("entity", {})
        kind = ent_meta.get("kind")
        name = ent_meta.get("name", "").strip()
        section = e.get("section", "")
        if kind not in valid_sections or not name or section not in valid_sections[kind]:
            report["entries"].append({"text": e.get("text", ""), "action": "skipped-invalid"})
            continue
        aliases = [a for a in ent_meta.get("aliases", []) if a]
        eid, path = file_ctx(kind, name, aliases)
        rid = make_entry_id(eid, section, e.get("claim", {}), sid)
        op = e.get("op", "assert")
        if op == "retract":
            per_file[path]["retract"].add((section, rid))
        else:
            per_file[path]["assert"][(section, rid)] = {"text": e.get("text", ""), "link": source_link}
            new_records.append({"id": rid, "entity_id": eid, "section": section})
        if per_file[path]["collided"]:
            report["needs_confirm"].append(str(path))

    # Per-source reconciliation: stale ids (previously from this source, not now).
    prior = state["sources"].get(sid, {}).get("entries", [])
    new_ids = {r["id"] for r in new_records}
    for rec in prior:
        if rec["id"] not in new_ids:
            ent = state["entities"].get(rec["entity_id"])
            if not ent:
                continue
            path = vault / OUTPUT_DIRS[ent["kind"]] / f"{ent['slug']}.md"
            per_file.setdefault(path, {"eid": rec["entity_id"], "kind": ent["kind"],
                                       "ent": ent, "assert": {}, "retract": set(),
                                       "collided": False})
            per_file[path]["retract"].add((rec["section"], rec["id"]))
            report["removed"].append(rec["id"])

    # Apply edits per file, atomically, bumping `updated:` only on real change.
    for path, ctx in per_file.items():
        _apply_file(path, ctx, date, report)

    # Record source state (sha + entries) for next run's diff + reconciliation.
    state["sources"][sid] = {
        "path": str(Path(source_path)),
        "link": source_link,
        "sha": source_sha,
        "entries": new_records,
    }
    return report


def _apply_file(path: Path, ctx: dict, date: str, report: dict) -> None:
    kind, ent, eid = ctx["kind"], ctx["ent"], ctx["eid"]
    sections = PERSON_SECTIONS if kind == "person" else PROJECT_SECTIONS

    if path.exists():
        old_text = path.read_text(encoding="utf-8")
        fm, body = split_frontmatter(old_text)
    else:
        old_text = None
        fm, body = {}, ""

    # Ensure frontmatter scalars.
    fm["type"] = kind
    fm["id"] = eid
    fm["name"] = ent["name"]
    fm["aliases"] = aliases_to_field(ent["aliases"])
    if kind == "project":
        fm.setdefault("status", "active")

    new_body = body
    # Asserts: upsert bullet into its section region. Value: rid -> {text, link}.
    by_section_assert: dict[str, dict[str, dict]] = {}
    for (section, rid), line in ctx["assert"].items():
        by_section_assert.setdefault(section, {})[rid] = line
    # Retracts: move to <section>-archive.
    by_section_retract: dict[str, set[str]] = {}
    for (section, rid) in ctx["retract"]:
        by_section_retract.setdefault(section, set()).add(rid)

    for section in sections:
        asserts = by_section_assert.get(section, {})
        retracts = by_section_retract.get(section, set())
        if not asserts and not retracts:
            continue
        current = read_region_bullets(new_body, section)
        current_map = dict(current)
        order = [rid for rid, _ in current]
        archived = read_region_bullets(new_body, f"{section}-archive")
        archived_map = dict(archived)
        archived_order = [rid for rid, _ in archived]

        for rid in retracts:
            if rid in current_map:
                if rid not in archived_map:
                    archived_order.append(rid)
                archived_map[rid] = current_map[rid]
                del current_map[rid]
                order = [r for r in order if r != rid]
        for rid, info in asserts.items():
            old = current_map.get(rid)
            # Preserve an existing entry's first-seen date so re-running on a
            # later day is a no-op; only brand-new entries get today's date.
            d = date
            if old:
                m = BULLET_DATE_RE.search(old)
                if m:
                    d = m.group(1)
            line = bullet_line(info["text"], info["link"], d, rid)
            if rid not in current_map:
                order.append(rid)
            current_map[rid] = line

        new_body = write_region(new_body, section,
                                [current_map[r] for r in order], heading_for(section))
        if archived_map:
            new_body = write_region(new_body, f"{section}-archive",
                                    [archived_map[r] for r in archived_order],
                                    f"{heading_for(section)} (archived)")

    body_changed = (new_body != body) or (old_text is None)
    if not body_changed:
        report["entries"].append({"file": str(path), "action": "unchanged"})
        return
    fm["updated"] = date
    atomic_write_text(path, render_frontmatter(fm, kind) + new_body)
    report["entries"].append({"file": str(path), "action": "written"})


def _load_entries(args) -> list[dict]:
    if args.entries_json:
        raw = args.entries_json
    elif args.entries_file:
        raw = Path(args.entries_file).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge extracted entries into entity files.")
    p.add_argument("--vault", required=True)
    p.add_argument("--source-path", required=True)
    p.add_argument("--source-link", default="")
    p.add_argument("--source-sha", default="")
    p.add_argument("--entries-json", default="")
    p.add_argument("--entries-file", default="")
    args = p.parse_args(argv)

    vault = Path(args.vault)
    entries = _load_entries(args)
    state = load_state(vault)
    report = upsert_source(vault, args.source_path, args.source_link,
                           args.source_sha, entries, state)
    save_state(vault, state)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
