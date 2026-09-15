#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lazy-modules splitter — turn an oversized aggregated context file into groups.

Input: the user's AGENTS.md (or SOUL.md / .cursorrules-style aggregated prompt file).
Two structure modes, auto-detected:

  * explicit markers — lines of the form ``<space>module: <name>.md`` split the file
    into named modules (the shape #110868 was authored against);
  * headings — each top-level ``## `` section becomes one module (title = filename
    slug), so an unmodified AGENTS.md can be split with zero prep.

Output into the plugin data dir (default ``<HERMES_HOME>/lazy-modules/``):

  modules/<name>.md   one file per module
  manifest.json       groups: name, always(bool), triggers(list[str], substring-
                      matched case-insensitively against the user message), modules
  <source>.slim       head + always-groups only — the drop-in replacement for the
                      original context file (static layer stays small; the rest is
                      injected on demand by the plugin's pre_llm_call hook)

An optional ``groups.json`` next to the data dir overrides the auto grouping:
``[{"name": ..., "always": bool, "triggers": [...], "modules": [...]}, ...]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MODULE_RE = re.compile(r"(?m)^ module: (.+?\.md)\s*$")
H2_RE = re.compile(r"(?m)^## +(.+?)\s*$")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "when",
    "use",
    "using",
    "this",
    "that",
    "are",
    "is",
    "be",
    "from",
    "by",
    "as",
    "at",
}


def _slug(title: str) -> str:
    words = re.findall(r"[\w\-]+", title.lower(), re.UNICODE)[:6]
    return "-".join(words)[:60] or "section"


def _title_triggers(title: str) -> "list[str]":
    raw = re.split(r"[\s/\-–—::,()\[\]]+", title.lower())
    out: "list[str]" = []

    def add(w: str) -> None:
        if w and w not in _STOP and w not in out:
            out.append(w)

    for w in raw:
        if not w:
            continue
        if re.search(r"[\u4e00-\u9fff]", w):
            add(w)  # whole CJK token first
            if len(w) >= 4:  # then prefix/suffix bigrams: 部署规范 -> 部署, 规范
                add(w[:2])
                add(w[-2:])
        elif len(w) >= 2:
            add(w)
        if len(out) >= 6:
            break
    return out[:6]


def split_markers(text: str) -> "dict[str, str]":
    """Explicit ``module:`` marker format: head is discarded from modules, each
    marker line starts a module body that runs to the next marker."""
    parts = MODULE_RE.split(text)
    mods: "dict[str, str]" = {}
    for i in range(1, len(parts), 2):
        mods[parts[i].strip()] = parts[i + 1]
    return mods


def split_headings(text: str) -> "dict[str, str]":
    """Heading mode: preamble + each ``## `` section. Section name is de-duped."""
    matches = list(H2_RE.finditer(text))
    mods: "dict[str, str]" = {}
    used: "dict[str, int]" = {}
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        slug = _slug(m.group(1))
        n = used.get(slug, 0)
        used[slug] = n + 1
        name = f"{slug}.md" if n == 0 else f"{slug}-{n + 1}.md"
        mods[name] = text[start:end]
    return mods


def auto_groups(mods: "dict[str, str]") -> "list[dict]":
    """One group per module, triggers from the module's own title/heading."""
    groups = []
    for name in mods:
        stem = name[:-3] if name.endswith(".md") else name
        title = mods[name].lstrip().splitlines()[0] if mods[name].strip() else stem
        title = re.sub(r"^#+\s*", "", title)[:80]
        groups.append({
            "name": stem,
            "always": False,
            "triggers": _title_triggers(title) or [stem.replace("-", " ")],
            "modules": [name],
        })
    return groups


def build(source: Path, root: Path) -> int:
    text = source.read_text(encoding="utf-8", errors="replace")
    mods = split_markers(text)
    head_text = MODULE_RE.split(text, maxsplit=1)[0] if mods else ""
    if not mods:
        mods = split_headings(text)
        head_text = H2_RE.split(text, maxsplit=1)[0] if H2_RE.search(text) else ""
    if not mods:
        print(
            f"ERROR: {source} has neither ' module: <name>.md' markers nor '## ' headings",
            file=sys.stderr,
        )
        return 1

    groups_override = root / "groups.json"
    if groups_override.is_file():
        try:
            groups = json.loads(groups_override.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"ERROR: groups.json unreadable ({exc}); refusing to guess",
                file=sys.stderr,
            )
            return 1
    else:
        groups = auto_groups(mods)

    known = {m for g in groups for m in g.get("modules", [])}
    for name in mods:  # modules not claimed by groups.json still loadable
        if name not in known:
            groups.extend(auto_groups({name: mods[name]}))

    mdir = root / "modules"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, body in mods.items():
        (mdir / name).write_text(body.strip() + "\n", encoding="utf-8")

    manifest = {
        "version": 1,
        "source": str(source),
        "groups": [
            {
                "name": g["name"],
                "always": bool(g.get("always")),
                "triggers": list(g.get("triggers", [])),
                "modules": list(g["modules"]),
            }
            for g in groups
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    slim = source.with_suffix(source.suffix + ".slim")
    always_bodies = [
        mods[m]
        for g in manifest["groups"]
        if g["always"]
        for m in g["modules"]
        if m in mods
    ]
    slim.write_text(
        head_text + "\n\n".join(b.strip() + "\n" for b in always_bodies),
        encoding="utf-8",
    )

    total = sum(len(b) for b in mods.values())
    static = len(head_text) + sum(len(b) for b in always_bodies)
    print(f"OK: {len(mods)} modules -> {mdir}")
    print(f"    {len(manifest['groups'])} groups -> {root / 'manifest.json'}")
    print(f"    slim static layer -> {slim}")
    print(
        f"    chars: total={total} static={static} lazy={total - static} "
        f"({100 * (total - static) // max(total, 1)}% on demand)"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lazy-modules-split")
    ap.add_argument("--source", required=True)
    ap.add_argument("--root", required=True)
    ns = ap.parse_args(argv)
    return build(Path(ns.source), Path(ns.root))


if __name__ == "__main__":
    sys.exit(main())
