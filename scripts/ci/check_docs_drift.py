#!/usr/bin/env python3
"""Docs drift ratchet: every DEFAULT_CONFIG key and slash command must be documented.

A config key counts as documented when the docs mention its dotted path (``agent.max_turns``),
a placeholder path covers it (``auxiliary.<task>.timeout``), or a YAML example in the docs nests
it under the right parents. A slash command counts when ``/name`` appears in
``website/docs/reference/slash-commands.md``.

What was undocumented when the ratchet landed lives in ``docs_drift_baseline.json``. Only NEW
gaps fail; documenting a baselined item or deleting it never breaks anything (``--update``
prunes the baseline). Pure stdlib + repo imports, no network, deterministic.

    python scripts/ci/check_docs_drift.py            # exit 1 on new undocumented items
    python scripts/ci/check_docs_drift.py --update   # rewrite the baseline to current gaps
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "website" / "docs"
SLASH_PAGE = DOCS / "reference" / "slash-commands.md"
BASELINE = Path(__file__).with_name("docs_drift_baseline.json")
_FENCE = re.compile(r"```ya?ml[^\n]*\n(.*?)```", re.DOTALL)
_DOTTED = re.compile(r"(?<![\w.])[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+")
_PLACEHOLDER = re.compile(r"(?<![\w.])((?:[a-z_][\w-]*\.)+(?:<[\w-]+>|\*)(?:\.(?:[a-z_][\w-]*|<[\w-]+>|\*))*)")
# Globs only count inside inline code (`auxiliary.*.provider`): bare `*` in prose is emphasis.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def _docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8-sig") for p in sorted(DOCS.rglob("*.md*")))


def config_leaves() -> list[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    out: list[str] = []

    def walk(d: dict, pre: tuple[str, ...]) -> None:
        for k, v in d.items():
            p = (*pre, str(k))
            if str(k).startswith("_"):
                continue  # internal bookkeeping (_config_version), never user-facing
            if isinstance(v, dict) and v:
                walk(v, p)
            else:
                out.append(".".join(p))

    walk(DEFAULT_CONFIG, ())
    return sorted(out)


def slash_commands() -> list[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hermes_cli.commands import COMMAND_REGISTRY

    return sorted(c.name for c in COMMAND_REGISTRY)


def _yaml_paths(text: str) -> set[str]:
    import warnings

    from hermes_yaml import safe_load
    from ruamel.yaml.error import YAMLError

    paths: set[str] = set()

    def flat(d: dict, pre: tuple[str, ...]) -> None:
        for k, v in d.items():
            p = (*pre, str(k))
            paths.add(".".join(p))
            if isinstance(v, dict):
                flat(v, p)

    for m in _FENCE.finditer(text):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ruamel's YAML 1.1 float notes on doc snippets
            try:
                doc = safe_load(m.group(1))
            except YAMLError:  # a partial or templated snippet documents nothing reliably
                continue
        if isinstance(doc, dict):
            flat(doc, ())
    return paths


def _placeholder_patterns(text: str) -> list[re.Pattern[str]]:
    """``a.<task>.b`` / ``a.*.b`` cover one segment; a trailing ``.*`` covers every leaf below."""
    pats = set()
    spans = [m.group(1) for m in _CODE_SPAN.finditer(text)]
    for m in (p for s in spans for p in _PLACEHOLDER.finditer(s)):
        segs = m.group(1).split(".")
        tail = r"(?:\..+)?" if segs[-1] == "*" else ""
        if tail:
            segs = segs[:-1]
        parts = [r"[^.]+" if re.fullmatch(r"<[\w-]+>|\*", s) else re.escape(s) for s in segs]
        # `agent.*` in prose names a section, it doesn't document every key under it: a
        # trailing glob needs two segments ahead of it (`auxiliary.<task>.*`, `display.pet.*`).
        if len(parts) >= 2:
            pats.add(r"\.".join(parts) + tail)
    return [re.compile(p + r"$") for p in sorted(pats)]


def _dotted_mentions(text: str) -> set[str]:
    """Every dotted path written in the docs, plus each of its parent prefixes
    (``a.b.c`` documents ``a.b`` too). One pass instead of a regex scan per key."""
    seen: set[str] = set()
    for m in _DOTTED.finditer(text):
        parts = m.group(0).rstrip(".").split(".")
        for i in range(2, len(parts) + 1):
            seen.add(".".join(parts[:i]))
    return seen


def find_gaps() -> dict[str, list[str]]:
    text = _docs_text()
    mentions = _dotted_mentions(text) | _yaml_paths(text)
    words = set(re.findall(r"(?<![\w.])[A-Za-z_][\w-]*(?![\w.])", text))  # top-level single-word keys
    placeholders = _placeholder_patterns(text)

    def documented(dotted: str) -> bool:
        if dotted in mentions or ("." not in dotted and dotted in words):
            return True
        return any(p.match(dotted) for p in placeholders)

    slash = SLASH_PAGE.read_text(encoding="utf-8-sig")
    return {
        "config": [k for k in config_leaves() if not documented(k)],
        "slash_commands": [c for c in slash_commands()
                           if not re.search(r"`/" + re.escape(c) + r"(?![\w-])", slash)],
    }


def _load_baseline() -> dict[str, list[str]]:
    if not BASELINE.exists():
        return {"config": [], "slash_commands": []}
    return json.loads(BASELINE.read_text(encoding="utf-8-sig"))


def compare(gaps: dict[str, list[str]], baseline: dict[str, list[str]]) -> tuple[dict, dict]:
    new = {k: sorted(set(v) - set(baseline.get(k, []))) for k, v in gaps.items()}
    stale = {k: sorted(set(baseline.get(k, [])) - set(v)) for k, v in gaps.items()}
    return new, stale


def failure_message(new: dict[str, list[str]]) -> str:
    lines = ["Undocumented user-facing surface (docs drift ratchet):"]
    if new["config"]:
        lines.append("  config keys (mention the dotted path, or add a YAML example, in website/docs/ "
                     "-- usually user-guide/configuration.md):")
        lines += [f"    {k}" for k in new["config"]]
    if new["slash_commands"]:
        lines.append("  slash commands (add a row to website/docs/reference/slash-commands.md):")
        lines += [f"    /{c}" for c in new["slash_commands"]]
    lines.append("Deliberately undocumented? `python scripts/ci/check_docs_drift.py --update` adds it to "
                 "the baseline; reviewers see that in the diff.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Docs drift ratchet for config keys and slash commands.")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline to the current gaps")
    args = ap.parse_args(argv)
    gaps = find_gaps()
    if args.update:
        BASELINE.write_text(json.dumps(gaps, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline: {len(gaps['config'])} config keys, {len(gaps['slash_commands'])} slash commands")
        return 0
    new, stale = compare(gaps, _load_baseline())
    for kind, items in stale.items():
        if items:
            print(f"note: {len(items)} baselined {kind} now documented or removed; "
                  "`--update` shrinks the baseline")
    if any(new.values()):
        print(failure_message(new))
        return 1
    print(f"docs drift: ok ({len(gaps['config'])} config keys, {len(gaps['slash_commands'])} "
          "slash commands still baselined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
