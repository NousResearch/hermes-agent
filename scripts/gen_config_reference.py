#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/_hermes-python" "$0" "$@"'
"""Render the config reference from code — port of the gateway-contracts codegen pattern.

Single source of truth: ``DEFAULT_CONFIG`` (hermes_cli/config_defaults.py) unified with the
known-key lists in hermes_cli/config.py and the chat-CLI overlay defaults. The committed
artifact at docs/reference/config-reference.generated.md is diffed byte-for-byte by
tests/hermes_cli/test_config_reference_generated.py, so any defaults change must land in the
same commit as the regenerated artifact.

The chat-CLI defaults (hermes_cli/cli_config_load._cli_config_defaults) legitimately disagree
with DEFAULT_CONFIG on some keys (agent.max_turns 500 vs None=unlimited,
delegation.max_iterations 45 vs 250): the CLI overlay is applied before the file config, so
both are real defaults. The renderer records both under a "cli:" note instead of silently
picking one side — the generator refuses to codify half the split.

Also derived here (one scan, shared by the artifact and the drift test):
- Reader coverage: every ``<section>_cfg.get("key")`` / chained ``.get("section").get("key")``
  site under the agent-facing trees must name a key some defaults dict owns, or belong to a
  documented open-dict section. Keys owned by neither land in the artifact's "Unset reader
  keys" appendix — visible drift, not a test failure, so an in-progress reader merges first.
- The counts docs hand-write: providers (plugins/model-providers dirs), toolsets (TOOLSETS).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli.cli_config_load import _cli_config_defaults  # noqa: E402
from hermes_cli.config import (  # noqa: E402
    _EXTRA_KNOWN_ROOT_KEYS,
    _OPEN_SUBKEY_TOP_LEVEL_KEYS,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG  # noqa: E402
from toolsets import TOOLSETS  # noqa: E402

OUT = ROOT / "docs" / "reference" / "config-reference.generated.md"

# Trees scanned for config readers. Deliberately the agent-facing trees: plugins/platforms
# adapters read their own protocol payloads, not config.yaml sections.
READER_SCAN_DIRS = ("hermes_cli", "agent", "gateway", "tools", "tui_gateway", "cron", "acp_adapter")

# <section>_cfg.get("key") where the variable is named after its config section.
_SECTION_RE = re.compile(r"\b([a-z][a-z0-9_]*)_cfg\s*\.\s*get\(\s*[\"']([a-z_][a-z0-9_]*)[\"']")
# Chained section-then-key reads: config.get("section", ...).get("key", ...).
_CHAINED_RE = re.compile(
    r"\.get\(\s*[\"']([a-z_][a-z0-9_]*)[\"']\s*(?:,[^)]*)?\)\s*\.get\(\s*[\"']([a-z_][a-z0-9_]*)[\"']"
)


def _cli_defaults() -> dict[str, dict]:
    return _cli_config_defaults()


def _sections() -> list[dict]:
    """Deterministic section records: DEFAULT_CONFIG ∪ known root keys ∪ CLI overlay sections."""
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)

    for name in DEFAULT_CONFIG:
        if name != "_config_version":
            add(name)
    for name in sorted(_EXTRA_KNOWN_ROOT_KEYS):
        add(name)
    for name in sorted(_OPEN_SUBKEY_TOP_LEVEL_KEYS):
        add(name)
    for name in sorted(_cli_defaults()):
        add(name)

    sections = []
    for name in names:
        core = DEFAULT_CONFIG.get(name)
        cli = _cli_defaults().get(name)
        sections.append(
            {
                "name": name,
                "core": core if isinstance(core, dict) else None,
                "cli": cli if isinstance(cli, dict) else None,
                "core_keys": set(core) if isinstance(core, dict) else set(),
                "cli_keys": set(cli) if isinstance(cli, dict) else set(),
                "open_dict": name in _OPEN_SUBKEY_TOP_LEVEL_KEYS,
                "extra": name in _EXTRA_KNOWN_ROOT_KEYS and name not in DEFAULT_CONFIG,
                "cli_only": name not in DEFAULT_CONFIG and isinstance(cli, dict),
                "scalar_core": isinstance(core, str),
            }
        )
    return sections


def _scan_readers(sections: list[dict]) -> tuple[dict[str, set[str]], set[str], dict[str, set[str]]]:
    """Classify every section-scoped reader site under READER_SCAN_DIRS.

    Returns (owned key -> reader files, undocumented open-dict keys, unset key -> reader files).
    """
    section_names = {s["name"] for s in sections}
    owned: dict[str, set[str]] = {
        f"{s['name']}.{key}": set() for s in sections for key in s["core_keys"] | s["cli_keys"]
    }
    open_undoc: set[str] = set()
    unset: dict[str, set[str]] = {}
    for d in READER_SCAN_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [m.groups() for m in _SECTION_RE.finditer(text)]
            hits += [m.groups() for m in _CHAINED_RE.finditer(text)]
            rel = path.relative_to(ROOT).as_posix()
            for sec, key in hits:
                if sec not in section_names:
                    continue
                dotted = f"{sec}.{key}"
                if dotted in owned:
                    owned[dotted].add(rel)
                elif sec in _OPEN_SUBKEY_TOP_LEVEL_KEYS:
                    open_undoc.add(dotted)
                else:
                    unset.setdefault(dotted, set()).add(rel)
    return owned, open_undoc, unset


def _fmt(v: object) -> str:
    if isinstance(v, str):
        return f'`"{v}"`' if v else '`""` (empty)'
    if v is None:
        return "`null` (unlimited)"
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return "`[]` (empty list)" if not v else f"`{json.dumps(v)}`"
    return f"`{json.dumps(v)}`"


def _rows_for(section: dict) -> list[tuple[str, str, str]]:
    """(key, default, note) rows; dict-valued leaves flatten one level (agent_cache.max_size)."""
    core, cli = section["core"], section["cli"]
    rows: list[tuple[str, str, str]] = []
    for key in sorted(section["core_keys"] | section["cli_keys"]):
        cval = core.get(key) if core and key in core else _MISSING
        clival = cli.get(key) if cli and key in cli else _MISSING
        if isinstance(cval, dict) and cval:
            for sub in sorted(cval):
                rows.append((f"{key}.{sub}", _fmt(cval[sub]), ""))
            continue
        default = _fmt(cval) if cval is not _MISSING else "—"
        note = ""
        if clival is not _MISSING:
            cv = _fmt(clival)
            if cval is not _MISSING and cv != _fmt(cval):
                note = f"cli: {cv}"
            elif cval is _MISSING:
                default, note = cv, "cli only"
        rows.append((key, default, note))
    return rows


_MISSING = object()


def render(sections: list[dict], unset: dict[str, set[str]], open_undoc: set[str]) -> str:
    lines: list[str] = []
    w = lines.append
    w("<!-- generated by scripts/gen_config_reference.py — DO NOT EDIT; run the script to regenerate -->")
    w("")
    w("# Configuration reference (generated)")
    w("")
    w("Sources: `hermes_cli/config_defaults.py::DEFAULT_CONFIG` ∪ known root keys")
    w("(`hermes_cli/config.py`) ∪ chat-CLI overlay defaults (`hermes_cli/cli_config_load`).")
    w("`cli:` notes mark keys whose chat-CLI default differs from the core default — both are")
    w("real: the overlay applies before `config.yaml`.")
    w("")
    n_keys = sum(len(s["core_keys"] | s["cli_keys"]) for s in sections)
    n_providers = len([p for p in (ROOT / "plugins" / "model-providers").iterdir() if p.is_dir()])
    w(f"{len(sections)} sections · {n_keys} leaf keys · {n_providers} provider plugins · {len(TOOLSETS)} toolsets")
    w("")

    for s in sections:
        w(f"## `{s['name']}`")
        if s["open_dict"]:
            w("")
            w("*Open dict — user-defined sub-keys are valid; config validation checks the first segment only.*")
        if s["extra"]:
            w("")
            w("*Known key with no core defaults (written by setup/tools flows).*")
        if s["scalar_core"]:
            w("")
            w("*Core `DEFAULT_CONFIG` value is a scalar (legacy string shorthand); the leaf keys below")
            w("come from the chat-CLI overlay.*")
        elif s["cli_only"]:
            w("")
            w("*Chat-CLI only — not part of the core `DEFAULT_CONFIG`.*")
        rows = _rows_for(s)
        if not rows:
            w("")
            continue
        w("")
        w("| key | default | notes |")
        w("|---|---|---|")
        for key, default, note in rows:
            w(f"| `{key}` | {default} | {note} |")
        w("")

    w("## Unset reader keys")
    w("")
    w("Section-scoped reader keys (`.get(...)` call sites in the agent-facing trees) that no")
    w("defaults dict owns. Each should land as a documented default with its change — this is")
    w("the gap class that shipped `agent.reasoning_effort` before any default existed.")
    w("")
    if unset:
        w("| key | readers |")
        w("|---|---|")
        for dotted, readers in sorted(unset.items()):
            w(f"| `{dotted}` | {', '.join(f'`{r}`' for r in sorted(readers))} |")
    else:
        w("(none)")
    w("")
    open_list = ", ".join(f"`{k}`" for k in sorted(open_undoc)) or "(none)"
    w(f"Open-dict sections with reader sites but no enumerated defaults: {open_list}")
    w("")
    return "\n".join(lines)


def render_all() -> dict[Path, str]:
    sections = _sections()
    _owned, open_undoc, unset = _scan_readers(sections)
    return {OUT: render(sections, unset, open_undoc)}


def main(argv: list[str] | None = None) -> int:
    for path, text in render_all().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
