# Skills Prompt Budget & Nudge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two-tier skill index and signal-based nudge described in `plans/skills-prompt-budget-and-nudge-redesign.md`, both gated behind feature flags, with full test coverage.

**Architecture:** Phase A introduces a new `agent/skill_inventory.py` module that owns parsed skill metadata, refactors `agent/prompt_builder.py` to consume it, and adds a Tier 1/Tier 2 rendering path plus a new `skill_describe` tool — all behind `skills.index_v2`. Phase B adds signal-evaluation state and `_evaluate_skill_nudge_signals()` to `AIAgent`, replaces the time-based end-of-turn gate, and adds a `/skills nudge off|on` intercept — all behind `skills.nudge_signals.enabled`. Both phases are backward compatible; Phase 3 (default flips) ships in a follow-up PR.

**Tech Stack:** Python 3.11 (`pyproject.toml`), `pytest` for unit/integration tests, YAML frontmatter via `pyyaml`, atomic JSON writes via `utils.atomic_json_write`, conventional-commits style.

**Spec reference:** `plans/skills-prompt-budget-and-nudge-redesign.md`

---

## File Structure

### Files to create
- `agent/skill_inventory.py` — typed `SkillEntry` dataclass + `load_inventory()` returning a structured list. Single source of truth for parsed skill metadata; consumed by both `build_skills_system_prompt()` and the new `skill_describe` tool. ~250 lines.
- `agent/skill_usage_tracker.py` — read/write `~/.hermes/.skill_usage.json` and `~/.hermes/.skill_known_clis.json`. Best-effort, never raises. ~120 lines.
- `agent/skill_nudge_signals.py` — `SignalEvaluator` class wrapping the per-tool argument-signature function and S1–S4 evaluation. Pure logic, no I/O. ~200 lines.
- `tests/agent/test_skill_inventory.py` — unit tests for the new inventory module.
- `tests/agent/test_skill_usage_tracker.py` — unit tests for the usage/known-CLI files (corruption, pruning, atomic writes).
- `tests/agent/test_skill_nudge_signals.py` — unit tests for S1–S4.
- `tests/tools/test_skill_describe.py` — unit tests for the new `skill_describe` tool.
- `tests/integration/test_skills_index_v2_e2e.py` — synthetic 100-skill end-to-end test.

### Files to modify
- `agent/prompt_builder.py:466-847` — keep public `build_skills_system_prompt()` signature (additive args only); delegate parsing to `agent/skill_inventory`; add v2 rendering path; bump `_SKILLS_SNAPSHOT_VERSION` to 2; update cache key.
- `tools/skills_tool.py:1388-1445` — add `SKILL_DESCRIBE_SCHEMA` and `registry.register("skill_describe", ...)` next to the existing `skill_view` registration. Add usage-tracking call inside `skill_view()` body.
- `tools/skill_manager_tool.py:172-208` — extend `_validate_frontmatter` to validate `priority ∈ {"critical", "normal"}` if present.
- `run_agent.py:1495-1614` — add nudge state initialization (`_nudge_signals`, `_nudge_disabled`, `_signal_evaluator`, etc.) alongside `_iters_since_skill`.
- `run_agent.py:9295-9299` — call `_evaluate_skill_nudge_signals()` after the counter increment.
- `run_agent.py:12176-12204` — replace the time-only gate with the four-branch gate from spec §5.2.
- `run_agent.py:3014-3060` — add `triggered_signals: set[str] | None` parameter to `_spawn_background_review`; pass into the review prompt.
- `cli.py:5816-5819` — intercept `/skills nudge off|on` before delegating to `handle_skills_slash`.
- `hermes_cli/skills_config.py` — load new `skills.index_v2`, `skills.index_token_budget`, `skills.nudge_signals.*` keys. (If a single config-loader module doesn't exist, the keys are read directly from `_agent_cfg` in run_agent's `__init__`; see Task B1 for the chosen path.)
- `cli-config.yaml.example:481-494` — document new keys per spec §6.

### Files NOT in scope for this PR
- TUI slash-command parity (`ui-tui/src/app/slash/commands/ops.ts`) — out of scope; users on TUI use `HERMES_SKILL_NUDGE_DISABLE=1` env var. Tracked as a Phase 2.5 follow-up in the wrap-up doc.
- Gateway slash-command parity — same reasoning.

---

## Sanity Check Before Starting

- [ ] **Step 0.1: Confirm working directory**

Run: `git -C /Users/zhuosama/.hermes/hermes-agent rev-parse --show-toplevel`
Expected: `/Users/zhuosama/.hermes/hermes-agent`

- [ ] **Step 0.2: Confirm tests run on a clean tree**

Run: `cd /Users/zhuosama/.hermes/hermes-agent && python -m pytest tests/agent/test_prompt_builder.py -q`
Expected: all green. If anything fails on `main` before this work starts, fix or skip those before adding new tests.

- [ ] **Step 0.3: Create a feature branch**

Run:
```bash
cd /Users/zhuosama/.hermes/hermes-agent
git checkout -b feat/skills-index-v2-and-nudge-signals
```

---

# Phase A — Skill Index v2

Gated behind `skills.index_v2: true`. Default `false` until Phase 3.

## Task A1: Extract `SkillInventory` module (refactor only)

**Files:**
- Create: `agent/skill_inventory.py`
- Modify: `agent/prompt_builder.py:537-742`
- Test: `tests/agent/test_skill_inventory.py`

**Goal:** Move the parse-and-cache logic out of `prompt_builder.py` into a dedicated module. Behavior identical to today; this is a pure refactor that creates the seam needed by `skill_describe`.

- [ ] **Step 1.1: Write failing test for `load_inventory()` returning a list of `SkillEntry`**

```python
# tests/agent/test_skill_inventory.py
from pathlib import Path
import textwrap
import pytest

from agent.skill_inventory import SkillEntry, load_inventory


def _write_skill(root: Path, name: str, description: str, category: str = ""):
    skill_dir = root / category / name if category else root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        ---
        body
    """))


def test_load_inventory_returns_skill_entries(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "alpha", "First skill", category="cat1")
    _write_skill(skills_dir, "beta", "Second skill", category="cat1")

    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])

    inv = load_inventory()
    names = sorted(e.name for e in inv.entries)
    assert names == ["alpha", "beta"]
    assert all(e.category == "cat1" for e in inv.entries)
    assert {e.description for e in inv.entries} == {"First skill", "Second skill"}
```

- [ ] **Step 1.2: Run test to verify it fails (module not yet created)**

Run: `python -m pytest tests/agent/test_skill_inventory.py -v`
Expected: `ImportError: cannot import name 'SkillEntry' from 'agent.skill_inventory'`

- [ ] **Step 1.3: Create `agent/skill_inventory.py`**

```python
"""Parsed skill metadata — single source of truth for the index and skill_describe tool.

Extracted from agent/prompt_builder.py.  This module owns the on-disk snapshot,
the in-process LRU, and the platform/disabled/condition filtering.  Renderers
(prompt builder, skill_describe handler) consume the typed entries below.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home, get_skills_dir
from agent.skill_utils import (
    extract_skill_conditions,
    extract_skill_description,
    get_all_skills_dirs,
    get_disabled_skill_names,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_platform,
)
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Bump on any structural change to SkillEntry / snapshot payload — forces re-scan.
SNAPSHOT_VERSION = 2


@dataclass(frozen=True)
class SkillEntry:
    name: str                  # frontmatter `name` (preferred) or directory name
    skill_name: str            # always directory name
    category: str              # path-derived, defaults to "general"
    description: str
    priority: str = "normal"   # "critical" | "normal"
    platforms: tuple[str, ...] = ()
    conditions: dict = field(default_factory=dict)
    source: str = "local"      # "local" | "external"


@dataclass(frozen=True)
class Inventory:
    entries: tuple[SkillEntry, ...]
    category_descriptions: dict[str, str]


_INVENTORY_CACHE: OrderedDict[tuple, Inventory] = OrderedDict()
_INVENTORY_CACHE_MAX = 8
_INVENTORY_LOCK = threading.Lock()


def _snapshot_path() -> Path:
    return get_hermes_home() / ".skills_prompt_snapshot.json"


def _build_manifest(skills_dir: Path) -> dict[str, list[int]]:
    manifest: dict[str, list[int]] = {}
    for filename in ("SKILL.md", "DESCRIPTION.md"):
        for path in iter_skill_index_files(skills_dir, filename):
            try:
                st = path.stat()
            except OSError:
                continue
            manifest[str(path.relative_to(skills_dir))] = [st.st_mtime_ns, st.st_size]
    return manifest


def _load_snapshot(skills_dir: Path) -> Optional[dict]:
    p = _snapshot_path()
    if not p.exists():
        return None
    try:
        snap = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(snap, dict):
        return None
    if snap.get("version") != SNAPSHOT_VERSION:
        return None
    if snap.get("manifest") != _build_manifest(skills_dir):
        return None
    return snap


def _write_snapshot(skills_dir: Path, manifest, entries, category_descriptions) -> None:
    payload = {
        "version": SNAPSHOT_VERSION,
        "manifest": manifest,
        "skills": [e.__dict__ if not isinstance(e, dict) else e for e in entries],
        "category_descriptions": category_descriptions,
    }
    try:
        atomic_json_write(_snapshot_path(), payload)
    except Exception as e:
        logger.debug("Could not write skills snapshot: %s", e)


def _entry_from_skill_file(skill_file: Path, base_dir: Path, source: str) -> Optional[SkillEntry]:
    try:
        raw = skill_file.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(raw)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", skill_file, e)
        return None

    if not skill_matches_platform(frontmatter):
        return None

    rel = skill_file.relative_to(base_dir)
    parts = rel.parts
    if len(parts) >= 2:
        skill_name = parts[-2]
        category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
    else:
        category = "general"
        skill_name = skill_file.parent.name

    platforms = frontmatter.get("platforms") or []
    if isinstance(platforms, str):
        platforms = [platforms]

    priority_raw = frontmatter.get("priority", "normal")
    priority = str(priority_raw).strip().lower() if priority_raw else "normal"
    if priority not in ("critical", "normal"):
        priority = "normal"

    return SkillEntry(
        name=str(frontmatter.get("name", skill_name)),
        skill_name=skill_name,
        category=category,
        description=extract_skill_description(frontmatter),
        priority=priority,
        platforms=tuple(str(p).strip() for p in platforms if str(p).strip()),
        conditions=extract_skill_conditions(frontmatter),
        source=source,
    )


def _read_category_descriptions(base_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for desc_file in iter_skill_index_files(base_dir, "DESCRIPTION.md"):
        try:
            content = desc_file.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(content)
            cat_desc = fm.get("description")
            if not cat_desc:
                continue
            rel = desc_file.relative_to(base_dir)
            cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
            out[cat] = str(cat_desc).strip().strip("'\"")
        except Exception as e:
            logger.debug("Could not read DESCRIPTION.md at %s: %s", desc_file, e)
    return out


def load_inventory(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
) -> Inventory:
    """Return parsed, filtered skill metadata.

    - Local snapshot at ~/.hermes/.skills_prompt_snapshot.json is reused when its
      mtime/size manifest matches.
    - External dirs are always scanned directly (read-only and small).
    - Local entries take precedence on name collision.
    """
    skills_dir = get_skills_dir()
    all_dirs = get_all_skills_dirs()
    external_dirs = all_dirs[1:] if len(all_dirs) > 1 else []

    if not skills_dir.exists() and not external_dirs:
        return Inventory(entries=(), category_descriptions={})

    from gateway.session_context import get_session_env
    platform_hint = (
        os.environ.get("HERMES_PLATFORM")
        or get_session_env("HERMES_SESSION_PLATFORM")
        or ""
    )
    disabled = get_disabled_skill_names()
    cache_key = (
        str(skills_dir.resolve()),
        tuple(str(d) for d in external_dirs),
        tuple(sorted(str(t) for t in (available_tools or set()))),
        tuple(sorted(str(ts) for ts in (available_toolsets or set()))),
        platform_hint,
        tuple(sorted(disabled)),
    )
    with _INVENTORY_LOCK:
        cached = _INVENTORY_CACHE.get(cache_key)
        if cached is not None:
            _INVENTORY_CACHE.move_to_end(cache_key)
            return cached

    entries: list[SkillEntry] = []
    category_descriptions: dict[str, str] = {}

    # Local: snapshot fast path or full scan
    snapshot = _load_snapshot(skills_dir) if skills_dir.exists() else None
    if snapshot is not None:
        for raw in snapshot.get("skills", []):
            if not isinstance(raw, dict):
                continue
            entry = SkillEntry(**raw)
            entries.append(entry)
        category_descriptions.update(
            {str(k): str(v) for k, v in (snapshot.get("category_descriptions") or {}).items()}
        )
    elif skills_dir.exists():
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            e = _entry_from_skill_file(skill_file, skills_dir, source="local")
            if e is not None:
                entries.append(e)
        category_descriptions.update(_read_category_descriptions(skills_dir))
        _write_snapshot(skills_dir, _build_manifest(skills_dir), entries, category_descriptions)

    # External dirs (always direct scan)
    seen_names = {e.name for e in entries}
    for ext_dir in external_dirs:
        if not ext_dir.exists():
            continue
        for skill_file in iter_skill_index_files(ext_dir, "SKILL.md"):
            e = _entry_from_skill_file(skill_file, ext_dir, source="external")
            if e is None or e.name in seen_names:
                continue
            entries.append(e)
            seen_names.add(e.name)
        for cat, desc in _read_category_descriptions(ext_dir).items():
            category_descriptions.setdefault(cat, desc)

    # Apply disabled filter and condition filter at use time
    def _allowed(e: SkillEntry) -> bool:
        if e.name in disabled or e.skill_name in disabled:
            return False
        if available_tools is None and available_toolsets is None:
            return True
        at = available_tools or set()
        ats = available_toolsets or set()
        c = e.conditions or {}
        for ts in c.get("fallback_for_toolsets", []):
            if ts in ats:
                return False
        for t in c.get("fallback_for_tools", []):
            if t in at:
                return False
        for ts in c.get("requires_toolsets", []):
            if ts not in ats:
                return False
        for t in c.get("requires_tools", []):
            if t not in at:
                return False
        return True

    filtered = tuple(e for e in entries if _allowed(e))
    inv = Inventory(entries=filtered, category_descriptions=dict(category_descriptions))

    with _INVENTORY_LOCK:
        _INVENTORY_CACHE[cache_key] = inv
        _INVENTORY_CACHE.move_to_end(cache_key)
        while len(_INVENTORY_CACHE) > _INVENTORY_CACHE_MAX:
            _INVENTORY_CACHE.popitem(last=False)
    return inv


def clear_inventory_cache(*, clear_snapshot: bool = False) -> None:
    with _INVENTORY_LOCK:
        _INVENTORY_CACHE.clear()
    if clear_snapshot:
        try:
            _snapshot_path().unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Could not remove snapshot: %s", e)
```

- [ ] **Step 1.4: Refactor `prompt_builder.py` to consume the inventory**

Replace the body of `build_skills_system_prompt()` (lines 621-847) so it loads the inventory and renders the V1 format from `Inventory.entries` / `Inventory.category_descriptions`. Delete now-unused helpers (`_build_skills_manifest`, `_load_skills_snapshot`, `_write_skills_snapshot`, `_build_snapshot_entry`, `_parse_skill_file`, `_skill_should_show`, `_SKILLS_PROMPT_CACHE_*`). Keep `clear_skills_system_prompt_cache()` as a thin wrapper that also calls `clear_inventory_cache()`:

```python
# agent/prompt_builder.py — replacement for the skills-prompt section

from agent.skill_inventory import (
    Inventory,
    SkillEntry,
    clear_inventory_cache,
    load_inventory,
)

def clear_skills_system_prompt_cache(*, clear_snapshot: bool = False) -> None:
    """Drop the in-process skills prompt cache (and optionally the disk snapshot)."""
    clear_inventory_cache(clear_snapshot=clear_snapshot)


def build_skills_system_prompt(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
) -> str:
    inv = load_inventory(available_tools, available_toolsets)
    if not inv.entries:
        return ""
    return _render_v1(inv)


def _render_v1(inv: Inventory) -> str:
    skills_by_category: dict[str, list[SkillEntry]] = {}
    for e in inv.entries:
        skills_by_category.setdefault(e.category, []).append(e)

    index_lines: list[str] = []
    for category in sorted(skills_by_category.keys()):
        cat_desc = inv.category_descriptions.get(category, "")
        index_lines.append(f"  {category}: {cat_desc}" if cat_desc else f"  {category}:")
        seen: set[str] = set()
        for entry in sorted(skills_by_category[category], key=lambda x: x.name):
            if entry.name in seen:
                continue
            seen.add(entry.name)
            if entry.description:
                index_lines.append(f"    - {entry.name}: {entry.description}")
            else:
                index_lines.append(f"    - {entry.name}")

    return (
        "## Skills (mandatory)\n"
        "Before replying, scan the skills below. If a skill matches or is even partially relevant "
        "to your task, you MUST load it with skill_view(name) and follow its instructions. "
        "Err on the side of loading — it is always better to have context you don't need "
        "than to miss critical steps, pitfalls, or established workflows. "
        "Skills contain specialized knowledge — API endpoints, tool-specific commands, "
        "and proven workflows that outperform general-purpose approaches. Load the skill "
        "even if you think you could handle the task with basic tools like web_search or terminal. "
        "Skills also encode the user's preferred approach, conventions, and quality standards "
        "for tasks like code review, planning, and testing — load them even for tasks you "
        "already know how to do, because the skill defines how it should be done here.\n"
        "If a skill has issues, fix it with skill_manage(action='patch').\n"
        "After difficult/iterative tasks, offer to save as a skill. "
        "If a skill you loaded was missing steps, had wrong commands, or needed "
        "pitfalls you discovered, update it before finishing.\n"
        "\n"
        "<available_skills>\n"
        + "\n".join(index_lines) + "\n"
        "</available_skills>\n"
        "\n"
        "Only proceed without loading a skill if genuinely none are relevant to the task."
    )
```

- [ ] **Step 1.5: Run tests**

Run: `python -m pytest tests/agent/test_skill_inventory.py tests/agent/test_prompt_builder.py -q`
Expected: all green. The existing prompt_builder tests must still pass — this is the refactor's safety net.

- [ ] **Step 1.6: Commit**

```bash
git add agent/skill_inventory.py agent/prompt_builder.py tests/agent/test_skill_inventory.py
git commit -m "refactor(skills): extract SkillInventory module from prompt_builder

Pure refactor — renderer behavior unchanged.  Creates the seam needed by
skill_describe and the v2 rendering path.  Snapshot version bumped to 2 so
old on-disk snapshots are invalidated on first run."
```

---

## Task A2: Validate `priority` field in `skill_manage`

**Files:**
- Modify: `tools/skill_manager_tool.py:172-208`
- Test: `tests/tools/test_skill_manager_tool.py` (extend if exists, else create)

- [ ] **Step 2.1: Write failing test**

```python
# tests/tools/test_skill_manager_tool.py
import textwrap
from tools.skill_manager_tool import _validate_frontmatter


def test_priority_must_be_critical_or_normal():
    bad = textwrap.dedent("""\
        ---
        name: x
        description: y
        priority: legendary
        ---
        body
    """)
    err = _validate_frontmatter(bad)
    assert err is not None and "priority" in err


def test_priority_critical_is_accepted():
    good = textwrap.dedent("""\
        ---
        name: x
        description: y
        priority: critical
        ---
        body
    """)
    assert _validate_frontmatter(good) is None


def test_priority_absent_is_accepted():
    plain = textwrap.dedent("""\
        ---
        name: x
        description: y
        ---
        body
    """)
    assert _validate_frontmatter(plain) is None
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_skill_manager_tool.py::test_priority_must_be_critical_or_normal -v`
Expected: FAIL with `assert None is not None`

- [ ] **Step 2.3: Add the validation**

In `tools/skill_manager_tool.py:_validate_frontmatter`, after the existing `description` length check:

```python
    # Validate optional priority field (introduced for skills.index_v2)
    priority_raw = parsed.get("priority")
    if priority_raw is not None:
        priority = str(priority_raw).strip().lower()
        if priority not in ("critical", "normal"):
            return (
                f"Frontmatter 'priority' must be 'critical' or 'normal' "
                f"(got {priority_raw!r})."
            )
```

- [ ] **Step 2.4: Run tests**

Run: `python -m pytest tests/tools/test_skill_manager_tool.py -v`
Expected: all three new tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add tools/skill_manager_tool.py tests/tools/test_skill_manager_tool.py
git commit -m "feat(skills): validate priority frontmatter field in skill_manage

Allows skills to declare priority: critical so the v2 index keeps their
description in Tier 1.  Rejects unknown values to fail fast at create/patch
time."
```

---

## Task A3: Wire `skills.index_v2` config

**Files:**
- Modify: `agent/prompt_builder.py` (signature of `build_skills_system_prompt`)
- Modify: `run_agent.py:1607-1614` and the call site at `run_agent.py:4287`

- [ ] **Step 3.1: Write failing test**

```python
# tests/agent/test_prompt_builder.py — append
def test_build_skills_system_prompt_v2_flag_changes_render(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    (skills_dir / "cat" / "alpha").mkdir(parents=True)
    (skills_dir / "cat" / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: A description of alpha\n---\nbody\n"
    )
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr("agent.prompt_builder.get_skills_dir", lambda: skills_dir)
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)

    from agent.prompt_builder import build_skills_system_prompt
    v1 = build_skills_system_prompt()
    v2 = build_skills_system_prompt(index_v2=True)
    assert "A description of alpha" in v1, "V1 keeps the description inline"
    assert "## Skills" in v2 and "skill_describe" in v2, "V2 references skill_describe"
```

- [ ] **Step 3.2: Run test, expect failure**

Run: `python -m pytest tests/agent/test_prompt_builder.py::test_build_skills_system_prompt_v2_flag_changes_render -v`
Expected: FAIL — `index_v2` is not a recognized kwarg yet.

- [ ] **Step 3.3: Add the `index_v2` and `index_token_budget` params (no body change yet)**

In `agent/prompt_builder.py`:

```python
def build_skills_system_prompt(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
    *,
    index_v2: bool = False,
    index_token_budget: int = 2000,
) -> str:
    inv = load_inventory(available_tools, available_toolsets)
    if not inv.entries:
        return ""
    if index_v2:
        return _render_v2(inv, token_budget=index_token_budget)
    return _render_v1(inv)


def _render_v2(inv: Inventory, *, token_budget: int) -> str:
    # Implemented in Task A4 — placeholder for now so the test wiring works.
    return "## Skills\n\n[skill_describe wiring placeholder]\n"
```

- [ ] **Step 3.4: Load `skills.index_v2` config in run_agent**

In `run_agent.py:1607-1614`, replace the existing skills-config block with:

```python
        # Skills config: nudge interval + index v2 flag
        self._skill_nudge_interval = 10
        self._skill_index_v2 = False
        self._skill_index_token_budget = 2000
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
            self._skill_index_v2 = bool(skills_config.get("index_v2", False))
            self._skill_index_token_budget = int(skills_config.get("index_token_budget", 2000))
        except Exception:
            pass
```

At the existing call site `run_agent.py:4287-4290`, extend the call with the two new keyword arguments (the existing `available_tools=self.valid_tool_names, available_toolsets=avail_toolsets` stay as-is):

```python
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
                index_v2=self._skill_index_v2,
                index_token_budget=self._skill_index_token_budget,
            )
```

- [ ] **Step 3.5: Run tests**

Run: `python -m pytest tests/agent/test_prompt_builder.py -v`
Expected: pass — V2 returns the placeholder string and V1 still renders fully.

- [ ] **Step 3.6: Commit**

```bash
git add agent/prompt_builder.py run_agent.py tests/agent/test_prompt_builder.py
git commit -m "feat(skills): wire index_v2 and index_token_budget config flags

Threads the new flags through build_skills_system_prompt; v2 path is a
placeholder until A4."
```

---

## Task A4: Implement v2 rendering format + critical-count warning

**Files:**
- Modify: `agent/prompt_builder.py` (`_render_v2`)
- Test: `tests/agent/test_prompt_builder.py`

- [ ] **Step 4.1: Write failing test**

```python
# tests/agent/test_prompt_builder.py — append
import textwrap


def _make_skill(skills_dir, name, description, *, category="cat", priority=None):
    d = skills_dir / category / name
    d.mkdir(parents=True, exist_ok=True)
    front = ["---", f"name: {name}", f"description: {description}"]
    if priority:
        front.append(f"priority: {priority}")
    front.append("---")
    (d / "SKILL.md").write_text("\n".join(front) + "\nbody\n")


def test_v2_renders_critical_with_description_normal_without(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "alpha", "Alpha desc", category="cat-a", priority="critical")
    _make_skill(skills_dir, "beta", "Beta desc", category="cat-a")
    _make_skill(skills_dir, "gamma", "Gamma desc", category="cat-b")
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr("agent.prompt_builder.get_skills_dir", lambda: skills_dir)
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)

    from agent.prompt_builder import build_skills_system_prompt
    out = build_skills_system_prompt(index_v2=True)

    # Critical: full description present
    assert "alpha: Alpha desc" in out
    # Normal: name only, descriptions of beta/gamma absent
    assert "Beta desc" not in out
    assert "Gamma desc" not in out
    # Names of normal skills appear as comma list
    assert "beta" in out and "gamma" in out
    # Categories appear
    assert "cat-a" in out and "cat-b" in out
    # skill_describe is referenced in the preamble
    assert "skill_describe" in out
```

- [ ] **Step 4.2: Run test, expect failure**

Run: `python -m pytest tests/agent/test_prompt_builder.py::test_v2_renders_critical_with_description_normal_without -v`
Expected: FAIL — placeholder body.

- [ ] **Step 4.3: Implement `_render_v2`**

Replace the placeholder in `agent/prompt_builder.py`:

```python
_V2_PREAMBLE = (
    "## Skills\n"
    "\n"
    "Tools available:\n"
    "  - skill_view(name): load full skill content\n"
    "  - skill_describe(category=..., names=[...]): get one-line descriptions\n"
    "  - skill_manage: create / patch / write_file\n"
    "\n"
    "The list below shows every available skill organized by category. Skills "
    "with descriptions are critical workflows you should consider on every task. "
    "For other skills, the name is a hint — if any look relevant, call "
    "skill_describe(category=\"<cat>\") to see one-line descriptions, then "
    "skill_view(name) to load the full content.\n"
    "\n"
)

_V2_OPEN = "<available_skills>\n"
_V2_CLOSE = "</available_skills>\n"


def _render_v2(inv: Inventory, *, token_budget: int) -> str:
    by_cat: dict[str, list[SkillEntry]] = {}
    for e in inv.entries:
        by_cat.setdefault(e.category, []).append(e)

    # Q1 safeguard: warn if `priority: critical` looks like it's becoming default.
    n_critical = sum(1 for e in inv.entries if e.priority == "critical")
    if n_critical > 15:
        logger.warning(
            "skills.index_v2: %d skills marked priority=critical (>15). "
            "Tier 1 budget gains erode quickly past this threshold.",
            n_critical,
        )

    # Render every category — budget enforcement comes in Task A6.
    body_lines: list[str] = []
    for cat in sorted(by_cat.keys()):
        cat_desc = inv.category_descriptions.get(cat, "")
        body_lines.append(f"  {cat}/  {cat_desc}".rstrip() if cat_desc else f"  {cat}/")
        criticals = sorted(
            (e for e in by_cat[cat] if e.priority == "critical"),
            key=lambda e: e.name,
        )
        normals = sorted(
            (e for e in by_cat[cat] if e.priority != "critical"),
            key=lambda e: e.name,
        )
        for e in criticals:
            if e.description:
                body_lines.append(f"    - {e.name}: {e.description}")
            else:
                body_lines.append(f"    - {e.name}")
        if normals:
            body_lines.append("    " + ", ".join(e.name for e in normals))

    return _V2_PREAMBLE + _V2_OPEN + "\n".join(body_lines) + "\n" + _V2_CLOSE
```

- [ ] **Step 4.4: Run tests**

Run: `python -m pytest tests/agent/test_prompt_builder.py -v`
Expected: all green.

- [ ] **Step 4.5: Commit**

```bash
git add agent/prompt_builder.py tests/agent/test_prompt_builder.py
git commit -m "feat(skills): implement v2 tier-1 rendering for skill index

Critical-priority skills keep full descriptions; normal-priority skills are
listed by name only, organized under their category. Logs a warning when
more than 15 skills are marked critical (spec Q1)."
```

---

## Task A5: Skill usage tracker (`~/.hermes/.skill_usage.json` + `~/.hermes/.skill_known_clis.json`)

**Files:**
- Create: `agent/skill_usage_tracker.py`
- Test: `tests/agent/test_skill_usage_tracker.py`

- [ ] **Step 5.1: Write failing tests**

```python
# tests/agent/test_skill_usage_tracker.py
from pathlib import Path
import json
import time

from agent import skill_usage_tracker as t


def test_record_and_top_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "_usage_path", lambda: tmp_path / ".skill_usage.json")
    t.record_category_use("cat-a")
    t.record_category_use("cat-a")
    t.record_category_use("cat-b")
    top = t.top_categories(within_seconds=3600)
    assert top[0] == "cat-a"
    assert "cat-b" in top


def test_record_corrupted_file_falls_back(tmp_path, monkeypatch):
    p = tmp_path / ".skill_usage.json"
    p.write_text("{not json")
    monkeypatch.setattr(t, "_usage_path", lambda: p)
    t.record_category_use("cat-a")  # must not raise
    assert "cat-a" in t.top_categories()


def test_known_clis_pruned_after_window(tmp_path, monkeypatch):
    p = tmp_path / ".skill_known_clis.json"
    monkeypatch.setattr(t, "_known_clis_path", lambda: p)
    t.record_cli_seen("rg")
    # Force-age the entry
    raw = json.loads(p.read_text())
    raw["rg"] = time.time() - (40 * 86400)
    p.write_text(json.dumps(raw))
    assert t.is_known_cli("rg", window_days=30) is False
```

- [ ] **Step 5.2: Run tests, expect import failure**

Run: `python -m pytest tests/agent/test_skill_usage_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: agent.skill_usage_tracker`

- [ ] **Step 5.3: Create the module**

```python
# agent/skill_usage_tracker.py
"""Best-effort persistence for skill-category usage and known-CLI history.

Both files are best-effort: corruption falls back to empty state, never raises.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_TIMESTAMPS_PER_CAT = 30
_DEFAULT_USAGE_WINDOW_S = 30 * 86400
_DEFAULT_CLI_WINDOW_DAYS = 30


def _usage_path() -> Path:
    return get_hermes_home() / ".skill_usage.json"


def _known_clis_path() -> Path:
    return get_hermes_home() / ".skill_known_clis.json"


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("Could not read %s: %s", path, e)
        return {}


def _write_json(path: Path, payload: dict) -> None:
    try:
        atomic_json_write(path, payload)
    except Exception as e:
        logger.debug("Could not write %s: %s", path, e)


def record_category_use(category: str) -> None:
    if not category:
        return
    data = _read_json(_usage_path())
    history = list(data.get(category) or [])
    history.append(int(time.time()))
    history = history[-_MAX_TIMESTAMPS_PER_CAT:]
    data[category] = history
    _write_json(_usage_path(), data)


def top_categories(within_seconds: int = _DEFAULT_USAGE_WINDOW_S) -> list[str]:
    data = _read_json(_usage_path())
    cutoff = int(time.time()) - within_seconds
    scored: list[tuple[int, str]] = []
    for cat, ts_list in data.items():
        if not isinstance(ts_list, list):
            continue
        recent = sum(1 for t in ts_list if isinstance(t, (int, float)) and t >= cutoff)
        if recent > 0:
            scored.append((recent, cat))
    scored.sort(reverse=True)
    return [c for _, c in scored]


def record_cli_seen(cli: str) -> None:
    if not cli:
        return
    data = _read_json(_known_clis_path())
    data[cli] = int(time.time())
    _write_json(_known_clis_path(), data)


def is_known_cli(cli: str, *, window_days: int = _DEFAULT_CLI_WINDOW_DAYS) -> bool:
    data = _read_json(_known_clis_path())
    last = data.get(cli)
    if not isinstance(last, (int, float)):
        return False
    return (time.time() - last) <= (window_days * 86400)


def usage_rank_epoch() -> str:
    """Day-level key derived from the usage file mtime — used for cache invalidation.

    Stable within a session, advances at most once per day even if the file is
    written more often.
    """
    p = _usage_path()
    try:
        mtime = p.stat().st_mtime if p.exists() else 0
    except OSError:
        mtime = 0
    return time.strftime("%Y-%m-%d", time.gmtime(mtime))
```

- [ ] **Step 5.4: Run tests**

Run: `python -m pytest tests/agent/test_skill_usage_tracker.py -v`
Expected: all green.

- [ ] **Step 5.5: Hook tracker into `skill_view` and `skill_describe` (later)**

In `tools/skills_tool.py:828-...` (the `skill_view` body), after a successful load, add:

```python
    # Best-effort usage tracking — never fails the tool call
    try:
        from agent import skill_usage_tracker
        # Look up the entry's category from the inventory
        from agent.skill_inventory import load_inventory
        inv = load_inventory()
        for entry in inv.entries:
            if entry.name == name or entry.skill_name == name:
                skill_usage_tracker.record_category_use(entry.category)
                break
    except Exception:
        pass
```

- [ ] **Step 5.6: Commit**

```bash
git add agent/skill_usage_tracker.py tools/skills_tool.py tests/agent/test_skill_usage_tracker.py
git commit -m "feat(skills): add usage tracker for categories and known CLIs

Persists best-effort usage stats to ~/.hermes/.skill_usage.json (categories)
and ~/.hermes/.skill_known_clis.json (CLIs).  Hooked into skill_view; consumed
by index v2 budget folding and S2 nudge signal."
```

---

## Task A6: Token budget + fold-back

**Files:**
- Modify: `agent/prompt_builder.py` (`_render_v2`)
- Test: `tests/agent/test_prompt_builder.py`

- [ ] **Step 6.1: Write failing test**

```python
# tests/agent/test_prompt_builder.py — append
def test_v2_budget_folds_least_used_categories(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    # Generate 20 categories, 10 skills each
    for ci in range(20):
        for si in range(10):
            d = skills_dir / f"cat-{ci:02d}" / f"skill-{ci:02d}-{si:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(
                f"---\nname: skill-{ci:02d}-{si:02d}\ndescription: desc\n---\nbody\n"
            )
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)

    # No usage history → fold-back will engage
    from agent.prompt_builder import build_skills_system_prompt
    out = build_skills_system_prompt(index_v2=True, index_token_budget=300)
    assert "and" in out and "more categories" in out, "Fold-back line absent"
    assert "skill_describe" in out


def test_v2_under_budget_renders_full(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    for i in range(3):
        d = skills_dir / "cat" / f"sk-{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: sk-{i}\ndescription: d\n---\nbody\n"
        )
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)
    from agent.prompt_builder import build_skills_system_prompt
    out = build_skills_system_prompt(index_v2=True, index_token_budget=2000)
    assert "more categories" not in out
```

- [ ] **Step 6.2: Run, expect failure**

Run: `python -m pytest tests/agent/test_prompt_builder.py::test_v2_budget_folds_least_used_categories -v`
Expected: FAIL — current renderer doesn't fold.

- [ ] **Step 6.3: Implement budget enforcement**

Replace the body of `_render_v2` in `agent/prompt_builder.py`:

```python
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _render_v2(inv: Inventory, *, token_budget: int) -> str:
    from agent import skill_usage_tracker

    by_cat: dict[str, list[SkillEntry]] = {}
    for e in inv.entries:
        by_cat.setdefault(e.category, []).append(e)

    n_critical = sum(1 for e in inv.entries if e.priority == "critical")
    if n_critical > 15:
        logger.warning(
            "skills.index_v2: %d skills marked priority=critical (>15). "
            "Tier 1 budget gains erode past this threshold.",
            n_critical,
        )

    def _category_block(cat: str) -> list[str]:
        cat_desc = inv.category_descriptions.get(cat, "")
        head = f"  {cat}/  {cat_desc}".rstrip() if cat_desc else f"  {cat}/"
        out = [head]
        criticals = sorted((e for e in by_cat[cat] if e.priority == "critical"),
                           key=lambda e: e.name)
        normals = sorted((e for e in by_cat[cat] if e.priority != "critical"),
                         key=lambda e: e.name)
        for e in criticals:
            if e.description:
                out.append(f"    - {e.name}: {e.description}")
            else:
                out.append(f"    - {e.name}")
        if normals:
            out.append("    " + ", ".join(e.name for e in normals))
        return out

    # Category ordering for fold-back: most-used first, then alphabetical
    top = skill_usage_tracker.top_categories()
    ordered_cats = list(dict.fromkeys(top + sorted(by_cat.keys())))
    ordered_cats = [c for c in ordered_cats if c in by_cat]

    body_lines: list[str] = []
    folded: list[str] = []
    running_tokens = _estimate_tokens(_V2_PREAMBLE) + _estimate_tokens(_V2_OPEN + _V2_CLOSE)
    for cat in ordered_cats:
        block = _category_block(cat)
        block_text = "\n".join(block) + "\n"
        cost = _estimate_tokens(block_text)
        if running_tokens + cost > token_budget and body_lines:
            folded.append(cat)
        else:
            body_lines.extend(block)
            running_tokens += cost

    if folded:
        body_lines.append(
            f"  … and {len(folded)} more categories: "
            + ", ".join(folded)
            + ". Call skill_describe(category=...) to expand any of these."
        )

    return _V2_PREAMBLE + _V2_OPEN + "\n".join(body_lines) + "\n" + _V2_CLOSE
```

- [ ] **Step 6.4: Run tests**

Run: `python -m pytest tests/agent/test_prompt_builder.py -v`
Expected: all green.

- [ ] **Step 6.5: Update cache key**

Inside `agent/skill_inventory.py:load_inventory`, the cache key already includes the platform/disabled set; for the v2 budget output we need a separate cache for the rendered string. Since `build_skills_system_prompt` is called once per `_build_system_prompt`, simply do not cache the v2 rendered string — recomputing the render from the cached inventory is cheap (sub-millisecond for 100 skills). No change needed; document this in a comment in `_render_v2`:

```python
def _render_v2(inv: Inventory, *, token_budget: int) -> str:
    # Note: not cached at the prompt-string level — the inventory cache covers
    # the expensive parsing step, and rendering 100 skills is sub-ms.
    ...
```

- [ ] **Step 6.6: Commit**

```bash
git add agent/prompt_builder.py tests/agent/test_prompt_builder.py
git commit -m "feat(skills): enforce token budget with fold-back in v2 index

Categories are ordered by recent usage (skill_usage_tracker.top_categories)
and added until the budget is met; the rest are listed by name with a hint
to call skill_describe."
```

---

## Task A7: New `skill_describe` tool

**Files:**
- Modify: `tools/skills_tool.py:1388-1445`
- Test: `tests/tools/test_skill_describe.py`

- [ ] **Step 7.1: Write failing tests**

```python
# tests/tools/test_skill_describe.py
import textwrap
import pytest


def _setup_skills(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    for cat, name, desc in [
        ("cat-a", "alpha", "A description"),
        ("cat-a", "beta", "B description"),
        ("cat-b", "gamma", "C description"),
    ]:
        sd = d / cat / name
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "SKILL.md").write_text(textwrap.dedent(f"""\
            ---
            name: {name}
            description: {desc}
            ---
            body
        """))
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: d)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [d])
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)


def test_skill_describe_by_category(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe
    res = skill_describe(category="cat-a")
    assert res["success"] is True
    names = sorted(s["name"] for s in res["skills"])
    assert names == ["alpha", "beta"]


def test_skill_describe_by_names(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe
    res = skill_describe(names=["alpha", "gamma"])
    assert res["success"] is True
    names = sorted(s["name"] for s in res["skills"])
    assert names == ["alpha", "gamma"]


def test_skill_describe_unknown_name_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe
    res = skill_describe(names=["does-not-exist"])
    assert res["success"] is False
    assert "does-not-exist" in res["error"]


def test_skill_describe_unknown_category_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe
    res = skill_describe(category="missing")
    assert res["success"] is False


def test_skill_describe_empty_args_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe
    res = skill_describe()
    assert res["success"] is False
```

- [ ] **Step 7.2: Run, expect failure**

Run: `python -m pytest tests/tools/test_skill_describe.py -v`
Expected: FAIL — `skill_describe` not exported.

- [ ] **Step 7.3: Implement `skill_describe` in `tools/skills_tool.py`**

After the existing `skill_view` definition (line 828 onwards), add:

```python
def skill_describe(
    category: str | None = None,
    names: list[str] | None = None,
    task_id: str | None = None,
) -> dict:
    """Return one-line descriptions for one or more skills.

    Either `category` or `names` must be provided.  Returns success=False on
    unknown categories/names or empty input.  Best-effort updates the category
    usage tracker so future v2 prompt-budget folding ranks the touched
    category higher.
    """
    if not category and not names:
        return {
            "success": False,
            "error": "Provide either category=<name> or names=[...].",
        }

    from agent.skill_inventory import load_inventory
    inv = load_inventory()
    by_name = {e.name: e for e in inv.entries}
    by_skill_name = {e.skill_name: e for e in inv.entries}
    by_category: dict[str, list] = {}
    for e in inv.entries:
        by_category.setdefault(e.category, []).append(e)

    out: list[dict] = []
    if category:
        if category not in by_category:
            return {
                "success": False,
                "error": f"Unknown category {category!r}.  Categories: "
                         + ", ".join(sorted(by_category.keys())),
            }
        for e in sorted(by_category[category], key=lambda e: e.name):
            out.append({
                "name": e.name,
                "category": e.category,
                "description": e.description,
                "priority": e.priority,
            })
        try:
            from agent import skill_usage_tracker
            skill_usage_tracker.record_category_use(category)
        except Exception:
            pass

    if names:
        missing: list[str] = []
        for n in names:
            entry = by_name.get(n) or by_skill_name.get(n)
            if entry is None:
                missing.append(n)
                continue
            out.append({
                "name": entry.name,
                "category": entry.category,
                "description": entry.description,
                "priority": entry.priority,
            })
        if missing:
            return {
                "success": False,
                "error": f"Unknown skill name(s): {', '.join(missing)}.",
            }

    return {"success": True, "skills": out}
```

- [ ] **Step 7.4: Register the tool schema**

In `tools/skills_tool.py:1392-1445`, add after `SKILL_VIEW_SCHEMA`:

```python
SKILL_DESCRIBE_SCHEMA = {
    "name": "skill_describe",
    "description": (
        "Return one-line descriptions for skills. Use category=<name> to expand "
        "a category from the system-prompt index, or names=[...] for specific "
        "skills you've already identified. After reading a description, call "
        "skill_view(name) to load the full SKILL.md content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Category to expand — returns descriptions for every skill in it.",
            },
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific skill names to describe.",
            },
        },
        "required": [],
    },
}

registry.register(
    name="skill_describe",
    toolset="skills",
    schema=SKILL_DESCRIBE_SCHEMA,
    handler=lambda args, **kw: skill_describe(
        category=args.get("category"),
        names=args.get("names"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
```

- [ ] **Step 7.5: Run tests**

Run: `python -m pytest tests/tools/test_skill_describe.py -v`
Expected: all green.

- [ ] **Step 7.6: Commit**

```bash
git add tools/skills_tool.py tests/tools/test_skill_describe.py
git commit -m "feat(skills): add skill_describe tool for tier-2 description lookup

Lets the model expand a category or fetch descriptions for specific skills
when their names alone aren't enough to decide whether to skill_view.
Records category usage to inform future budget-folding."
```

---

## Task A8: Phase A integration test

**Files:**
- Test: `tests/integration/test_skills_index_v2_e2e.py`

- [ ] **Step 8.1: Create the integration test**

```python
# tests/integration/test_skills_index_v2_e2e.py
"""End-to-end: 100 synthetic skills, assert prompt budget is respected and
no skill is unreachable via skill_describe."""
from pathlib import Path
import pytest


def _seed_skills(skills_dir: Path, n: int = 100, n_categories: int = 10) -> None:
    skills_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cat = f"cat-{i % n_categories:02d}"
        d = skills_dir / cat / f"skill-{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        priority = "critical" if i < 5 else "normal"
        (d / "SKILL.md").write_text(
            f"---\nname: skill-{i:03d}\ndescription: synthetic skill {i}\npriority: {priority}\n---\nbody\n"
        )


def test_v2_budget_and_full_reachability(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    _seed_skills(skills_dir, n=100, n_categories=10)
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr("agent.prompt_builder.get_skills_dir", lambda: skills_dir)
    from agent.skill_inventory import clear_inventory_cache
    clear_inventory_cache(clear_snapshot=True)

    from agent.prompt_builder import build_skills_system_prompt
    prompt = build_skills_system_prompt(index_v2=True, index_token_budget=2000)

    # 1. Budget respected within tolerance (the heuristic is len/4)
    assert len(prompt) // 4 <= 2200, f"Prompt size {len(prompt)//4} exceeds budget"

    # 2. All 5 critical skills appear with descriptions
    for i in range(5):
        assert f"skill-{i:03d}: synthetic skill {i}" in prompt

    # 3. Every skill is reachable via skill_describe
    from tools.skills_tool import skill_describe
    for i in range(100):
        res = skill_describe(names=[f"skill-{i:03d}"])
        assert res["success"], f"skill-{i:03d} not reachable"
```

- [ ] **Step 8.2: Run**

Run: `python -m pytest tests/integration/test_skills_index_v2_e2e.py -v`
Expected: green.

- [ ] **Step 8.3: Commit**

```bash
git add tests/integration/test_skills_index_v2_e2e.py
git commit -m "test(skills): integration test for v2 index with 100 synthetic skills"
```

---

## Task A9: Update `cli-config.yaml.example` (Phase A part)

**Files:**
- Modify: `cli-config.yaml.example:481-494`

- [ ] **Step 9.1: Edit the skills section**

Replace the existing `skills:` block (lines 481-494) with:

```yaml
skills:
  # Nudge the agent to create skills after complex tasks.
  # Phase 1-2 fallback only — primary triggers are signal-based (see below).
  # Set to 0 to disable.
  creation_nudge_interval: 15

  # Skill index v2 — Tier 1 (system prompt) shows category + names only;
  # descriptions are fetched on demand via skill_describe.  Skills with
  # `priority: critical` in their frontmatter keep descriptions in Tier 1.
  # Default false until Phase 3 of the rollout.
  index_v2: false

  # Soft cap on the Tier 1 skill block (token estimate, len // 4).  When
  # exceeded, lowest-priority categories are folded behind a one-line hint
  # that points the model at skill_describe.
  index_token_budget: 2000

  # External skill directories — share skills across tools/agents without
  # copying them into ~/.hermes/skills/.  Each path is expanded (~ and ${VAR})
  # and resolved to an absolute path.  External dirs are read-only: skill
  # creation always writes to ~/.hermes/skills/.  Local skills take precedence
  # when names collide.
  # external_dirs:
  #   - ~/.agents/skills
  #   - /home/shared/team-skills
```

(Phase B keys will be added in Task B7.)

- [ ] **Step 9.2: Commit**

```bash
git add cli-config.yaml.example
git commit -m "docs(config): document skills.index_v2 and index_token_budget"
```

---

# Phase B — Signal-Based Nudge

Gated behind `skills.nudge_signals.enabled: true`. Default `false` until Phase 3.

## Task B1: Wire `skills.nudge_signals` config + AIAgent state init

**Files:**
- Modify: `run_agent.py:1495-1614`

- [ ] **Step 10.1: Write failing test**

```python
# tests/run_agent/test_nudge_signals_init.py
def test_nudge_state_initialized_with_defaults(monkeypatch):
    from run_agent import AIAgent
    a = AIAgent.__new__(AIAgent)  # bypass full __init__ for an isolated check
    AIAgent._init_nudge_state(a, {})
    assert a._nudge_disabled is False
    assert a._nudge_signals == set()
    assert a._nudge_signals_enabled is False
    assert a._nudge_repeated_threshold == 3
    assert a._nudge_error_threshold == 2
    assert a._nudge_user_phrases == ["next time", "remember", "from now on", "记一下", "下次", "以后"]
    assert "git" in a._nudge_common_clis_suppressed


def test_nudge_state_reads_config():
    from run_agent import AIAgent
    cfg = {
        "skills": {
            "nudge_signals": {
                "enabled": True,
                "repeated_pattern_threshold": 5,
                "error_repeat_threshold": 3,
                "novel_cli_window_days": 14,
                "common_cli_suppressions": ["bun"],
                "user_phrases": ["foo"],
            }
        }
    }
    a = AIAgent.__new__(AIAgent)
    AIAgent._init_nudge_state(a, cfg)
    assert a._nudge_signals_enabled is True
    assert a._nudge_repeated_threshold == 5
    assert a._nudge_error_threshold == 3
    assert a._nudge_cli_window_days == 14
    assert a._nudge_common_clis_suppressed == ["bun"]
    assert a._nudge_user_phrases == ["foo"]
```

- [ ] **Step 10.2: Run, expect failure**

Run: `python -m pytest tests/run_agent/test_nudge_signals_init.py -v`
Expected: FAIL — `_init_nudge_state` does not exist.

- [ ] **Step 10.3: Add `_init_nudge_state` and call it from `__init__`**

In `run_agent.py`, immediately after the existing block at lines 1607-1613, replace and extend:

```python
        # Skills config: nudge interval + index v2 flag (existing)
        self._skill_nudge_interval = 10
        self._skill_index_v2 = False
        self._skill_index_token_budget = 2000
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
            self._skill_index_v2 = bool(skills_config.get("index_v2", False))
            self._skill_index_token_budget = int(skills_config.get("index_token_budget", 2000))
        except Exception:
            pass

        # Phase B: signal-based nudge state
        self._init_nudge_state(_agent_cfg)
```

Add the method as a real method on `AIAgent`:

```python
    _DEFAULT_USER_PHRASES = ["next time", "remember", "from now on", "记一下", "下次", "以后"]
    _DEFAULT_COMMON_CLIS = ["git", "python", "python3", "node", "npm", "pnpm",
                            "uv", "pytest", "rg", "sed", "cat", "ls", "mkdir"]

    def _init_nudge_state(self, agent_cfg: dict) -> None:
        from collections import deque
        self._nudge_disabled = False
        self._nudge_signals: set[str] = set()
        self._tool_call_history: deque = deque(maxlen=10)
        self._error_history: deque = deque(maxlen=10)
        self._last_error_tool: dict[str, str] = {}  # tool_name -> last error hash

        skills_cfg = (agent_cfg or {}).get("skills", {}) or {}
        ns = skills_cfg.get("nudge_signals", {}) or {}
        self._nudge_signals_enabled = bool(ns.get("enabled", False))
        self._nudge_repeated_threshold = int(ns.get("repeated_pattern_threshold", 3))
        self._nudge_error_threshold = int(ns.get("error_repeat_threshold", 2))
        self._nudge_cli_window_days = int(ns.get("novel_cli_window_days", 30))
        self._nudge_common_clis_suppressed = list(
            ns.get("common_cli_suppressions", self._DEFAULT_COMMON_CLIS)
        )
        self._nudge_user_phrases = list(ns.get("user_phrases", self._DEFAULT_USER_PHRASES))

        import os
        if os.environ.get("HERMES_SKILL_NUDGE_DISABLE") == "1":
            self._nudge_disabled = True
```

- [ ] **Step 10.4: Run tests**

Run: `python -m pytest tests/run_agent/test_nudge_signals_init.py -v`
Expected: green.

- [ ] **Step 10.5: Commit**

```bash
git add run_agent.py tests/run_agent/test_nudge_signals_init.py
git commit -m "feat(nudge): initialize signal-based nudge state on AIAgent

Adds _init_nudge_state() reading skills.nudge_signals.* and the
HERMES_SKILL_NUDGE_DISABLE env var.  No behavior change yet — signal eval
arrives in B2/B3."
```

---

## Task B2: Signal evaluator module (S1–S4)

**Files:**
- Create: `agent/skill_nudge_signals.py`
- Test: `tests/agent/test_skill_nudge_signals.py`

- [ ] **Step 11.1: Write failing tests**

```python
# tests/agent/test_skill_nudge_signals.py
import pytest

from agent.skill_nudge_signals import SignalEvaluator


@pytest.fixture
def ev():
    return SignalEvaluator(
        repeated_threshold=3,
        error_threshold=2,
        common_clis_suppressed=["git", "ls"],
        cli_window_days=30,
        user_phrases=["next time", "记一下"],
    )


def test_s1_repeated_terminal_first_token(ev):
    for _ in range(3):
        ev.observe_tool_call("terminal", {"command": "gh pr view 123"})
    assert "S1" in ev.fired_signals


def test_s1_unknown_tool_does_not_fire(ev):
    for _ in range(5):
        ev.observe_tool_call("custom_tool", {"foo": i for i in range(1)})
    assert "S1" not in ev.fired_signals


def test_s2_novel_cli_fires(ev, monkeypatch):
    monkeypatch.setattr(
        "agent.skill_usage_tracker.is_known_cli", lambda *a, **kw: False
    )
    monkeypatch.setattr(
        "agent.skill_usage_tracker.record_cli_seen", lambda *a, **kw: None
    )
    ev.observe_tool_call("terminal", {"command": "exotic-cli --do-thing"}, success=True)
    assert "S2" in ev.fired_signals


def test_s2_common_cli_suppressed(ev, monkeypatch):
    monkeypatch.setattr(
        "agent.skill_usage_tracker.is_known_cli", lambda *a, **kw: False
    )
    monkeypatch.setattr(
        "agent.skill_usage_tracker.record_cli_seen", lambda *a, **kw: None
    )
    ev.observe_tool_call("terminal", {"command": "git status"}, success=True)
    assert "S2" not in ev.fired_signals


def test_s3_user_phrase_match(ev):
    ev.observe_user_message("This always crashes — please remember next time")
    assert "S3" in ev.fired_signals


def test_s4_resolved_repeated_error(ev):
    ev.observe_tool_result("terminal", error_text="ENOENT: file 'x' not found")
    ev.observe_tool_result("terminal", error_text="ENOENT: file 'x' not found")
    ev.observe_tool_result("terminal", error_text=None)  # success
    assert "S4" in ev.fired_signals


def test_clear_resets_state(ev):
    for _ in range(3):
        ev.observe_tool_call("terminal", {"command": "gh pr view"})
    assert ev.fired_signals
    ev.clear()
    assert not ev.fired_signals
```

- [ ] **Step 11.2: Run, expect failure**

Run: `python -m pytest tests/agent/test_skill_nudge_signals.py -v`
Expected: FAIL — module not found.

- [ ] **Step 11.3: Create the evaluator**

```python
# agent/skill_nudge_signals.py
"""Signal-based nudge evaluator (S1-S4 from the spec).

Pure logic — does not own any of the persistence; consumers wire it up.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field


def _terminal_signature(args: dict) -> str:
    cmd = (args or {}).get("command") or ""
    cmd = str(cmd).strip()
    if not cmd:
        return ""
    parts = cmd.split()
    head = parts[0].rsplit("/", 1)[-1] if parts else ""
    sub = parts[1] if len(parts) > 1 else ""
    return f"{head} {sub}".strip() if head in ("git", "gh", "kubectl") else head


def _path_signature(args: dict) -> str:
    p = (args or {}).get("path") or (args or {}).get("file_path") or ""
    p = str(p)
    return p.rsplit("/", 1)[0] if "/" in p else ""


_PER_TOOL_SIG = {
    "terminal": _terminal_signature,
    "process": _terminal_signature,
    "read_file": _path_signature,
    "write_file": _path_signature,
    "edit_file": _path_signature,
}


def _hash_error(text: str) -> str:
    return hashlib.sha1(text[:200].encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class SignalEvaluator:
    repeated_threshold: int = 3
    error_threshold: int = 2
    common_clis_suppressed: list[str] = field(default_factory=list)
    cli_window_days: int = 30
    user_phrases: list[str] = field(default_factory=list)

    fired_signals: set[str] = field(default_factory=set)
    _tool_calls: deque = field(default_factory=lambda: deque(maxlen=20))
    _error_counts: dict[str, int] = field(default_factory=dict)
    _failed_tools: set[str] = field(default_factory=set)

    # ---- public hooks -----------------------------------------------------

    def observe_user_message(self, text: str) -> None:
        if not text:
            return
        lower = text.lower()
        for phrase in self.user_phrases:
            if phrase.lower() in lower:
                self.fired_signals.add("S3")
                return

    def observe_tool_call(
        self,
        tool_name: str,
        args: dict | None,
        *,
        success: bool | None = None,
    ) -> None:
        sig_fn = _PER_TOOL_SIG.get(tool_name)
        if sig_fn is not None:
            sig = sig_fn(args or {})
            if sig:
                self._tool_calls.append((tool_name, sig))
                count = sum(1 for n, s in self._tool_calls if n == tool_name and s == sig)
                if count >= self.repeated_threshold:
                    self.fired_signals.add("S1")

        # S2: novel external CLI on terminal/process
        if tool_name in ("terminal", "process"):
            cmd = str((args or {}).get("command") or "").strip()
            if cmd:
                head = cmd.split()[0].rsplit("/", 1)[-1]
                if head and head not in self.common_clis_suppressed:
                    from agent import skill_usage_tracker
                    if not skill_usage_tracker.is_known_cli(head, window_days=self.cli_window_days):
                        # Fire only when the call succeeded or has repeated this turn
                        if success is True or (tool_name, head) in self._failed_tools:
                            self.fired_signals.add("S2")
                        skill_usage_tracker.record_cli_seen(head)

    def observe_tool_result(
        self,
        tool_name: str,
        *,
        error_text: str | None,
    ) -> None:
        if error_text:
            h = _hash_error(error_text)
            self._error_counts[h] = self._error_counts.get(h, 0) + 1
            self._failed_tools.add(tool_name)
        else:
            # A success after >= error_threshold prior errors of the same hash
            for h, count in list(self._error_counts.items()):
                if count >= self.error_threshold:
                    self.fired_signals.add("S4")
                    self._error_counts.pop(h, None)

    def clear(self) -> None:
        self.fired_signals.clear()
        self._tool_calls.clear()
        self._error_counts.clear()
        self._failed_tools.clear()
```

- [ ] **Step 11.4: Fix the test typo and run**

The S1 negative-case test had a stray dict comprehension. Fix it:

```python
def test_s1_unknown_tool_does_not_fire(ev):
    for _ in range(5):
        ev.observe_tool_call("custom_tool", {"foo": "bar"})
    assert "S1" not in ev.fired_signals
```

Run: `python -m pytest tests/agent/test_skill_nudge_signals.py -v`
Expected: all green.

- [ ] **Step 11.5: Commit**

```bash
git add agent/skill_nudge_signals.py tests/agent/test_skill_nudge_signals.py
git commit -m "feat(nudge): SignalEvaluator implementing S1-S4 from spec

S1 repeated pattern, S2 novel CLI, S3 user phrase, S4 resolved repeated
error.  Pure logic; persistence handled by skill_usage_tracker."
```

---

## Task B3: Hook signal evaluator into per-iteration loop

**Files:**
- Modify: `run_agent.py:1495-1614` (init), `run_agent.py:9295-9299` (per-iter), `run_agent.py:8049, 8372` (resets)
- Test: `tests/run_agent/test_nudge_loop_integration.py`

- [ ] **Step 12.1: Initialize the evaluator in `_init_nudge_state`**

Append to the method written in Task B1:

```python
        from agent.skill_nudge_signals import SignalEvaluator
        self._signal_evaluator = SignalEvaluator(
            repeated_threshold=self._nudge_repeated_threshold,
            error_threshold=self._nudge_error_threshold,
            common_clis_suppressed=self._nudge_common_clis_suppressed,
            cli_window_days=self._nudge_cli_window_days,
            user_phrases=self._nudge_user_phrases,
        )
```

- [ ] **Step 12.2: Observe user message at turn start**

In `run_conversation()` early in the function (find the line where `original_user_message` first becomes available), after it is captured, add:

```python
        if self._nudge_signals_enabled and not self._nudge_disabled:
            try:
                self._signal_evaluator.observe_user_message(original_user_message or "")
            except Exception as e:
                logger.debug("Signal evaluator (user msg) failed: %s", e)
```

- [ ] **Step 12.3: Add a single observation helper and call it from the existing tool-dispatch path**

The agent loop in `run_agent.py` is large; rather than scattering signal-eval calls across multiple iteration sites, add ONE method on `AIAgent` and call it from exactly two existing seams. Add this method near `_compute_should_review_skills` (Task B4):

```python
    def _observe_tool_activity(
        self,
        tool_name: str,
        arguments: dict | None,
        result_content: str | None,
    ) -> None:
        """Feed one (call, result) pair into the signal evaluator. Best-effort."""
        if not getattr(self, "_nudge_signals_enabled", False):
            return
        if getattr(self, "_nudge_disabled", False):
            return
        ev = getattr(self, "_signal_evaluator", None)
        if ev is None:
            return
        try:
            ev.observe_tool_call(tool_name, arguments or {})
            err_text: str | None = None
            content_str = str(result_content) if result_content else ""
            head = content_str[:200].lower()
            if content_str.startswith("Error:") or "error" in head or "traceback" in head:
                err_text = content_str
            ev.observe_tool_result(tool_name, error_text=err_text)
        except Exception as e:
            logger.debug("Signal evaluator failed for tool=%s: %s", tool_name, e)
```

Hook the method into the agent loop at exactly one place: the existing site where a parsed tool call is dispatched and the tool result content has just been computed. Search for the line that builds the `{"role": "tool", "tool_call_id": ..., "content": ...}` dict in `run_agent.py` (there are two such sites — one for the streaming path and one for the non-streaming/Anthropic path; use `grep -n '"role": "tool"' run_agent.py` to locate them). Right after `content` is finalized for that message, before it is appended to `messages`, insert:

```python
                self._observe_tool_activity(
                    tool_name=tool_name,
                    arguments=parsed_args if isinstance(parsed_args, dict) else None,
                    result_content=content if isinstance(content, str) else None,
                )
```

Use whatever local names the surrounding code already has for the parsed tool name, the parsed arguments dict, and the tool result string. The wrapping try/except in `_observe_tool_activity` ensures a wrong local name fails silently rather than breaking the turn.

- [ ] **Step 12.4: Verify by smoke**

This step has no dedicated unit test — the wiring is exercised by Task B4's gate test plus Task B7's end-to-end test. Confirm import-clean with:

Run: `python -c "from run_agent import AIAgent; print(AIAgent._observe_tool_activity)"`
Expected: prints the bound method, no import error.

- [ ] **Step 12.5: Commit**

```bash
git add run_agent.py
git commit -m "feat(nudge): observe tool calls/results into SignalEvaluator

Hooks signal evaluation into the existing per-iteration loop with try/except
guards so signal-eval failures never break a turn."
```

---

## Task B4: End-of-turn gate update + thread signals into background review

**Files:**
- Modify: `run_agent.py:12176-12204` (end-of-turn gate)
- Modify: `run_agent.py:3014-3060` (`_spawn_background_review` signature)
- Test: `tests/run_agent/test_nudge_end_of_turn_gate.py`

- [ ] **Step 13.1: Write failing tests**

```python
# tests/run_agent/test_nudge_end_of_turn_gate.py
import pytest


def _agent_with_state(monkeypatch, **kwargs):
    """Build a minimally-initialized AIAgent for gate-only testing."""
    from run_agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    a.valid_tool_names = {"skill_manage"}
    a._skill_nudge_interval = kwargs.get("interval", 50)
    a._iters_since_skill = kwargs.get("iters", 0)
    a._nudge_disabled = kwargs.get("disabled", False)
    a._nudge_signals_enabled = kwargs.get("signals_enabled", True)
    from agent.skill_nudge_signals import SignalEvaluator
    a._signal_evaluator = SignalEvaluator(
        repeated_threshold=3, error_threshold=2,
        common_clis_suppressed=[], cli_window_days=30, user_phrases=[],
    )
    if kwargs.get("fired"):
        a._signal_evaluator.fired_signals.update(kwargs["fired"])
    return a


def test_gate_fires_when_signal_present(monkeypatch):
    a = _agent_with_state(monkeypatch, fired={"S1"})
    from run_agent import AIAgent
    res = AIAgent._compute_should_review_skills(a)
    assert res == (True, {"S1"})


def test_gate_falls_back_to_time_when_no_signal(monkeypatch):
    a = _agent_with_state(monkeypatch, iters=50, interval=50)
    from run_agent import AIAgent
    res = AIAgent._compute_should_review_skills(a)
    assert res == (True, set())


def test_gate_blocked_by_disable(monkeypatch):
    a = _agent_with_state(monkeypatch, fired={"S1"}, disabled=True)
    from run_agent import AIAgent
    res = AIAgent._compute_should_review_skills(a)
    assert res == (False, set())


def test_gate_blocked_when_skill_manage_unavailable(monkeypatch):
    a = _agent_with_state(monkeypatch, fired={"S1"})
    a.valid_tool_names = set()
    from run_agent import AIAgent
    res = AIAgent._compute_should_review_skills(a)
    assert res == (False, set())
```

- [ ] **Step 13.2: Run, expect failure**

Run: `python -m pytest tests/run_agent/test_nudge_end_of_turn_gate.py -v`
Expected: FAIL — `_compute_should_review_skills` doesn't exist.

- [ ] **Step 13.3: Add the helper and rewrite the gate**

In `run_agent.py`, add as an `AIAgent` method:

```python
    def _compute_should_review_skills(self) -> tuple[bool, set[str]]:
        """End-of-turn gate from spec §5.2.

        Returns (should_review, fired_signals). The signals set is empty when
        the time fallback fires.
        """
        if "skill_manage" not in self.valid_tool_names:
            return False, set()
        if getattr(self, "_nudge_disabled", False):
            return False, set()

        signals: set[str] = set()
        ev = getattr(self, "_signal_evaluator", None)
        if getattr(self, "_nudge_signals_enabled", False) and ev is not None and ev.fired_signals:
            signals = set(ev.fired_signals)
            ev.clear()
            self._iters_since_skill = 0
            return True, signals

        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval):
            self._iters_since_skill = 0
            return True, set()

        return False, set()
```

In the existing block at `run_agent.py:12176-12182`, replace:

```python
        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills, _triggered_signals = self._compute_should_review_skills()
```

In the `_spawn_background_review` call at line ~12198, pass the signals:

```python
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                    triggered_signals=_triggered_signals,
                )
            except Exception:
                pass  # Background review is best-effort
```

In the `_spawn_background_review` definition (line 3014), accept and use the new arg:

```python
    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
        triggered_signals: set[str] | None = None,
    ) -> None:
        """... (docstring unchanged) ..."""
        import threading

        if review_memory and review_skills:
            base_prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            base_prompt = self._MEMORY_REVIEW_PROMPT
        else:
            base_prompt = self._SKILL_REVIEW_PROMPT

        if review_skills and triggered_signals:
            signal_hint = (
                "\n\nThe following nudge signals fired this turn: "
                + ", ".join(sorted(triggered_signals))
                + ". Consider whether they suggest a reusable skill."
            )
            prompt = base_prompt + signal_hint
        else:
            prompt = base_prompt

        # ... rest unchanged ...
```

- [ ] **Step 13.4: Run tests**

Run: `python -m pytest tests/run_agent/test_nudge_end_of_turn_gate.py -v`
Expected: green.

- [ ] **Step 13.5: Commit**

```bash
git add run_agent.py tests/run_agent/test_nudge_end_of_turn_gate.py
git commit -m "feat(nudge): signal-first end-of-turn gate

Replaces the time-only trigger with the four-branch gate from spec §5.2 and
threads the fired signals into the background reviewer prompt so its
suggestion can reference *why* it fired."
```

---

## Task B5: `/skills nudge off|on` CLI intercept + slash-help

**Files:**
- Modify: `cli.py:5816-5819`
- Test: `tests/cli/test_skills_nudge_slash.py`

- [ ] **Step 14.1: Write failing test**

```python
# tests/cli/test_skills_nudge_slash.py
import types
import pytest


class _FakeAgent:
    def __init__(self):
        self._nudge_disabled = False


def test_skills_nudge_off_sets_flag(monkeypatch):
    from cli import HermesCLI  # adapt to real entry-point class name
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()
    cli._handle_skills_command("/skills nudge off")
    assert cli.agent._nudge_disabled is True


def test_skills_nudge_on_clears_flag(monkeypatch):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()
    cli.agent._nudge_disabled = True
    cli._handle_skills_command("/skills nudge on")
    assert cli.agent._nudge_disabled is False


def test_other_skills_subcommand_delegates_to_hub(monkeypatch):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _FakeAgent()
    called = {}
    def fake_handle(cmd, console):
        called["cmd"] = cmd
    monkeypatch.setattr("hermes_cli.skills_hub.handle_skills_slash", fake_handle)
    cli._handle_skills_command("/skills list")
    assert called["cmd"] == "/skills list"
```

- [ ] **Step 14.2: Run, expect failure**

(`HermesCLI` is the class — defined at `cli.py:1793`; `_handle_skills_command` lives at `cli.py:5816`.)

Run: `python -m pytest tests/cli/test_skills_nudge_slash.py -v`
Expected: FAIL — intercept does not exist.

- [ ] **Step 14.3: Edit the dispatcher**

Replace `cli.py:5816-5819` with:

```python
    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command.

        Special-cases /skills nudge off|on so it can mutate the active agent's
        per-session nudge state; everything else delegates to hermes_cli.skills_hub.
        """
        parts = cmd.strip().split()
        # parts[0] is "/skills"; subcommand at parts[1]
        if len(parts) >= 3 and parts[1].lower() == "nudge":
            sub = parts[2].lower()
            if sub == "off":
                if getattr(self, "agent", None) is not None:
                    self.agent._nudge_disabled = True
                print("Skill creation nudges silenced for this session.")
                return
            if sub == "on":
                if getattr(self, "agent", None) is not None:
                    self.agent._nudge_disabled = False
                print("Skill creation nudges re-enabled for this session.")
                return
            print("Usage: /skills nudge [off|on]")
            return

        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, ChatConsole())
```

- [ ] **Step 14.4: Run tests**

Run: `python -m pytest tests/cli/test_skills_nudge_slash.py -v`
Expected: green.

- [ ] **Step 14.5: Update slash-help**

In `hermes_cli/skills_hub.py:_print_skills_help()` (find the function, around the existing /skills help block), append:

```python
    _console.print("  /skills nudge [off|on]    silence or re-enable creation nudges this session")
```

- [ ] **Step 14.6: Commit**

```bash
git add cli.py hermes_cli/skills_hub.py tests/cli/test_skills_nudge_slash.py
git commit -m "feat(nudge): /skills nudge off|on CLI intercept

Lets the user silence creation nudges for the current session without
restarting.  Other /skills subcommands continue to flow through the hub
router unchanged."
```

---

## Task B6: Update `cli-config.yaml.example` (Phase B keys)

**Files:**
- Modify: `cli-config.yaml.example` (skills section)

- [ ] **Step 15.1: Append the nudge_signals block**

Add after the existing `index_token_budget` line in the skills section:

```yaml
  # Phase 2 opt-in: replace the time-based creation nudge with signal-based
  # triggers. When enabled, the time counter (creation_nudge_interval) becomes
  # a fallback that fires only after this many tool iterations without any
  # signal. Set creation_nudge_interval: 0 to disable the fallback entirely.
  nudge_signals:
    enabled: false
    # S1: trigger when the same (tool, arg-signature) appears N+ times in a turn
    repeated_pattern_threshold: 3
    # S2: window for "novel CLI" detection — CLIs not seen for this many days
    novel_cli_window_days: 30
    # S2: CLIs that should NOT count as novel even if not in the known-CLI file
    common_cli_suppressions:
      - git
      - python
      - python3
      - node
      - npm
      - pnpm
      - uv
      - pytest
      - rg
      - sed
      - cat
      - ls
      - mkdir
    # S3: user-message phrases that fire an explicit "remember this" signal
    user_phrases:
      - "next time"
      - "remember"
      - "from now on"
      - "记一下"
      - "下次"
      - "以后"
    # S4: a successful call after this many same-error failures fires a signal
    error_repeat_threshold: 2
```

- [ ] **Step 15.2: Commit**

```bash
git add cli-config.yaml.example
git commit -m "docs(config): document skills.nudge_signals block"
```

---

## Task B7: Per-session disable end-to-end test

**Files:**
- Test: `tests/run_agent/test_nudge_disable_e2e.py`

- [ ] **Step 16.1: Write the test**

```python
# tests/run_agent/test_nudge_disable_e2e.py
import os
import pytest


def test_env_var_disables_nudge_at_init(monkeypatch):
    monkeypatch.setenv("HERMES_SKILL_NUDGE_DISABLE", "1")
    from run_agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    AIAgent._init_nudge_state(a, {"skills": {"nudge_signals": {"enabled": True}}})
    assert a._nudge_disabled is True


def test_disabled_session_blocks_signal_gate():
    from run_agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    AIAgent._init_nudge_state(a, {"skills": {"nudge_signals": {"enabled": True}}})
    a._nudge_disabled = True
    a.valid_tool_names = {"skill_manage"}
    a._iters_since_skill = 999
    a._skill_nudge_interval = 50
    a._signal_evaluator.fired_signals.add("S1")
    res = AIAgent._compute_should_review_skills(a)
    assert res == (False, set())
```

- [ ] **Step 16.2: Run**

Run: `python -m pytest tests/run_agent/test_nudge_disable_e2e.py -v`
Expected: green.

- [ ] **Step 16.3: Commit**

```bash
git add tests/run_agent/test_nudge_disable_e2e.py
git commit -m "test(nudge): per-session disable via env var and slash command"
```

---

# Phase 3 wrap-up (documentation only — flag flips happen in a follow-up PR)

## Task C1: Add Phase 3 wrap-up doc

**Files:**
- Create: `plans/skills-prompt-budget-and-nudge-phase-3-checklist.md`

- [ ] **Step 17.1: Write the checklist**

```markdown
# Phase 3 Wrap-up Checklist

After ≥1 minor version of dogfooding with `skills.index_v2: true` and
`skills.nudge_signals.enabled: true` set in personal configs:

- [ ] Flip default of `skills.index_v2` to `true` in `cli-config.yaml.example`
- [ ] Flip default of `skills.nudge_signals.enabled` to `true` in `cli-config.yaml.example`
- [ ] Bump default of `skills.creation_nudge_interval` from `15` to `50`
- [ ] Update README / docs / website skills section to reference v2 rendering
- [ ] Add TUI parity: `/skills nudge off|on` in `ui-tui/src/app/slash/commands/ops.ts`
- [ ] Add gateway parity: same intercept in the gateway's slash dispatcher
- [ ] After one minor version with v2 default, remove the v1 rendering path
  (`_render_v1` in `agent/prompt_builder.py`) and the `index_v2` flag
  plumbing entirely — Phase 4 cleanup.
```

- [ ] **Step 17.2: Commit**

```bash
git add plans/skills-prompt-budget-and-nudge-phase-3-checklist.md
git commit -m "docs(plans): Phase 3 wrap-up checklist"
```

---

# Final Verification

- [ ] **Step 18.1: Full test suite**

Run: `python -m pytest tests/agent/test_skill_inventory.py tests/agent/test_skill_usage_tracker.py tests/agent/test_skill_nudge_signals.py tests/agent/test_prompt_builder.py tests/tools/test_skill_manager_tool.py tests/tools/test_skill_describe.py tests/cli/test_skills_nudge_slash.py tests/run_agent/test_nudge_signals_init.py tests/run_agent/test_nudge_end_of_turn_gate.py tests/run_agent/test_nudge_disable_e2e.py tests/integration/test_skills_index_v2_e2e.py -v`
Expected: all green.

- [ ] **Step 18.2: Full repo test suite for regressions**

Run: `python -m pytest -q`
Expected: all green. If any unrelated test fails, investigate before opening the PR — the refactor in Task A1 touched a hot path.

- [ ] **Step 18.3: Manual dogfood — Phase A**

```bash
# Add to ~/.hermes/config.yaml under skills:
#   index_v2: true
#   index_token_budget: 2000

# Run a normal agent session and inspect logs for prompt size:
hermes --debug 2>&1 | grep -i "skills prompt\|## Skills"
```

Expected: the system prompt shows the v2 format with category-grouped names
and a small number of full-description critical entries.

- [ ] **Step 18.4: Manual dogfood — Phase B**

```bash
# Add to ~/.hermes/config.yaml under skills:
#   nudge_signals:
#     enabled: true

# In a chat session:
# 1. Repeat a tool 3+ times: confirm a background skill review fires after the response
# 2. Type "remember next time" in a message: confirm signal fires
# 3. Run an exotic CLI not seen before: confirm a signal fires once it succeeds
# 4. Force the same error twice and resolve: confirm S4 fires
# 5. Run /skills nudge off and confirm subsequent triggers are silenced
```

- [ ] **Step 18.5: Open the PR**

```bash
git push -u origin feat/skills-index-v2-and-nudge-signals
gh pr create --title "feat(skills): two-tier index + signal-based nudge (gated)" \
    --body "$(cat plans/skills-prompt-budget-and-nudge-redesign.md | head -30)

See plans/skills-prompt-budget-and-nudge-redesign.md for the full spec and
plans/skills-prompt-budget-and-nudge-implementation.md for the task breakdown.
Both behaviors are gated behind feature flags (skills.index_v2,
skills.nudge_signals.enabled); defaults flip in Phase 3."
```
