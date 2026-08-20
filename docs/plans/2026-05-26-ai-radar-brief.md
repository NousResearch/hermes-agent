# AI Radar Brief Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a small local helper that turns manually collected or RSS-style AI/dev/community items into a Joe-style morning radar brief.

**Architecture:** Keep the workflow reversible and standalone under `scripts/`: parse items from JSON/YAML or RSS XML, score them against Joe-relevant themes, and render a Traditional Chinese markdown brief with facts, hypotheses, and actions. Tests import the script module directly and cover parsing, ranking, rendering, and empty-report behavior.

**Tech Stack:** Python stdlib + existing PyYAML dependency, pytest via `scripts/run_tests.sh`.

---

### Task 1: Add tests for item parsing and ranking

**Objective:** Define the expected behavior before implementation.

**Files:**
- Create: `tests/scripts/test_ai_radar_brief.py`
- Create: `scripts/ai_radar_brief.py`

**Step 1: Write failing tests**

Tests should assert:
- RSS XML items are parsed into `RadarItem` objects with title/url/source/published fields.
- Items matching Movement/BD/AI-agent themes rank above unrelated items.
- Rendered markdown starts with `## TL;DR` and includes facts, hypotheses, actions, and source URLs.
- Empty input returns `[SILENT]` when `silent_if_empty=True`.

**Step 2: Run test to verify failure**

Run: `scripts/run_tests.sh tests/scripts/test_ai_radar_brief.py -q`
Expected: FAIL because `scripts/ai_radar_brief.py` does not exist yet.

### Task 2: Implement minimal script API

**Objective:** Make the tests pass with a small, dependency-light implementation.

**Files:**
- Create: `scripts/ai_radar_brief.py`

**Step 1: Implement dataclass + parser helpers**

Functions:
- `RadarItem`
- `parse_rss_xml(xml_text, source)`
- `score_item(item, themes=DEFAULT_THEMES)`
- `rank_items(items, limit=5)`
- `render_brief(items, silent_if_empty=False)`

**Step 2: Run focused tests**

Run: `scripts/run_tests.sh tests/scripts/test_ai_radar_brief.py -q`
Expected: PASS.

### Task 3: Add CLI wrapper and smoke-check it

**Objective:** Make the helper usable from cron/manual workflows.

**Files:**
- Modify: `scripts/ai_radar_brief.py`

**Step 1: Add CLI**

Arguments:
- `--input PATH`: JSON/YAML list of items.
- `--rss PATH_OR_URL`: RSS XML path or URL; repeatable.
- `--limit N`
- `--silent-if-empty`

**Step 2: Verify CLI smoke**

Run: `python scripts/ai_radar_brief.py --input <fixture-json>`
Expected: Markdown brief with TL;DR and sources.

### Task 4: Final verification and PR

**Objective:** Keep nightly work reviewable.

**Step 1:** Run focused test wrapper.

**Step 2:** Commit with `feat: add AI radar brief helper`.

**Step 3:** Push branch to Joe fork and open PR against `NousResearch/hermes-agent:main`.
