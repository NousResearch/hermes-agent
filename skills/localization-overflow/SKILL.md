---
name: localization-overflow
description: Measure per-project width and row budgets to avoid overflow.
version: 0.1.1
author: Kuruki (flabanabba8), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game-localization, overflow, row-budget, width, text]
    related_skills: [game-localization-studio]
---

# Localization Overflow Skill

## What it does

Before writing translated rows, measure the target game's text-box budgets from
its own data, then keep every translated row inside those budgets. The
localization plugin counts **rows, not pixels**, and the default window does
**not** wrap long lines, so width and row count are hard limits, not soft
preferences. This skill measures both per project and wraps accordingly.

It does not translate for you; it sets the dimensional guardrails that
`game-localization-studio` then fills. It is not for prose editing, tone, or
placeholder handling — those belong to that skill.

## Why budgets differ per project

There is no universal char count or row count. Each game's window is sized
differently, so hardcoding a number (e.g. "35 chars", "4 rows") from one game
and reusing it in another produces false overflows or false safety. Derive the
budget from the project's own source rows — the author already fitted the
widest row on screen, and the row count is the window the game ships.

## When to Use

- Translating or reviewing game text where rows must not overflow the box.
- Deciding how many characters fit per line or how many rows a message may span.
- Checking an existing draft for rows that are too wide or too tall.
- The opposite of `game-localization-studio`'s translate step: set limits first.

Do not use for: tone/register, keeping placeholders `{0}`/`<tag>`, romanization
conventions, or building the translation itself. Those are that skill's job.

## The two budgets

**WIDTH (chars per row)** = the widest ORIGINAL (source) row in the unit's
file, in characters. The author fitted that many chars on screen, so a
translated row may not be wider. English glyphs are narrower than CJK, so this
budget is generous for English but is the proven value. Measure it **per file**
(per message group) — different windows have different widths; a single global
max is wrong.

**HEIGHT (rows)** = the number of rows the window shows. A message is ONE
sentence the author split across its rows; a translated message must stay within
that same row count and never repeat a sentence across two rows. Measure the
max source row count across the project; per unit, keep English rows
<= source rows.

## How to measure (run this script)

Read `scripts/measure_dimensions.py` from this skill and run it against the
draft:

```
terminal(command="python3 <skill>/scripts/measure_dimensions.py <draft.json>")
```

It prints per-file WIDTH, the global WIDTH/HEIGHT limits, and the width/height
distribution so you can see whether budgets are uniform or window-specific. For
the NeetGirlsLife Unity Mono draft this yields WIDTH=105 for
`textasset:text`, and per-file widths down to single-digit chars for small UI
labels; HEIGHT maxes at 20 rows with most units at 1-3.

## Wrapping procedure

1. For each unit, split the source into the author's rows (real newlines in
   `source`).
2. Translate the message as a whole, then place the target text back row by row.
3. For each translated row, keep its length <= that file's WIDTH. If a row is
   too long, shorten it or break it only while the whole message stays within
   HEIGHT.
4. Keep total rows <= the source row count. Do not split one source row into two
   target rows, and do not collapse two source rows into one if it widens past
   WIDTH.
5. Never repeat a sentence across two rows; a row must not trail into the next
   with a dangling conjunction.
6. Use escaped backslash-n (`\n`, two characters) for line breaks in the stored
   translation — never a real newline (chr 10).

## Pitfalls

- Measuring one global max and applying it everywhere: per-file widths vary
  widely (105 down to single digits in the same draft).
- Treating WIDTH as a soft preference: the default window does not wrap, so a
  row wider than the widest source row overflows.
- Counting pixels or font size: the plugin counts rows, so a "fine" pixel width
  can still overflow its row budget.
- Splitting one source row across two target rows to shorten each line: this
  raises the row count past HEIGHT and overflows the bottom.
- Using a real newline instead of escaped `\n`: the stored translation breaks.

## Verification

After wrapping, re-run `measure_dimensions.py` and confirm no unit's stored
translation has a row wider than its file's WIDTH or more rows than its source.
Run `localize_validate` from `game-localization-studio`; if it flags a wide row,
fix that row and validate again. Report which rows were caught and fixed.

## Verified state (this draft)

`measure_dimensions.py` is confirmed working. For the NeetGirlsLife Unity Mono
draft the measured budgets are **WIDTH=105 chars**, **HEIGHT=20 rows** (per-file
WIDTH ranges 2–105; most units are 1–3 source rows). The full translation is
imported and `localize_validate` reports `structural_ok: true`, 3424/3424
translated, 0 untranslated, 0 blocked; the only flags are English rows wider
than their Japanese source rows (expected JP→EN length growth), all inside the
generous ~1.4× allowance.
