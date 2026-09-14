# Control Center — enterprise UI/UX transformation

Companion to `docs/PHASE_UI_ENTERPRISE_AUDIT.md`, which was written before any redesign code.

This document separates two things that are easy to conflate and were kept apart deliberately:

- **Visual quality** — what the dashboard looks like. Judged from rendered screenshots.
- **Functional validation** — what was actually measured, against a running server and real
  API responses, with the measurement itself checked.

Where something was not proven, it says so.

---

## 1. Scope and constraints honoured

| Constraint from the brief | Outcome |
|---|---|
| Preserve the existing CSP | **Unchanged, byte for byte.** `default-src 'self'; script-src 'self'; style-src 'self'; …` still in `nova/control/server.py:64` and still on the live response. No relaxation was needed — see §7. |
| Preserve the stdlib server / build architecture | Unchanged. `nova/control/server.py` and `api.py` have **zero diff** this phase. |
| Do not move into the upstream `apps/*` workspace | The app stayed at `nova/control/ui/`, building to `nova/control/static/`. |
| Do not modify Hermes core | Not touched. Patch budget still 1 row in root `AGENTS.md`. |
| No fabricated data | No metric, trend, rollup or figure is invented. §6 lists the places where the redesign chose "—" over a plausible-looking number. |
| Accessibility not sacrificed | 0 contrast failures across 2,176 measured text nodes; every tab stop has a ring. §5. |
| Avoid heavy animation libraries | `framer-motion` **removed**; all motion is CSS. Bundle fell 449 kB → 368 kB. |

---

## 2. Files changed

**Added**

```
nova/control/ui/src/components/glass.tsx     171   surface primitives, status pills, empty/skeleton
nova/control/ui/src/components/panel.tsx     154   Panel/PanelBody (4 states), ErrorState, MetricCard
nova/control/ui/src/components/shell.tsx     293   nav, atmosphere, top bar, ⌘K command bar
nova/control/ui/src/components/tooltip.tsx    62   Radix wrappers, Hint, InfoDot
nova/control/ui/src/lib/state.ts             155   state→label mapping, time, initials
nova/control/ui/src/screens/agents.tsx       384   agent cards + agent detail workspace
nova/control/ui/src/screens/misc.tsx         746   the other eight screens
nova/control/ui/src/screens/types.ts               API response types
docs/PHASE_UI_ENTERPRISE_AUDIT.md            166   the pre-change audit
```

**Modified** — `src/app.tsx` (288), `src/index.css` (268), `src/lib/hooks.ts`, `package.json`.

**Deleted** — `src/components/bits.tsx` and the five `src/components/ui/*` shadcn primitives, all
superseded. `bits.tsx` still imported `framer-motion` after it was dropped from dependencies.

**Not changed** — every Python file. The redesign consumes the existing API and adds no endpoint.

---

## 3. Design system

Centralised in `src/index.css` as tokens; no screen hard-codes a colour.

- **Four surface levels** — `--glass-1` (raised), `--glass-2` (elevated), `--glass-3` (solid /
  overlay), plus the page canvas. `.glass::before` adds a top highlight; `.lucent::after` is a
  pointer-following radial.
- **Semantic state colours** — `--running`, `--waiting`, `--blocked`, `--info`, `--neutral`.
  State is *never* colour-alone: every pill carries a word, and dots sit beside a label.
- **Three ink tiers** — `--ink`, `--ink-muted`, `--ink-faint`, spaced so all three clear WCAG AA
  on the *glass* surfaces they sit on (the lighter, binding case), not merely on the canvas.
- **Atmosphere** — two slow drifting bloom layers (44s / 52s), the only decorative animation.
- **Fallback** — `@supports not (backdrop-filter: blur(1px))` swaps in opaque surfaces.

### Motion

All CSS. `pulse-running` and `pulse-waiting` mark live state; `.interactive` gives a 180 ms lift;
`.shimmer` drives loading skeletons. Pointer-aware lighting writes two custom properties
(`--px`, `--py`) on mousemove and lets CSS do every repaint — no per-frame React render, no 3D
tilt, no cursor trails, no particles.

`prefers-reduced-motion: reduce` removes the bloom drift entirely and collapses the rest.
**Verified, not assumed:** under that media query `document.getAnimations()` returns `0`.

---

## 4. Navigation and screens

Ten sections: Overview, Agents, Objectives, Work, Approvals, Activity, Knowledge, Channels,
Policies, Usage. Each maps to an endpoint that exists. There is deliberately **no** Integrations
or Organization item — the backend has neither, and inventing them would be inventing capability.

The ⌘K command bar searches only data already loaded in the client; it issues no query the
dashboard could not otherwise answer.

---

## 5. Functional validation

Everything below was run against a live `nova serve` process with real API responses, on two
deployments:

- **Empty** (`nova/examples/acme`, port 8931) — 2 agents, no work, no decisions.
- **Loaded** (port 8932) — 12 declared agents + 1 derived channel variant, 26 board items
  (8 blocked, 6 running, 10 ready, 2 todo), 103 real `policy.decision` audit events, one
  objective in `blocked` with a named blocking step, and a deliberately 100-character agent name.

The loaded fixture is **not mock JSON**. Agents were materialised by `nova apply`; work items were
created by `nova objective submit` plus the runtime's own `kanban_db` API; the 103 decisions were
produced by executing the *real* materialised policy plugin
(`<profile>/plugins/nova-policy/__init__.py`) — the same `pre_tool_call` that runs inside a
worker — and letting it write its own audit rows.

### Test suite

| Environment | Result |
|---|---|
| `.venv` (full deps) | **680 passed** |
| `/tmp/mockenv` (stdlib + PyYAML only) | **680 passed** |

Identical to the pre-change baseline recorded in the audit. No test was changed, skipped or added
to accommodate the redesign.

### Accessibility

Measured in-browser across 10 screens × 2 themes.

| Check | Result |
|---|---|
| Contrast (WCAG AA: 4.5:1 body, 3:1 large) | **0 failures** across 2,176 measured text nodes |
| Keyboard focus rings | **33/33** real tab stops show a 2px ring |
| Skip link | Present, and it is the first tab stop |
| Landmarks | One `<h1>`, one `<nav>`, one `<main>` per screen |
| Unnamed buttons / alt-less images | None |
| Loading announced | `aria-busy` + an `sr-only` "Loading" on every skeleton region |
| Console errors | **None**, on any screen, in either theme |
| Horizontal overflow | None at 1680 / 1280 / 900 px |

> **On the measurement itself.** The first contrast pass reported ~400 failures. That result was
> wrong: the palette is authored in `oklch()` and the checker was reading the first three numbers
> as RGB. It was rewritten to convert OKLCH → OKLab → linear sRGB and to composite translucent
> glass over what is behind it. The corrected pass found real failures clustered on one token,
> which were then fixed (§6). Similarly, a first pass flagged "10 elements with no focus ring" —
> an artefact of calling `.focus()` programmatically, which does not match `:focus-visible`;
> tabbing with a real keyboard showed all 33 stops ringed. Both numbers are reported here only
> after the tool was corrected.

### Performance

| Check | Result |
|---|---|
| Frame pacing, 5 s idle on the heaviest screen (1,187 nodes) | median **16.7 ms**, p95 18.4 ms, worst 35.1 ms |
| Long tasks (>50 ms) | **none** |
| JS heap | **5.8 MB** |
| Elements with a blur | **4 per screen** (two bloom layers, two glass surfaces) |
| Elements actually transitioning | 39–105 |
| …of which transition a **layout** property | **0** — all are paint or compositor properties |
| Bundle | **368.11 kB JS** (113.72 kB gzip), **30.74 kB CSS** (6.73 kB gzip) |

`transition-property: all` appears on nearly every node in a naive count; that is the CSS *initial
value* with a `0s` duration, not a live transition. Only elements with a non-zero
`transition-duration` are counted above.

### Visual matrix covered

Desktop 1680, narrow desktop 1280, tablet 900; dark and light; hover, keyboard focus, command
bar open; loading, empty, error, forbidden; 12 agents, 26 work items, 8 approvals, 103-event
history; a 100-character agent name. 68 screenshots were captured under
`scratchpad/shots/`. **They are not committed** — they are build artefacts of the validation run,
not source.

---

## 6. Defects found by looking at it, and fixed

The redesign was not written and declared done. Rendering it against real data exposed twelve
genuine faults, several of which predate this phase:

1. **The Activity timeline and the Approvals escalation list were silently empty against real
   data.** The UI read `d.outcome`; `GET /decisions` returns `effect` and never has. `Decision`
   was typed `Record<string, any>`, so TypeScript could not catch it. The type is now spelled out
   and the five read sites corrected. With 103 real decisions present, the page had been showing
   "No escalation has happened yet".
2. **Every agent avatar showed the same initials.** `display_name.slice(0, 2)` renders "AC" for
   both "Acme Support Assistant" and "Acme Ops" — tenants prefix agents with the company name, so
   the collision is the normal case, not an edge case.
3. **The theme ignored the operating system.** It defaulted to dark and only moved when the toggle
   was clicked. It now follows `prefers-color-scheme` until an explicit choice is made, and only
   an explicit toggle is persisted (writing the derived value on mount would freeze the first
   observed preference forever).
4. **Metric cards asserted `0` when the API failed.** "0 agents" is a false statement, not a
   missing one. They now show "—" and distinguish *reading…* from *could not be read* from *not
   visible to your role*.
5. **The Activity subtitle claimed "every allow".** The endpoint deliberately omits permitted
   calls. The copy now says so.
6. **Activity filter counts mixed scopes** — chips used whole-log counts above a list that only
   ever holds one page, promising "Refused 75" over at most 80 rows. Counts are now page-scoped,
   with a separate "Showing the 80 most recent of 103" line.
7. **"Working now" counted work items under a caption that said agents.**
8. **`--ink-faint` was below AA in both themes** (3.34:1 light, 3.72:1 dark on glass), and the
   light-mode status colours were worse — amber at 2.71:1. Both re-stepped, with `--ink-muted`
   re-spaced to keep three distinct tiers.
9. **The nav count badge was below AA** at 10px.
10. **The objective badge rendered the raw API value**, so one pill read lowercase `blocked`
    among Title Case pills.
11. **Two useful API fields were dropped on the floor** — `objectives[].blocking` (which step is
    holding an objective up) and `knowledge.index_detail` (why every corpus reads "Not indexed",
    and the command that fixes it). Both are now shown.
12. **The Overview duplicated the Agents screen** and left its right-hand column ragged. It now
    samples six agents with "View all 12", and the row stretches.

---

## 7. Notes carried forward

- **CSP and CSSOM.** `style-src 'self'` blocks parsing inline `style` attributes and `<style>`
  elements, but does not govern CSSOM writes (`el.style.x = …`). React and Radix use CSSOM, so
  the pointer-lighting and the portalled tooltips work under the unmodified policy. No relaxation
  was required, and none was made.
- **Operational finding, unrelated to the UI.** `nova --home X objective submit` writes the work
  board to `$HERMES_HOME/kanban.db` (defaulting to `~/.hermes`), while `nova --home X serve`
  reads it from `X/kanban.db`. With only `--home` set, work submitted through NOVA's own CLI is
  invisible to NOVA's own dashboard; setting `HERMES_HOME` to the same directory reconciles them.
  Recorded here, not fixed — it is a CLI/runtime path question, outside this phase.
- **Tablet nav drops the count badges.** At ≤900 px the sidebar becomes a horizontal row and the
  Agents/Work/Approvals counts are omitted for width. On Overview the numbers are still on the
  metric cards; on other screens they are not. Known, accepted, not fixed.

## 8. What is **not** claimed

- No claim about real browsers other than the bundled Chromium. Testing was Chromium-only.
- No claim about screen-reader behaviour. Landmarks, `aria-busy`, `aria-live`, accessible names
  and focus order were verified structurally; no assistive technology was actually driven.
- No claim about performance on low-end hardware. The numbers in §5 are from this container.
- The contrast figures are computed, not sampled from rendered pixels. Compositing is modelled
  (translucent layers over their ancestors); `backdrop-filter` blur of underlying content is not,
  since blurring averages toward the surrounding surface and does not change the result
  materially for the flat backgrounds used here.
