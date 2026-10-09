# system.md — the multi-screen register

`build` makes one excellent surface. `direct` makes a film. Neither survives a product: an app, a
dashboard, an admin, a docs site, anything with routes. Not because the taste changes — the taste
core is identical — but because **the unit of design changes and so does the failure mode.**

| | build / direct | system |
|---|---|---|
| unit of design | the section / the scene | the **component × state** |
| failure mode | boredom — it looks like every other AI page | **drift** — screen 7 invents a fourth button |
| how you find it | eyes on one page | crawl every route and diff what was painted |
| the peak | exactly one wow moment | **none. A settings screen with a wow moment is a bug** |

The last row is the one people get wrong. In this register there is no peak, and hunting for one
produces a dashboard with a hero animation nobody wants to sit through twice. What replaces it is
**coherence** — a visitor should be unable to tell which screen the team cared about most — plus at
most one *moment of care*: an empty state that is genuinely helpful, a table that does something
smart, a keyboard flow that feels designed. A **failure** state is a legitimate answer here and often
the best one: a wall display that never blanks when its data goes stale, keeping the numbers and
draining the colour out of them and stamping their age, is more care than any animation. Say
"care" and people hear "delight", and delight walks you straight back toward a peak. Quiet is the
goal; quiet is harder.

## Routing here

Take this register when the brief has more than one screen and they must feel like one product:
app, dashboard, admin, settings, onboarding, a docs or content site with real navigation, or a
marketing site big enough to have sections that outlive the launch. If it is a single page, use
`build`. If it is a launch page that has to stop the scroll, use `direct`. If you are already in
`build` and the second screen appears, stop and come here — half a system is worse than either.

## Process

**1. Intake.** Product · audience · the screens that exist (name them) · who uses it how often (a
tool used daily wants less personality than one used monthly) · brand constraints · stack.

**2. Recon** (`recon.md`, bounded). Product UI has its own reflex table and it is not the landing
page one — for this register the moodboard matters less and the *mechanics* matter more. Scout
products, not campaigns.

**3. The system-sheet — this is the gate.** Copy `templates/SYSTEM-SHEET.md` into
`design/SYSTEM-SHEET.md` and fill it before any markup. Two halves:

- **the route map** — every screen, its job in one line, its layout family, and what it shares with
  the shell. A route nobody can describe in one line is a route nobody designed.
- **the component inventory** — every control the product needs, with its variant count declared up
  front: `button: primary / ghost / danger` and nothing else. This number becomes the budget
  `systemscan` enforces later. Declaring four button variants and shipping nine is the whole disease.

**GATE:** every route has a one-line job; every component in the inventory has a named, justified
variant count; every variant has its full state row filled (below). Building before this is filled
is how you get a product that needs a redesign at screen five.

**4. Commit-sheet** (SKILL.md, all six fields). Field 1 "Peak" becomes **the moment of care** — the
one place the product is more than correct. Field 5 "Motion budget" shrinks hard: an app earns
transitions, not choreography. State the durations and stop.

**5. Tokens, then the shell, then screens.** Tokens as OKLCH custom properties before any component.
Then the persistent shell (nav, sidebar, header, the page frame) — it is on every screen, so its
mistakes are on every screen. Then screens in order of *traffic*, not of interest: the boring
high-traffic one sets the patterns, and building the exciting one first guarantees the rest look
like afterthoughts.

**6. States are the product.** Every interactive component ships every row that applies:

| state | the rule that gets broken |
|---|---|
| default | — |
| hover | gate it in `@media (hover:hover)`, or touch devices get sticky hover |
| **focus-visible** | must be *designed* and visible on every control. `outline: none` without a replacement is a defect, and `systemscan` fails on it — it presses Tab and looks |
| active / pressed | a button with no pressed state feels broken before it feels ugly |
| disabled | must be distinguishable without relying on colour alone, and must say *why* if the reason is not obvious |
| loading | in place, not a full-page spinner; preserve layout so nothing jumps |
| empty | the highest-value screen in most apps and the one most often skipped — say what this is, why it is empty, and the one action that fills it |
| error | next to the thing that failed, in words, with the recovery action |
| selected / current | navigation without a current state is a map with no "you are here" |

Empty, loading and error are not edge cases in a product; they are the first thing a new user sees.

**7. Density is a decision, not an accident.** Tables stay tables — sticky header, `tabular-nums`,
right-aligned numerals, a real sort affordance — never card-ified into unscannability. Decide rows
per screen deliberately. Charts route to the `dataviz` skill, which owns that craft; do not improvise
a palette for series colour here.

> **The sticky-header trap.** Every dense table also needs `overflow-x: auto` on narrow viewports,
> and that container **silently becomes the scrollport, which kills `position: sticky` on the head**.
> It still looks perfectly correct in a screenshot, and in a `--full` capture the head even appears
> stuck. Verify by measuring `getBoundingClientRect().top` after a real scroll, not by looking. The
> head also has to be offset by the height of any sticky page chrome above it; `top: 0` parks it
> underneath your own header.

**8. Verify** (`verify.md` + the gate below).

**9. Lock the contract.** `design/DESIGN.md` in this register is not a style note, it is the
**component contract**: the inventory with its final variant counts, the state recipes with real
values, and the rule that a new variant is a design decision that edits this file first. Every later
edit (the `edit` route) starts here.

## The gate: systemscan

```bash
node scripts/systemscan.mjs http://localhost:3000 --routes /,/settings,/billing,/team
node scripts/systemscan.mjs <url1> <url2> <url3> --max-variants 4
```

It crawls every route, reads what the browser **actually painted**, and reports the system as built
rather than as documented. A token in the stylesheet that never renders is not part of the system; a
one-off inline style is.

**Budgets are per kind:** `--max-variants button=5,select=1,toggle=3,4` — a bare number sets the
default for everything unnamed. Pass the numbers from your system-sheet, not one global figure.

**What counts as a variant** is a *painted signature*: `background | colour | border-colour |
border-width | radius | font-size/weight | padding | shadow`. A small button and an icon button are
separate variants whether or not you named them — write the budget knowing that. Note also that
`<a class="btn…">` is classified as a **button**, not a link.

**A state is not a variant.** `[disabled]` / `aria-disabled`, `aria-current` / `aria-selected`, and
any control sitting inside an element carrying a non-default `data-state` are counted and printed
separately, never against the budget. The reason is that the state matrix above is compulsory: a
disabled secondary button is *supposed* to paint differently, and a run number is supposed to invert
inside a late row. Charging those against the variant budget would mean a product with no disabled
state scores better than one that implements it properly — the gate would be pushing against its own
doctrine. Give your states real hooks (`data-state`, `aria-*`) and the gate reads them as states;
paint a fifth button by hand with no hook and it is drift, correctly.

It fails on:
- **a control type over its variant budget** — the number you declared in the system-sheet;
- **a route that returned an error status or rendered nothing measurable** — a mistyped route list
  used to make the gate quieter instead of louder, which is the wrong direction for a gate;
- **an interactive element with no visible focus state** — it presses Tab, reads what changed, and a
  control that paints identically focused and unfocused is a fail. (It drives real keypresses because
  `:focus-visible` is a heuristic that programmatic focus does not reliably trigger, and because
  tabbing skips disabled controls for free.)

It warns on:
- **a variant used exactly once** — either promote it into the system or delete it; a one-off is
  drift with a nice reason attached;
- **a colour, type step or radius that appears on exactly one route** — that is where the system is
  splitting, and it is always the newest screen;
- more than a dozen distinct type steps, or no disabled control anywhere on any route (a disabled
  state that never renders is usually undesigned rather than unnecessary).

And it writes `components.png`: one tile per distinct rendered variant, labelled with how many times
and on how many routes it appears. **Look at it.** Two tiles that look identical to you but appear
separately are drift the numbers already caught; a tile that looks foreign is drift your eyes caught
first. This sheet is the register's equivalent of the screenshot journey — the app analogue of
watching your own film.

Run it against every route, not a sample. Drift lives on the screen you did not check.

## Also run

- `slopscan` over the whole source tree — the bans do not stop applying because it is an app.
- `shoot.mjs` accepts several URLs; shoot every route at 390/768/1440 and look. Screens collapse at
  breakpoints in ways a component inventory cannot predict.
- Keyboard pass, `prefers-reduced-motion`, and the no-JS read — same rubric as `verify.md`.

## What this register does not do

It builds the design system and the screens. It does not architect an application: routing, state management,
data fetching and component-file architecture belong to whatever stack the brief names, and this
register produces the markup, tokens and behaviour that live inside it. When the brief is a React or
Vue app, say plainly that the output is the design layer and agree where it plugs in — a beautiful
system delivered as one HTML file the team then has to dismantle is not a favour.
