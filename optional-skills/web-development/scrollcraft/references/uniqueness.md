# The structure axis

## 1. The template trap

Four sites were built with this skill: a protein coffee brand, a personal brand,
a landscape design-build firm, and an agent observability product. Four
industries, four worlds, one light canvas and three dark. The owner looked at
them side by side and said they felt like a template. He was right, and the
evidence is in the files.

All four open with a full-bleed `scrub` under a fixed minimal top bar carrying a
wordmark and one CTA. All four anchor the hero headline in the lead corner with
a greet cue and kinetic lines. All four run a pinned type act where lines
crossfade. All four hand off to a flow section, pan a card rail with a
`data-sc-tilt="6"` on each card, and close on `data-sc-act="pin"` with
`data-sc-span="1.15"`, `data-sc-spotlight` on the stage and
`data-sc-magnet="0.26"` on the CTA. All four land between 13.6 and 13.8
viewport-heights across 6 or 7 acts with exactly one accent colour.

What actually varied was the order of the middle acts and the palette.

The device kit is an **aesthetic** axis. It changes how a page looks. It has no
opinion on what a page *is*, so every build reached for the same shape, because
the shape was never a decision anybody made.

> The world changes how a page LOOKS. The grammar changes what a page IS.
> A build that only changes world is a re-skin.

This file is the structure axis. Read it after the interview and before the
score table. It has three parts that are not optional: pick a **grammar**,
invent a **signature move**, and pass the **fingerprint gate**.

---

## 2. Page grammars

A grammar is the page's organising logic: what a section is, what the chrome is
for, how the visitor knows where they are, and what the ending is. Two pages in
the same grammar will feel related no matter how far apart their palettes are.
That is the whole finding above.

Each grammar below names what it **forbids**. The forbids are the point. They
are what stops a build drifting back to the filmic default halfway through,
which is what happens when a grammar is a preference instead of a constraint.

Pick one. Do not blend two: a chaptered page with a continuous world underneath
is a filmic one-shot with extra headings.

---

### 2.1 Filmic one-shot

The original skeleton, and now one choice among eight rather than the house
style.

**Fits:** a single linear argument with one emotional arc. Consumer products,
launches, anything where the visitor should feel carried rather than
navigating.

**The scroll feels like:** a film you are pushing through. Continuous, no seams,
each act handing off before the last has left.

**Forbids:** visible sequence (chapter numbers, an index, a progress readout);
hard cuts between grounds; any chrome that implies the page is a tool; more than
one entry point. If the visitor can jump, it is not one shot.

**Nav, hero, close:** fixed minimal bar, wordmark and one CTA. Full-bleed scrub
hero, corner-anchored kinetic headline on a greet cue. Pinned close with a
spotlight and a magnetic CTA.

**Leans on:** `scrub`, `pin`, `drift`, `kinetic`. **Bans:** nothing structural,
which is exactly why it is the default drift and why four builds landed here.

**Use it when the interview earns it, and say in the report why the other seven
did not fit.** This grammar now carries a burden of proof the others do not.

---

### 2.2 Chaptered editorial

The page is a printed feature. Chapters are the unit, not acts.

**Fits:** long-form substance. A method, a manifesto, a founder story, a
research-backed product, anything where the visitor should feel they read
something rather than watched something.

**The scroll feels like:** turning pages. Full-stop intertitles between
chapters, then dense asymmetric spreads. Hard cuts, not crossfades. Each chapter
lands on its own ground and stays there.

**Forbids:** `drift` as a continuous gradient (each chapter is a hard change of
ground, not an interpolation); the full-bleed scrub hero; pinned crossfade type
acts; a magnetic CTA; centred hero copy. Media never bleeds under type here, it
sits in its own column with a caption.

**Nav, hero, close:** no fixed bar. A folio in the margin, chapter number and
title, updating as chapters pass. The hero is a **title page**: type on the
paper ground, no media above the fold, the media starts in chapter one. The
close is a colophon or masthead plate, small type, the CTA set as a line of
running text rather than a button island.

**Leans on:** `flow` + `in`, `reveal` at chapter boundaries, `parallax` inside a
media column, `count` for real figures inside prose. **Bans:** `scrub` beyond
one chapter, `spotlight`, `magnet`.

---

### 2.3 Live surface

The page behaves like the product. Not a screenshot of it, and not a div-built
fake: the actual surface, running, with scroll driving its state.

**Fits:** software, tools, dashboards, editors, anything where the demo is the
argument. If the honest pitch is "watch what it does", this is the grammar.

**The scroll feels like:** operating something. Panels populate, a log fills, a
graph advances, a selection moves. The visitor is inside the thing.

**Forbids:** marketing chrome of any kind. No wordmark-plus-CTA bar, no scrims,
no full-bleed photography, no kinetic headline stacks, no hero claim laid over
footage. Copy lives in the surface's own idiom: labels, tooltips, empty states,
status lines, a help panel. A section heading in 6rem display type breaks this
grammar instantly.

**Nav, hero, close:** app chrome replaces nav. A sidebar, a tab strip, a status
bar, a breadcrumb, whatever the real product would have, and it is real enough
to be the navigation. The hero is the surface already in a state, not a title.
The close is an **actual input**: a command line, a field, a first-run step,
something the visitor puts a cursor in. A magnetic button is the wrong ending
for a page that spent its whole length being a tool.

**Leans on:** `pin` (the surface holds while state advances), `count` on real
telemetry, pointer devices where the real product would have them, and `--sc-p`
driven CSS for anything the kit does not cover. **Bans:** `scrub`, `kinetic`,
`spotlight`, `drift` past two stops.

**The honesty rule.** taste.md forbids fake dashboards and fake terminals, and
that rule is not suspended here. The surface has to be real markup running real
logic on real or clearly-labelled sample data. That labelled-sample escape is
how a concept product can still use this grammar: every panel is operable
markup computing its state from data arrays in the page, and the page says on
its face that the scenario is a demo. What stays banned is the painting of a
surface, an image or dummy divs posing as something that runs. If the panels
cannot actually compute, the grammar is unavailable. Pick another.

---

### 2.4 Continuous world

One canvas, fixed for the entire scroll, and the page travels through it.
Waypoints, not sections.

> **This grammar REQUIRES worldflight mode.** `data-sc-mode="worldflight"`, one
> fixed stage, one spacer, legs that crossfade. See references/worldflight.md.
>
> Building it out of pinned acts is not a lesser version of this grammar, it is
> a different and worse page, and it has already been tried. The owner's verdict
> on the act-based attempt: "awful... you're literally going from scrolling down
> to static page and then you start scrolling down again... weird clear page
> lines scrolling up... very cheap looking." Every one of those is the same
> defect. A pinned act is a block in the document; a document made of blocks has
> seams; and a world with seams is not a world. Do not reach for `scrub` acts
> here, however long you make the spans.

**Fits:** a journey with real geography. A supply chain, a process with physical
stages, a place, a build, anything where "where you are" is meaningful.

**The scroll feels like:** moving through a single space that never cuts. The
visitor never leaves the frame.

**Forbids:** section boundaries of any kind. No `sc-section` blocks, no acts at
all, no second stage, no `drift` steps (one continuous grade across the whole
travel, authored into the world, not interpolated between legs). Nothing may
scroll *over* the canvas: copy arrives inside it, at waypoints, in the fixed
copy layer. The only element in document flow is the spacer.

**Nav, hero, close:** the nav is a **map**. A waypoint list, a depth readout, a
position marker, and it is clickable, because a world you cannot skip around in
is a video. The hero is an establishing position inside the world, not a
separate title stage. The close is arrival at a place in the same canvas, and
the CTA is an object in that place.

**Leans on:** worldflight legs with `data-sc-linger`, copy windows against the
whole track, the `sc:waypoint` event driving a rail the page draws itself.
**Bans:** every act device, `flow`, `pan`, hard cuts, `src` swapping.

**This is the expensive one.** The chain warning in SKILL.md applies: a single
unbroken flight is the most fragile thing you can build, and the seam law in
worldflight.md section 6 is not optional. Choose this grammar only when the
brief is literally about travel through a place, and budget for the reroll.

---

### 2.5 Typographic poster

Type is the imagery. Media is minimal or entirely absent, and scale contrast
does every job that photography would have done.

**Fits:** a brand whose asset is a sentence. Manifestos, agencies with strong
verbal identity, launches with one claim, anything where a stock-looking image
would weaken the page rather than support it. Also the right answer when there
are no good assets and generating them would produce eight plausible, forgettable
frames.

**The scroll feels like:** words arriving at wildly different weights. A word at
40vw, then a paragraph at 16px, then silence. Rhythm comes from scale, not
motion.

**Forbids:** photographic ground, `scrub`, scrims (nothing to scrim), cards of
any kind, and decorative motion. If a device is doing the work instead of the
typography, the grammar has already failed.

**Nav, hero, close:** the wordmark is set as part of the composition, at
composition scale, not as a 14px bar item. There may be no persistent nav at
all. The hero is a single word or one line at extreme scale, filling the
viewport, with a real `<h1>` behind it. The close inverts the whole page: the
smallest type on the site, the CTA as a plain underlined link, quiet after all
that volume.

**Leans on:** `kinetic` (this is the one grammar where character splitting can
be right), `pin` with scale driven from `--sc-p`, `reveal` as a wipe across
letterforms, `drift` doing heavy lifting because the ground is most of the
frame. **Bans:** `scrub`, `pan` rails of cards, `tilt`, `parallax` on text.

**The typography floor doubles here.** taste.md caps display at ~6rem outside a
hero moment. This grammar is one continuous hero moment, so the cap lifts, but
the tracking, measure and optical-correction rules tighten: at 40vw, default
tracking is a visible defect and one bad kern is the whole page.

---

### 2.6 Gallery / catalog

Objects in a walkable collection. Museum labels, not marketing copy.

**Fits:** a range. Products with variants, a portfolio, a menu, a materials
library, case studies, anything where the visitor's real question is "what are
the options" rather than "should I believe you".

**The scroll feels like:** walking a room. Lateral drift with vertical scroll,
objects entering and leaving at their own pace, each one labelled with fact
rather than pitch.

**Forbids:** the argument-shaped pinned type act; a single hero claim; scrim
copy over media; persuasion in the object labels. A label reads
`Cedar. Air-dried 18 months. Kiln-finished.` and not `Craftsmanship you can
feel.` Every object gets the same label schema, no exceptions, because the
schema is what makes it a collection instead of a grid.

**Nav, hero, close:** the nav is an **index of objects**, and it jumps. The hero
is object one, already in view, already labelled, with no separate title
treatment: the collection starts at the top of the page. The close is either the
last object or an inquiry plate typeset exactly like a label, so the ask reads
as part of the collection.

**Leans on:** `pan` as the spine rather than as one act, `reveal` per object,
`tilt` on objects the visitor would pick up, `count` for real specs. **Bans:**
`kinetic` headlines, `spotlight`, `magnet`, more than one `scrub`.

**The rail copy constraint from devices.md §3 becomes structural here**, not a
caveat. Labels are read cropped for most of their life, so the schema has to
survive being half-visible.

---

### 2.7 Split stage

Two columns held in tension for the whole page, resolved by scroll.

**Fits:** any argument with two sides. Before and after, cost and saving, manual
and automated, what you have and what you would have. The comparison is the
product.

**The scroll feels like:** watching a balance tip. Both halves are always
present, both move, and the page is going somewhere specific: the moment one
side wins.

**Forbids:** full-bleed anything before the resolve; centred copy; the
corner-anchored hero; a symmetric close. Neither column may be decorative, both
carry real content the whole way down. The instant one side becomes a caption
for the other, this collapses into a zigzag layout with extra steps.

**Nav, hero, close:** no bar. The **divider is the chrome**, and it carries the
labels for both sides plus the progress of the argument. The hero establishes
the split at 50/50 on the first screen, with both headlines readable at once, so
the visitor understands the format before they scroll. The close is the
**collapse**: the divider travels to one edge, one column takes the full width,
and the CTA lives in the winning column. That collapse is the ending, and it
should be the single most satisfying moment on the page.

**Leans on:** `pin` with divider position driven from `--sc-p`, `reveal` per
side, `count` for the comparison figures if they are real. **Bans:** `pan`,
`spotlight`, `magnet`, more than one `scrub`, `drift` (two grounds, one per
side, and they hold).

---

### 2.8 Rhythmic cutlist

Short hard-cut acts at speed. No pinning, no dwell, no crossfades.

**Fits:** energy brands. Streetwear, sport, events, music, drinks, youth
products, anything where the visitor should feel a pulse rather than follow an
argument.

**The scroll feels like:** a cut every second. Twelve to twenty short sections
rather than six long ones, each one landing whole and gone. Total page length
stays inside the 8 to 14 viewport-height budget precisely because nothing is
held.

**Forbids:** any act over ~1.4 viewport-heights; `data-sc-dwell` above 0.1;
`pin` entirely; overlapping cue windows; slow easing. This grammar is the exact
inverse of the filmic one-shot: where that one hides its seams, this one is
made of them.

**Nav, hero, close:** the bar is loud, not minimal. Full-width, high-contrast,
possibly a marquee, possibly the CTA at the same weight as the wordmark. The
hero is one screen that cuts to the next in under a viewport, so there is no
settling shot and no greet-and-hold. The close is abrupt: the last cut is the
CTA, at full bleed, no spotlight, no drift-down.

**Leans on:** `flow` + `in` at short stagger, `reveal` on nearly every section,
`count` if the figures are real, hard `drift` steps between adjacent grounds.
**Bans:** `pin`, `spotlight`, `magnet`, `dwell`, `parallax`.

**The taste floor still applies at speed.** Fast is not an excuse for a
1.2 second entrance that the reader outruns. Cue windows here are short *and*
front-loaded, so a section is fully legible within the first third of its own
span.

**The peak problem, and how to resolve it.** This grammar bans `pin` and `dwell`
outright while feel.md insists the peak gets the most scroll room and the
biggest hold. Those pull in opposite directions, and the quiet failure is a
build that reaches for `pin` at its peak and still calls itself a cutlist.

**Hold in the fixed chrome layer, and keep every act short and unpinned.** The
loud bar this grammar already asks for is a persistent element that does not
belong to any act, so it can unfurl, run a long choreography and hold as long as
the peak needs while the acts underneath keep cutting at full speed. Drive it
from page scroll rather than from an act's `--sc-p`, since the whole point is
that it outlives the act it started in. The airfield build's departures board
runs its entire peak (unfurl, populate, cascade, reveal, hold, collapse) in
the chrome, with no pinned act anywhere on the page and nothing over 1.3vh.

The general form: **when a grammar bans the device your peak wants, move the
peak out of the act stack rather than breaking the grammar.** The bans are on
what the acts do, not on what the page can do.

---

## 3. The signature move

Every build must invent **one bespoke interaction that exists on that site
alone**. Not in the device kit, not in any prior build, not a parameter change.
Coded in the page, with `data-sc-*` attributes of your own naming or plain
inline JS reading `--sc-p`. The engine stays untouched, always.

This is the thing that makes a page memorable after the visitor closes the tab,
and it is the only part of a build that cannot be arrived at by following rules.

### What counts

- **Scroll-as-playhead over a persistent trace rail.** A thin horizontal trace
  fixed at the bottom edge, present the whole page, drawing a real waveform or
  route or timeline. Scroll position is the playhead. Passing an act stamps a
  marker on the trace that stays. By the footer the trace is a complete record
  of what the visitor just went through, and it doubles as navigation.
- **A wordmark the pointer can pull apart.** The letters follow the cursor with
  different masses, separate under a drag, and settle back into perfect lockup
  when released. Only on the hero, only once, and the settle has to be exact.
- **A line drawing that builds itself.** An SVG technical illustration whose
  `stroke-dashoffset` is driven from `--sc-p`, so scrolling literally draws the
  object, then the dimension lines arrive, then the callouts. Pairs with the
  technical-drawing world in worlds.md.
- **A running receipt.** A small fixed panel that accumulates a line every time
  the visitor passes a claim, with real numbers, so the close arrives with a
  totalled ledger of the argument they just read. Only works with real figures,
  which is the check on it.
- **One control that regrades the whole page.** A time-of-day handle, a
  temperature, a load level: one input, and every image, ground and accent on
  the page shifts together. It has to affect everything at once or it is a
  widget.

### What does not count

- A recoloured spotlight. A spotlight at a different radius. Two spotlights.
- `data-sc-tilt="9"` instead of `6`. Any parameter change to any kit device.
- A different easing curve on kinetic lines.
- Five cards in the rail instead of three, or the rail scrolling the other way.
- A third `scrub` act. More of a device is not a new device.
- Something the engine already does, given a project-specific class name.

The test: **describe the move to someone who has seen the other builds. If they
cannot tell it apart from something the kit already does, it is not a signature
move.** Reaching for a kit parameter here is the same failure as reaching for
the filmic default in §2, one level down.

---

## 4. The fingerprint gate

The registry lives at `<workspace>/FINGERPRINTS.md`, where `<workspace>` is
whatever `node <skill>/scripts/workspace.mjs` prints. It is per-user and it
starts empty: the gate is about not repeating **yourself**, so your first build
has nothing to clear and every build after it does.

A worked twelve-row registry ships as `EXAMPLES.md` in the upstream scroll-craft repository *(upstream repo — not vendored in this port)*. Read it to see what a filled table looks like and which shapes tend
to collide. It is illustration, not constraint: those are somebody else's
builds and they do not gate yours.

**Before building:** read it. Every row is a shape that is now taken.

**Before writing markup:** check the planned build against every existing row on
these six dimensions.

| # | Dimension | What it records |
|---|---|---|
| 1 | Grammar | Which of §2, or a named new one |
| 2 | Nav treatment | What the chrome is and what it is for |
| 3 | Hero device | What the first screen does |
| 4 | Act-sequence shape | The device order, act count, total viewport-heights |
| 5 | Close pattern | How the last screen behaves and what the CTA sits in |
| 6 | Signature move | The one bespoke interaction, in a phrase |

**The gate: a new build must differ from EVERY existing row on at least 4 of the
6.** Not 4 of 6 on average across the table. Four against each row, individually.

Dimension 6 is free, because a signature move is unique by definition. So the
gate really asks for three more out of the remaining five, against each row, and
a build that changes only grammar and world will fail it.

**If the planned build fails the gate, change the plan, not the log.** Rewriting
a fingerprint row to make a new build fit is the one thing that makes this file
worthless. It is a record of what exists, not a description of what you wish
existed.

**After shipping:** append one row. Fill all six dimensions plus world and port.
Say plainly what it shares with prior rows, because the shared columns are what
the next build has to avoid.

---

## 5. Aesthetic range

Premium-minimal is a choice. It is not the costume this skill wears by default,
and four dark-or-paper pages with one accent each is what happens when nobody
decides otherwise.

The full range is available when the brand's vibe asks for it:

| Family | Reads as | Earned by |
|---|---|---|
| Brutalist | Blunt, structural, unstyled on purpose | Tools, infrastructure, anything anti-marketing |
| Maximalist | Dense, layered, loud, generous | Culture brands, events, food, anything abundant |
| Playful | Bouncy, coloured, informal | Kids, games, consumer apps, community |
| Retro | Specific to a decade, not vaguely nostalgic | Heritage brands, music, anything with a real lineage |
| Dense | Information-forward, small type, high count | Data products, catalogues, reference, finance |
| Editorial | Paper, folios, measure, restraint | Long-form substance |
| Premium-minimal | Quiet, dark, one accent, air | Luxury, and only when asked for |

Go where the interview points. If the human says "loud" and the page comes back
in charcoal with one accent, the interview was decorative.

**What does not flex:** the taste floor. Spacing scale and rhythm, type metrics
and measure, contrast ratios measured on the render, motion built from
`transform` and `opacity`, focus-visible on everything, reduced motion that
keeps meaning, real copy and real numbers. Every item in taste.md holds in every
aesthetic family.

A brutalist page still needs 4.5:1 body contrast. A maximalist page still needs a
spacing scale, and needs it more, because density without rhythm is just noise. A
playful page still cannot animate `top`. **The floor is what separates a chosen
aesthetic from a sloppy one**, and it is the reason range is safe to offer at
all.

Two specific traps stay banned in every family, because they are not aesthetics,
they are defaults with a look: the cream-and-brass artisan palette
(taste.md, Colour) and violet-to-blue AI gradients. Both are what a page reaches
for when nobody chose.
