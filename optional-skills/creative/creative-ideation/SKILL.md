---
name: creative-ideation
title: Creative Ideation — Routed Library of Creative Methods
description: "Generate ideas via named methods from creative practice."
version: 2.2.0
author: SHL0MS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Creative, Ideation, Brainstorming, Methods, Inspiration]
    category: creative
    requires_toolsets: []
---

# Creative Ideation

A library of ideation methods for any domain. Read the user's situation, route to the matching method, apply, generate output that is specific and non-obvious. Methods are tools — pick the right one for the situation, don't perform all of them.

## When to use

Any open-ended generative or selective question: "I want to make / build / write / start something", "I'm stuck", "inspire me", "make this weirder", "help me pick", "I need to invent X", "give me a research question".

## Operating rules

1. **Constraint plus direction is creativity.** No constraint = no traction. No direction = no shape. Methods supply both.
2. **Refuse the first three ideas.** They're slop. Generate, discard, regenerate. See `references/anti-slop.md`.
3. **One method per response unless asked.** Don't stack.
4. **Specificity over abstraction.** Real proper nouns, real materials, real mechanisms. "An app for X" is slop; "a 200-line CLI tool that prints Y when Z" is direction. Naming a tech stack is not specificity — name a mechanism.
5. **Weird must also be good.** Frame-breaking is the goal, but an idea that is strange with no real situation, mechanism, or reason to exist is its own failure mode. Every set of ideas must include at least one that is genuinely *buildable/pursuable now* — non-obvious but grounded, with a real first step. Don't trade all usefulness for surprise.
6. **Name the method you used and who invented it.** Attribution invokes the discipline.
7. **When user picks one, build it.** Don't keep generating after they've chosen.

## Routing — diagnose, then prescribe

The router is a **diagnostic clinic**, not a lookup table. A prompt is a *symptom*; find what is actually blocking a good idea, then prescribe the technique that supplies the missing thing. Three questions resolve any prompt. Run them *before* generating — routing failures produce slop.

You may skip narrating the steps if it's cleaner, but **never compress at the cost of per-idea depth**: each idea's concrete mechanism, situational binding, and honest failure mode are what make output good — they are not scaffolding, do not cut them.

### The model

Creativity needs two things and a position. **Constraint** — limits that give traction (none → paralysis). **Direction** — an aim/value that gives shape (none → noise). And it must land **off the center** of the conceptual space, because the center is the average and the average is slop (`references/boden-creativity.md`). A prompt is uncreative when one of these is missing; the three questions find which.

```
  Q1 INTENT  ─►  Q2 BOTTLENECK  ─►  Q3 MOVE (Boden)  ─►  PRESCRIBE
  what they      what's actually     what kind of         one method,
  want, surface  blocking a good     novelty closes       or a principled
  + underlying   idea right now      the gap              set (≤2)
```

### Q1 — INTENT · what does the user actually want?

Read the surface request **and** the underlying need.

- **Move** — generate / expand / select / unblock / subvert / refine / synthesize  (→ PHASE)
- **Object** — text / object / artifact / system / self / research / product  (→ DOMAIN)
- **Givens** — how much is already on the table: none / domain / project / problem  (→ SPECIFICITY)
- **Value sought** — what "good" means here: novel / useful / beautiful / rigorous / weird / personal. This is the *direction*; name it — it breaks ties.
- **Surface vs underlying** — high-slop requests ("startup ideas", "a habit app") carry an unspoken *"…but not the obvious one."* The underlying intent **overrides** the surface request.

**PHASE** — what stage is the user in?

| Phase | Cues |
|---|---|
| **GENERATING** | "give me an idea", "what should I make", "inspire me", no idea yet |
| **EXPANDING** | "what else", "more like this", "give me variations" — has a base idea |
| **SELECTING** | "help me pick", "which should I do", "I have these options" |
| **UNBLOCKING** | "I'm stuck", "blocked", "going in circles", "stale" — has material |
| **SUBVERTING** | "make it weirder", "less obvious", "this is too safe" |
| **REFINING** | "this is fine but missing something", "feels rough" |
| **SYNTHESIZING** | "I have a pile of notes / interviews / observations" |

**DOMAIN** — what is the user making/doing?

| Domain | Cues |
|---|---|
| **TEXT** | fiction, essay, poem, lyric, script, copy |
| **OBJECT** | visual art, music, sound, performance, installation, sculpture |
| **ARTIFACT** | software, hardware, mechanism, device |
| **SYSTEM** | org, civic, institution, ecology, community |
| **SELF** | life decision, career, personal practice |
| **RESEARCH** | paper, thesis, scholarly question |
| **PRODUCT** | business, market, service |

**SPECIFICITY** — how much constraint is in the prompt?

| Level | Cues |
|---|---|
| **NONE** | "I'm bored", "inspire me" — no domain, no project |
| **DOMAIN** | "I want to write something" — knows the field, no project |
| **PROJECT** | "I'm working on this specific X" |
| **PROBLEM** | "I have this specific friction within X" |

### Q2 — BOTTLENECK · what is blocking a good idea right now?

The diagnosis: find the **one** binding constraint. (Two at once → see Prescribe.)

| Bottleneck | Looks like | What it needs |
|---|---|---|
| **No constraint** | blank page, "anything", no traction | a constraint (exploratory) |
| **No direction** | aimless, "I don't know what I want" | clarify the aim, then select |
| **Frame lock** | too safe, obvious, "been done", high-slop | break a defining constraint (**transformational**) |
| **Disconnection** | has material but stale, no fresh angle | an outside link (combinational) |
| **Lost contact** | stuck mid-work, going in circles | a random relocation (exploratory / unblock) |
| **Overload** | too many options, can't choose | a selection test |
| **Raw pile** | notes / observations, no structure | synthesis |

### Q3 — MOVE · what kind of novelty closes the gap? (Boden)

The bottleneck names the **family** (`references/boden-creativity.md`): **combinational** (new links between existing ideas) / **exploratory** (search unvisited regions of the space) / **transformational** (change a defining constraint of the space). Combinational is cheapest, exploratory the workhorse, **transformational the strongest — and the one slop never reaches.** On slop terrain, aim transformational; exploratory variation on a slop seed stays slop.

### Prescribe — one technique, or a principled set

The move picks the **family**; PHASE + DOMAIN pick the **method** (tables below). Prescribe a **set** only when *two things are missing at once*, and stack exactly one technique per missing thing — never three:

- constraint **+** frame-lock ("weird startup ideas") → `jobs-to-be-done` **+** `lateral-provocations`
- generate **+** choose → `volume-generation` → `premortem-and-inversion`
- disconnection **+** no structure → `derive-and-mapping` → `affinity-diagrams`
- contradiction **+** needs an analog → `triz-principles` → `biomimicry`

Stacking to cover a weak diagnosis is an anti-pattern: bad pick + bad pick ≠ good pick.

### Overrides — fire first, they beat the diagnosis

- **User names a method** → use it.
- **User asks "which method?"** → surface 2–3 candidates, one line each, ask. Don't silently default.
- **Mood word** ("weird / strange / surprising / less obvious") → transformational: `references/methods/lateral-provocations.md` or `references/methods/pataphysics.md`, regardless of domain.
- **High-slop terrain** ("AI / startup / habit-tracker / productivity / wellness / fitness / food / travel app") → frame-lock by default → `references/methods/lateral-provocations.md` / `references/methods/pataphysics.md`; refuse the first **5** ideas, not 3.

### Route — move picks the family, phase + domain pick the method

**By phase (applies regardless of domain):**

| Phase | Default route |
|---|---|
| GENERATING + SPECIFICITY=NONE | `references/full-prompt-library.md` **General** section (constraint dispatch) |
| GENERATING + DOMAIN known | route by domain (next table) |
| EXPANDING | `references/methods/scamper.md` |
| SELECTING | `references/methods/premortem-and-inversion.md` (or `references/methods/compression-progress.md` for upside) |
| UNBLOCKING | `references/methods/oblique-strategies.md` |
| SUBVERTING | `references/methods/lateral-provocations.md` (fallback `references/methods/pataphysics.md`) |
| REFINING (text) | `references/methods/defamiliarization.md` |
| REFINING (other) | `references/methods/creative-discipline.md` (Tharp's spine) |
| SYNTHESIZING | `references/methods/affinity-diagrams.md` |
| Volume needed fast | `references/methods/volume-generation.md` |

**By domain (when GENERATING with DOMAIN known):**

| Domain | Default route |
|---|---|
| TEXT — formal / poetry | `references/methods/oulipo.md` |
| TEXT — narrative | `references/methods/story-skeletons.md` |
| TEXT — has source material to remix | `references/methods/chance-and-remix.md` |
| OBJECT (music, visual, performance) | `references/methods/oblique-strategies.md` |
| OBJECT — physical maker / wants a starting constraint | `references/full-prompt-library.md` **Physical / object** section |
| ARTIFACT — wants a starting constraint | `references/full-prompt-library.md` **Software / artifact** section |
| ARTIFACT — engineering invention with parameter conflict | `references/methods/triz-principles.md` |
| ARTIFACT — software architecture | `references/methods/pattern-languages.md` |
| ARTIFACT — has natural-system analog | `references/methods/biomimicry.md` |
| ARTIFACT — accumulated assumptions to question | `references/methods/first-principles.md` |
| SYSTEM (civic, org, institutional) | `references/methods/leverage-points.md` |
| SYSTEM — collective / participatory | `references/full-prompt-library.md` **Social / collective** section |
| SELF (life, career, what-to-study) | `references/methods/derive-and-mapping.md` |
| RESEARCH — picking a question | `references/methods/compression-progress.md` |
| RESEARCH — attacking a known problem | `references/methods/polya.md` |
| PRODUCT (business, service) | `references/methods/jobs-to-be-done.md` |
| Need to break a frame / find analogy | `references/methods/analogy-and-blending.md` |

### Resolve ambiguity

- **Multiple paths plausible** → pick the one closest to the user's phrasing, not the most impressive method.
- **Genuinely ambiguous** → ask ONE question (intent or object), don't guess. *"Generating ideas, or choosing between ones you have?"* / *"Fiction, essay, or something else?"*
- **Signals contradict** (e.g. "weird startup ideas" = product **+** frame-lock) → that is a *two-things-missing* case → stack two and say so: *"`jobs-to-be-done` for the framing + `lateral-provocations` to break the obvious shape."*
- **No match** → constraint dispatch (`references/full-prompt-library.md`) is the safe fallback.
- **Same question again** → switch method; variation in method = variation in idea distribution.

### Anti-default check (run before generating)

- About to write "Here are 5 ideas:" or a bare numbered list? → STOP. Pick a method first.
- About to default to generic LLM-mode brainstorming? → STOP. Pick a path above.
- Output looks like what an unrouted LLM would produce? → routing failed, redo.

The default LLM mode is exactly what this skill exists to displace. If you generate without routing, you've defeated the skill.

For deeper edge cases (mood signals, stacking, anti-patterns) see `references/heuristics.md`.

## Output format

For the constraint-dispatch default path:

```
## Constraint: [Name] — from [Source]
> [The constraint, one sentence]

### Ideas

1. **[One-line pitch]**
   [2-3 sentences — what specifically is made, why it's interesting]
   ⏱ [weekend/week/month]  •  🔧 [stack/medium/materials]

2. ...
3. ...
```

For other methods, use the format the method specifies (TRIZ produces a contradiction analysis; OuLiPo produces constrained text; Oblique Strategies produces a single applied card → next move). Don't force every method into the constraint template.

**Every idea set, regardless of method:**
- Name the method used. On slop terrain, name the obvious ideas you refused.
- Give each idea its concrete mechanism and its honest failure mode / tradeoff / who-it's-for. This depth is what makes ideas land — measured, not decorative.
- Mark at least one idea as the **grounded** one — buildable/pursuable now, non-obvious but with a real first step. The others can run further toward the strange; this one has to be genuinely doable. Don't let the whole set be weird-but-impractical.

## File map

- `references/boden-creativity.md` — the routing theory: novelty+value, the conceptual space, and the three creative moves (combinational / exploratory / transformational) that pick the method family.
- `references/full-prompt-library.md` — constraint library, sectioned by domain (General, Software, Physical, Social, Lists). Default path for SPECIFICITY=NONE.
- `references/method-catalog.md` — one-line summary + when-to-use per method
- `references/heuristics.md` — extended decision tree for edge cases
- `references/anti-slop.md` — anti-slop rules; apply to every output
- `references/exercises.md` — time-boxed exercises (5min / 30min / 1hr / day / week)
- `references/methods/` — 22 named methods, one file each, load only the one you're using

## Attribution

Constraint-dispatch core adapted from [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html). Methods drawn from primary sources cited in each method file.
