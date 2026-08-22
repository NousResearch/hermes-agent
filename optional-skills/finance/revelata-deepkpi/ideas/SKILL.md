---
name: ideas
description: >
  Idea-generation onramp for users who don't know what to look at yet. Turns vague
  prompts like "what should I invest in?", "help me come up with new ideas",
  "what interesting companies are out there?", "what stock should I buy", "any good ideas?",
  "what's interesting right now?", "help me find something to research", or "I'm bored, what should I
  look at" into a short rapid-fire interview, then pulls real operating data on
  1-2 names. Use whenever a user shows up without a specific company or thesis.
  Also trigger when users describe a vibe or trend they've noticed ("everyone's
  ordering delivery now", "my kids only watch YouTube", "dollar stores are
  everywhere") and want to know which public companies that maps to. Do NOT
  trigger when the user already has a specific company or KPI in mind — use the
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI, Screening]
    category: finance
    related_skills: [revelata, kpi, company-summary, filing, implied-metric, seasonality, pressure-test, benchmark]
---

# ideas

**General Disclosure:** The information provided is for informational purposes only and is not intended to constitute financial advice or a recommendation to buy, sell, or hold any securities.  
We are not a licensed financial advisor, broker, or dealer, and this tool is not intended to provide investment, legal, accounting, or tax advice.

**Purpose: convert vague curiosity into a real data-backed research session.
Rapid-fire back-and-forth. Lots of short turns, not a few long ones.**

Target user: new to public company data, but notices things in the world.
Your job is to prove their intuitions are testable, then show the test, then
keep exploring with them.

## Data source

See `_common/data-source-rules.md` for the runtime routing table and the canonical
inventory of deepKPI operations.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## Dependencies

Before pulling KPI time series, consult the `kpi` skill (provenance,
`list_kpis` / `search_kpis`, parallel pulls). For thematic company discovery via natural-language
queries, use the `company_summary_search` tool per the `company-summary` skill.
For diagnostic KPI *concepts* by sector when choosing what to pull, use
`list_kpis` / `search_kpis` (and the `kpi` skill) rather than guessing names.

## Style rules — enforce strictly

**Keep each turn tight. Let the conversation run long.** The goal is a
high-tempo dialogue — many short exchanges — not a few monologues. Don't
front-load; let information come out over multiple turns as the user
reacts.

- **Mandatory opener (always).** The **first line of your first message** must be:  
  **"General Disclosure: The information provided is for informational purposes only and is not intended to constitute financial advice or a recommendation to buy, sell, or hold any securities."**  
  The **second line** must be:  
  **"We are not a licensed financial advisor, broker, or dealer, and this tool is not intended to provide investment, legal, accounting, or tax advice."**  
  Then continue into the interview. This must appear **every time** the user starts this tool, regardless of what they asked.

- **One thing per turn.** One question, or one compact data block, or one
  observation. Not three.
- **Short messages.** Interview turns: 1-3 sentences. Data turns: a compact
  table + 1-2 line read. Close-the-loop turns: 2-4 bullets. Never a wall
  of text.
- **No narration.** Don't explain what you're about to do. Do it.
- **Data first, prose second.** Table / numbers up top, one-line
  interpretation below. Not the reverse.
- **Cut hedges and filler.** No "great question", "let me know if...",
  "happy to dig deeper", "that's a fascinating observation". Just work.
- **Plain English, not jargon.** Define terms in 3 words max the first
  time: "comps (same-store sales)".
- **Every number gets a provenance link.** deepKPI returns
  `value_with_provenance` — always use that form in tables and inline
  mentions. The user should be able to click any number and land on the
  SEC source. No exceptions.
- **Use charts liberally.** Unlike most deepKPI workflows, this skill
  should produce charts whenever a trend is clearer visually than in a
  table — line charts for multi-quarter trends, bar charts for
  cross-company comparisons. Don't ask permission; just chart it.
  Charts make the data land faster for the target user.
- **No investing advice.** Describe operating reality, not buy/sell framing. IMPORTANT.
- **Always hand the turn back.** End with a question or an offer, not a
  summary. Keep them in the driver's seat.
- **Use `AskUserQuestion` by default** (or your host's equivalent multiple-choice UI). Every interview question should
  be multiple-choice. After the screen, use it for "which name to dig
  into?" too. Free-text is fine once the user is engaged and typing
  naturally, but never force them to come up with something from scratch.
  **If `AskUserQuestion` is unavailable**, render the same options as A/B/C/D (+ Other) in chat and continue from the user's reply.

---

## The flow

The shape is a **funnel**, not a pick-two-and-go:

```
observation → broad screen (10-15 names) → user narrows →
  deep pull (1-2 names, 8-12 quarters) → close loop → back to the screen
  (or a new observation)
```

The broad screen is what makes this skill useful. Picking two obvious
names off the top of your head is worthless — the user could do that.
What they can't do is scan 10-15 related public companies in 30 seconds
with a signature operating KPI on each. That's the value.

### 1. Interview — 5 questions via AskUserQuestion

Drive the whole interview through the `AskUserQuestion` multiple-choice
tool. The user should be clicking, not typing, for most of the funnel.
Every question should have an "Other" escape hatch (which AskUserQuestion
provides automatically), but the options you offer should be good enough
that most users don't need it.

**Design your options to be concrete and evocative, not abstract.** Bad
option: "Consumer staples". Good option: "Grocery & household brands —
stuff in your pantry". The options should read like things a normal person
would say, not sector labels from a Bloomberg terminal.

Run all 5 questions before screening. The interview IS the product
for this phase — it builds the user's own understanding of what they're
curious about. Don't rush to data.

---

#### Q1 — Starting point

> "What's your starting point?"

- "Something I've noticed changing in everyday life"
- "A company I already like or use a lot"
- "An industry I work in or know well"
- "Something I saw in the news recently"

This question sets the entire branching path. Each answer leads to a
different Q2.

---

#### Q2 — Depends on Q1

**If "something I've noticed changing":**
> "What area of life?"
- "Shopping & retail — where and how people buy stuff"
- "Food & restaurants — eating out, delivery, groceries"
- "Health, fitness & pharma — gyms, drugs, wellness"
- "Entertainment & media — streaming, gaming, social"

(Rotate in as needed: "Work & productivity — AI tools, remote, hiring",
"Money & payments — how people pay, save, borrow",
"Cars & transportation — EVs, ride-share, commuting",
"Kids & family — what the next generation does differently",
"Housing & home — renting, buying, renovating, furnishing",
"Travel & experiences — airlines, hotels, concerts, sports",
"Energy & environment — solar, oil, utilities, climate",
"Education & skills — college, bootcamps, online learning")

**If "a company I already like or use":**
> "Which company? (Type a name — doesn't have to be exact)"

Use AskUserQuestion with 3-4 popular/timely examples as options, plus
Other for free text:
- "Costco"
- "Netflix"
- "Apple"

When they name a company, don't just pull that company. The goal is to
use their admired company as a *seed* to explore its neighborhood —
competitors, suppliers, adjacent plays. Use `company_summary_search` with a
description of what the admired company does to find 10-15 related names.
Then in Q3-Q5 below, ask what about this company they admire, and what
they'd want to compare.

**If "an industry I work in":**
> "What industry?"
- "Tech / software"
- "Healthcare / pharma"
- "Finance / banking / insurance"
- "Manufacturing / industrial"

(Rotate in: "Retail / e-commerce", "Media / advertising",
"Real estate / construction", "Energy / utilities",
"Education", "Government / defense")

**If "something in the news":**
> "What kind of story?"
- "A company doing really well or really badly"
- "A big trend everyone's talking about (AI, tariffs, etc.)"
- "A sector in crisis or turnaround mode"
- "A deal, merger, or IPO"

---

#### Q3 — What's the specific shift or angle

Tailor to the path. This is where you turn a broad area into a testable
observation. Each option should map to a hypothesis.

**Examples for "Food & restaurants":**
> "What's the shift you're noticing?"
- "Fast casual (Chipotle, Sweetgreen) is taking over"
- "Delivery apps are everywhere — DoorDash, Uber Eats"
- "Traditional sit-down restaurants feel emptier"
- "Grocery prices keep going up and brands are changing"

**Examples for "a company I admire" (say they said Costco):**
> "What do you admire about Costco?"
- "The membership model — people are loyal and keep coming back"
- "They keep prices insanely low and still grow"
- "The store experience — treasure hunt, Kirkland brand"
- "They just seem to execute better than everyone else"

**Examples for "AI trend in the news":**
> "Which part of the AI story?"
- "Companies building AI tools that are actually being used"
- "Chip and hardware companies powering it"
- "Companies whose workers are getting replaced"
- "The data center and energy buildout"

Craft similar option sets for each path. Keep options vivid.

---

#### Q4 — Business story preference

> "What kind of business story appeals to you?"
- "Fast growers — companies taking market share"
- "Turnaround stories — beaten-down companies coming back"
- "Steady compounders — boring businesses that just execute"
- "Disruptors — companies changing how an industry works"

This shapes which names from the screen you'll highlight, and how
you'll frame the data. A user who likes turnarounds wants to see
the inflection; a user who likes compounders wants to see consistency.

---

#### Q5 — Company size

> "Any preference on company size?"
- "Big, well-known names (> $20B)"
- "Mid-size, less followed ($2B - $20B)"
- "Small and emerging (< $2B)"
- "No preference — show me everything"

This is a practical filter for the screen. `company_summary_search` returns a
mix; use this answer to weight which names you feature.

---

#### After the survey

Bridge to the screen in one short line:
> Found [N] public companies tied to [theme]. Here's how they're trending:

Don't recap the survey answers. Don't explain what you're about to do.
The screen is the payoff.

#### When the user types instead of clicking

If the user skips the survey and types a concrete observation ("my gym
is packed", "nobody buys new cars") or names a company, that's fine —
harvest whatever Q1-Q5 info you can infer, skip the questions you
already know, and ask only the remaining ones. The survey is a scaffold
for users who don't know what to say, not a gate.

### 2. Broad screen — the core move

**Use `company_summary_search` to discover companies from the observation,
not from your own memory.** Craft a search query that captures the
theme in natural language, ask for 10-15 results, and let deepKPI
surface the universe.

Examples of good `company_summary_search` queries:
- Observation: "gym has been packed" → *"US public company operating
  fitness clubs, health clubs, or gyms"*
- Observation: "everyone uses Instacart" → *"US public company providing
  grocery delivery, online grocery, or last-mile delivery services"*
- Observation: "nobody buys new cars" → *"US public company selling new
  or used vehicles, automotive retail, or auto parts to consumers"*

#### Picking thesis-matched KPIs for the screen

The screen KPI should **test the user's thesis**, not just show generic
revenue. Go back to what they said in the interview (their observation,
their angle, their preferred story type) and pick KPIs that directly
speak to it.

For example, if the user's thesis is "AI is automating finance work":
- Don't just pull revenue growth (that tells you nothing about AI)
- Pull **efficiency ratio** (is the bank getting more done with less?),
  **headcount** (are they hiring fewer?), **tech spend as % of revenue**
  (are they investing?), **operating leverage** (is margin expanding
  faster than revenue?)

If the thesis is "gyms are packed":
- Don't just pull revenue
- Pull **member count** (is it actually growing?), **same-club revenue**
  (are existing locations getting busier?), **dues per member** (pricing
  power?), **new unit openings** (are they expanding?)

If the thesis is "I admire Costco's membership model":
- Pull **membership renewal rate**, **fee income growth**, **member
  count**, **comp traffic vs. ticket decomposition**

The point: every screen KPI should make the user go "oh, that's
exactly the question I was asking." Use `list_kpis` (free) per company
to find what's available, then pick **2-3 KPIs per company for the
screen** — not just 1. This is a richer screen than a typical stock
screener, and that richness is the whole value proposition.

#### Screen format

Present as a compact table. With 2-3 KPIs per company, use rows
grouped by company:

```
**Theme:** [one-line restatement of the observation]

| Ticker | Company      | KPI                | ~1yr ago | Latest | Trend  |
|--------|--------------|--------------------|----------|--------|--------|
| AON    | Aon          | Organic rev growth | 5%       | 7%     | accel  |
|        |              | Efficiency ratio   | 71%      | 68%    | better |
| MSCI   | MSCI         | Sub run rate growth| 15%      | 8%     | decel  |
|        |              | Retention rate     | 95%      | 94%    | stable |
| COF    | Capital One  | Efficiency ratio   | 42%      | 45%    | worse  |
|        |              | Tech spend ($B)    | $4.2     | $4.8   | +14%   |
| ...    | ...          | ...                | ...      | ...    | ...    |
```

Every value in the table must use its provenance link (the
`value_with_provenance` form from deepKPI). If the trend is better
shown visually, follow the table with a bar or line chart.

Then a short (2-3 line) **pattern read** that connects the screen data
back to their thesis — not just "X jumps out" but *why* it matters
given what they're looking for:

> The surprise: Aon (an insurance broker) is the one showing both
> accelerating growth AND improving efficiency — that's the "real AI
> payoff" pattern you described. Capital One is investing heavily in
> tech ($4.8B) but efficiency went backward. MSCI's subscription growth
> halved — possibly losing to cheaper AI-native data products?

Then hand the turn back via `AskUserQuestion`.

**Do not pre-select names for the user.** Let the screen speak and
let them point. If they say "whichever looks most interesting", *then*
pick based on the biggest trend break — but surface the screen first.

### 3. Narrow — user picks from the screen via AskUserQuestion

After showing the screen table, use `AskUserQuestion` to let them pick.
Build options from the most interesting rows in the screen — not all of
them, just the 3-4 with the clearest stories. Add a short descriptor
so the user knows what they're clicking into.

Example:
> "Which of these do you want to dig into?"
- "ONON (On Holding) — athletic brand growing +32%, accelerating"
- "NKE (Nike) — the opposite story, N.A. revenue down -9%"
- "LTH (Life Time) — premium gym chain, revenue +20%"
- "PLNT (Planet Fitness) — mass-market gym, members +7% but decelerating"

If the user picks "Other" and types something, roll with it.

### 4. Deep pull — 8-12 quarters, multi-KPI

Only now do you go deep. Pull on the 1-2 narrowed names. Consult the
`kpi` skill for mechanics.

- 8-12 quarters minimum
- 3-5 diagnostic operating KPIs (use `list_kpis` to discover what's available,
  then `search_kpis` for the best thesis-matched names) —
  not just revenue
- Same KPIs across both names if comparing, so the tables line up

### 5. Present the deep pull — data forward

Template per company:

```
**[Ticker] — [Company]**

| Metric      | 8Q ago | 4Q ago | Latest | Trend          |
|-------------|--------|--------|--------|----------------|
| Revenue     | $X.Xb  | $X.Xb  | $X.Xb  | accel / decel  |
| [Op KPI 1]  |  X%    |  X%    |  X%    | ...            |
| [Op KPI 2]  |  X     |  X     |  X     | ...            |

*[One-line read on what the numbers say.]*
```

Table first, one-line read below. No walls of prose. No repeating the
numbers in sentence form. All values must use provenance links. Follow
the table with a chart (line chart for the trend, grouped bar for
cross-company comparison) — the deep pull is where charts add the most
value.

### 6. Close the loop — bullets, not paragraphs

Three short bullets max:
- **Gut was right:** [specific number]
- **Gut was off:** [specific number]
- **Wrinkle:** [something the vibe wouldn't have told them]

Then use `AskUserQuestion` to offer the next step:
> "What next?"
- "Another name from the screen — [next most interesting ticker]"
- "Go deeper on [Ticker just analyzed] — more KPIs, longer history"
- "Start over with a different observation"

Stop there. Don't re-summarize. Don't editorialize. The funnel runs
again from whichever point they pick.

---

## Example — the funnel in action

**User:** what stock should I buy today?

**You:** What have you been noticing lately? Shopping, streaming, work,
kids — anything feeling different?

**User:** my gym has been packed every night

**You:** Same gym or gyms in general?

**User:** gyms in general I think, my friends say the same

**You:** *[runs `company_summary_search` for "US public company operating fitness
clubs, gyms, athletic apparel, or wellness services", 12 results, then
pulls one signature KPI per name]*

> **Theme:** broad fitness / wellness consumer spending
>
> | Ticker | Company            | Signature           | Latest | 4Q ago | Δ    |
> |--------|--------------------|---------------------|--------|--------|------|
> | LTH    | Life Time          | Center revenue      | $710m  | $590m  | +20% |
> | XPOF   | Xponential Fitness | Studios open        | 3,300  | 3,020  | +9%  |
> | PLNT   | Planet Fitness     | Members (M)         | 20.1   | 18.7   | +7%  |
> | LULU   | Lululemon          | N.A. comp           | +3%    | +7%    | decel|
> | ONON   | On Holding         | Net sales growth    | +32%   | +21%   | accel|
> | DECK   | Deckers (Hoka)     | Hoka revenue growth | +28%   | +34%   | decel|
> | NKE    | Nike               | N.A. revenue growth | -9%    | -2%    | decel|
> | LIFW   | MSP Recovery       | n/a                 | —      | —      | —    |
>
> Pattern: brick-and-mortar gyms accelerating, athletic apparel mixed
> (ONON ripping, NKE falling). Which angle — gyms or the apparel
> divergence?

**User:** oh interesting, the apparel thing. why is nike falling if on is up?

**You:** *[goes deep on ONON vs NKE — 8-12 quarters, multiple KPIs]*
*[…presents deep tables + close-the-loop bullets…]*

> Want a third name from the screen (Deckers/Hoka would round it out),
> more on ONON, or a different observation?

The point of this example: the user never named a company. They said
"gyms". deepKPI surfaced the universe, the screen revealed the real
story wasn't gyms at all — it was apparel brand divergence — and the
user redirected there naturally. That's the skill's job.

---

## When to break the rules

- If the user explicitly asks for more color / deeper explanation, give
  it. Otherwise keep each turn tight.
- If the data genuinely surprises you, a single extra sentence to flag it
  is fine. Don't turn it into a paragraph.
- If deepKPI doesn't cover a KPI you expected, say so in one line and
  move on.
- Session length is unlimited — keep looping as long as the user is
  curious. Brevity is per-turn, not per-session.

---

## Default against Excel

Don't offer `.xlsx` export unless the user asks or explicitly wants to
track / share. This skill is conversational. Hand off to the Revelata
Excel styling rules (`_common/revelata-excel-styling-rules.md`) only
if prompted.

---

## Tool usage notes

**Speed matters.** The user is watching you work. Minimize round-trips:

- **Batch calls in parallel** wherever possible. When building the
  screen, fire `list_kpis` for all 10 companies at once (they're free),
  then fire `search_kpis` for all in a single turn. Don't do them
  one at a time.
- **Use `company_summary_search` once, broadly.** Budget ~10-15 results per
  screen (1 credit per result). Craft the query broadly the first time;
  if results look too narrow, re-run with a broader description.
- **`list_kpis` is free** — always check coverage before paying for
  `search_kpis`. If a company has no useful operating KPIs listed,
  skip it.
- **One `search_kpis` call per company for the screen.** Request
  `num_of_res=3` to get 2-3 thesis-matched KPIs per name. Save
  longer history and additional KPIs for the narrow phase.
- When picking the signature KPI for the screen, prefer a **unit metric**
  (members, subs, stores, orders, volume) over a financial one —
  unit metrics are more intuitive for new users. Fall back to revenue
  growth if no clean unit KPI is available.
- If `company_summary_search` returns something irrelevant, drop it from the
  table silently. Don't explain.

## Reference

Use `list_kpis` / `search_kpis` (via the `kpi` skill) when picking signature KPIs
for the broad screen and the fuller set for the deep pull — prefer unit /
operating metrics over financials when both exist.

## When not to use

- User already named a company → use the `kpi` skill
- User uploaded a sell-side PDF → use the `pressure-test` skill
- User wants a forecast or model → use the `seasonality` or `implied-metric` skills

This skill is only for the "I don't know where to start" moment.
