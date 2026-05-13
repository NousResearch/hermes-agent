# Content Strategy

How to write, not how to lay out. Good typography with bad content is just "polished mediocrity". This document covers the writing principles for both Chinese and English output. Shared rules come first; language-specific details are called out where they matter.

---

## Core principles (all documents)

### 1. Data over adjectives

- Avoid: "Delivered significant business growth"
- Use: write the specific numbers and deltas

Every sentence should survive the follow-up question "how much, specifically?". If you can't answer, don't write it.

### 2. Judgment over execution

Junior writes "what they did". Mid writes "how they did it". **Senior writes "why they made that call, and what they predicted correctly"**.

- Avoid: "Led the platform build-out"
- Use: write what judgment you made and how it was proven right

### 3. Distinctive phrasing over industry clichés

- Avoid: "Embrace the AI era, pioneer digital transformation paradigms"
- Use: say it in your own words, skip the industry vocabulary

**Distinctive phrasing is memorable**. A line you invented beats a line borrowed from an earnings call. It sounds like a person thinking, not a deck regurgitating.

### 4. Honest boundaries

- If you didn't do it, don't claim it
- If you don't know the exact number, don't invent one. Write a vague but honest magnitude
- Attribute collaborators

### 5. Sources before phrasing

For companies, products, people, launch dates, versions, funding, financials, market data, or technical specs, verify the source before writing. Priority: user-provided source material > official pages / docs / press releases > filings / app stores / repo releases > credible media.

- Do not write "latest", "new", version numbers, or market figures before checking them
- If sources conflict, list the conflict and ask the user instead of choosing one
- If only the magnitude is known, write the magnitude instead of false precision

### 6. Materials serve recognition

Branded documents should first make the subject recognizable, then use decoration and atmosphere with restraint.

- Company / product / project docs should confirm logo, product image, UI screenshot, and brand color before layout
- If a key material is missing, mark the gap or ask the user. Do not fill the page with unrelated imagery
- Physical products prefer official product images; digital products prefer real UI screenshots
- If brand color is unknown, keep kami ink-blue rather than inventing a new color
- **Third-party figures**: when redrawing a paper figure, patent illustration, or official architecture diagram for visual consistency, mark the redraw as `Schematic redrawn` /「示意重绘」in the caption. Do not style a redrawn version to look like the original screenshot. If the figure carries primary evidentiary value (patent, official spec), embed the original with attribution rather than redrawing it

---

## Per-document strategies

### One-Pager

**Single purpose**: the reader grasps the point in 30 seconds.

**Structure**:
1. **Headline** (serif display) + one-line subtitle (sans body)
2. **Metrics** - 3-4 cards, numbers first
3. **Core argument** (1-2 paragraphs)
4. **Key evidence / roadmap** (3-5 short bullets)
5. **Next step / contact** (footer)

**Rules**:
- Length target: English 200-350 words; Chinese 400-600 characters
- All section headlines should work as a standalone outline - reading just the headlines should deliver the gist
- Data must fill 30%+ of the body
- Company / product one-pagers must confirm logo, core screenshot or product image, and source for key metrics
- No opening ceremony ("In recent years, as technology has rapidly evolved...")

### Long Document

**Structure**:
1. **Cover** - big title + subtitle + author + date
2. **Contents** (auto-generated or hand-written TOC)
3. **Executive Summary** (≤ 1 page + 3-5 takeaways)
4. **Body** - chapters that each stand alone as an essay
5. **Appendix / references** (if applicable)

**Rules**:
- Every chapter opens with a "claim paragraph" (2-3 sentences summarizing the argument)
- After long paragraphs (>5 lines), intersperse callouts / quotes / figures to relieve eye fatigue
- Highlight key data / conclusions with `<span class="hl">`
- Chapters with external facts must preserve source cues so readers can distinguish fact, judgment, and inference
- Use "chapter breaks" (blank page + chapter number) between major sections

### Letter

**Structure**:
1. Letterhead (sender info, top right or centered)
2. Date (right-aligned)
3. Recipient salutation (left-aligned)
4. Body (3-5 paragraphs)
5. Sign-off ("Sincerely," / "Best regards,")
6. Signature (serif 500)
7. Enclosures (if any)

**Rules**:
- Minimal - no decorative elements
- Body prefers serif (editorial feel)
- Slightly larger type (11-12pt body) - this will be read, not scanned
- Paragraph spacing ≥ 10pt

**Common use cases**:
- Resignation / notice
- Recommendation letter
- Formal collaboration proposal
- Personal statement

**Language notes**:
- Chinese sign-offs can use "此致 / 敬礼", "顺颂商祺", or a context-appropriate formal closing
- English sign-offs should stay simple: "Best regards," / "Sincerely," / "Warm regards,"

### Portfolio

**Structure**:
1. **Cover** (name + one-line positioning + contact)
2. **About** (half-page introduction)
3. **Per-project 1-2 pages**:
 - Project title + type tag + date range
 - One-line description
 - 2-3 hero images (if applicable)
 - Role + challenge + outcome
4. **Selected works list** (additional projects as a short list)
5. **Contact** (return to contact details)

**Rules**:
- Visuals first, text supports
- Every project's outcome must be quantifiable
- Final product screenshots / real photos > design mockups > code screenshots
- If project images are missing, mark the gap. Do not fill the layout with unrelated imagery
- Don't list every tech stack - a mono tag row is enough

### Proposal (long-doc variant)

A proposal is a long-doc whose body argues for a specific commercial engagement. Three voice rules are unique to this variant:

#### 1. Work-volume frame vs attention frame

A pricing page can be written two ways. The work-volume frame lists deliverables, hours, or session counts; readers price-anchor against market hourly rates. The attention frame describes a senior advisor's judgment allocated across a few directions; readers price-anchor against the value of the directions. Pick one frame and stay consistent.

| Dimension | Work-volume frame | Attention frame |
|---|---|---|
| What is sold | Execution of N items | Allocation of judgment |
| Pricing basis | Hours / deliverables | Total engagement |
| Workload density | Fixed commitment | Flexes with the buyer's cadence |
| Reader posture | Procurement | Hiring an advisor |
| Signals | "I can do these tasks" | "I bet on these directions" |
| Fits | Outsourced delivery | Senior advisory work |

The work-volume frame backfires for advisor pricing: the more itemized the breakdown, the more the reader divides total fee by hours and decides "this is a vendor, can it cut corners?" Avoid these patterns when writing in the attention frame:

- Rigid monthly volume: `每月 N 份 / 节 / 篇`. Replace with "by milestone" or "as the cadence requires".
- Exhaustive coverage: `每个发布节点全程参与`. Replace with "deep involvement at major milestones".
- Specific session counts: `20 节课程`. Replace with "depth of content" anchors and let scheduling adjust.
- "Full-X" totalizers: `全方位 / 全链路 / 全程`. Replace with "key X" or "core X".

#### 2. With-price vs without-price modes

Whether to print the number depends on the reader. Use these defaults and switch by audience:

| Audience | Mode | Reasoning |
|---|---|---|
| Operating founder, direct buyer | **With price** | Direct decision-makers want the number to evaluate fit |
| Intermediary, business contact | **Without price** | Leave room for the intermediary's own commercial layer |
| Investor, board reader | **Without price** | Strategy first, money second; a number breaks the register |

The without-price variant drops the large-figure hero and keeps the value anchors. Rename the chapter from "合作方案与投入" (Plan & investment) to "合作方向与价值" (Directions & value) so the title matches the content.

#### 3. Kickoff-period emphasis pattern

Twelve-month timelines fail when every month is described at equal weight: the reader cannot tell what is urgent. Surface the first 60 days as a separate beat ahead of the timeline table:

- One paragraph on time scarcity ("this window closes; recovering it later costs several times more").
- One paragraph per month: bold the date, name the module, list 2-3 concrete actions, state the outcome.
- Then the full month-by-month table, which reads as detail rather than the story.

The pattern is "story before grid": the reader leaves the timeline section knowing one thing (the kickoff is dense), not twelve things (every month).

### Resume

The most constrained document type in kami.

**Hard constraints**:
- Strictly 2 A4 pages
- Every project follows three-part: Role / Actions / Impact
- 5 core skills, each with at least one brand-color emphasis
- Team size, tech stack, narrative voice must stay consistent throughout

**Key sections**:
- Header + 4 metric cards
- Summary (~50 English words or ~80 Chinese characters)
- Timeline (3 steps - long-range evolution signal)
- 3-5 core projects
- Public work / impact (optional)
- 5 core skills
- Education

**Metric card selection rule**:
- 1 card on **time** (years, consistency)
- 1 card on **scale** (team, users, projects, or other quantifiable scope)
- 2 cards on **results** (quantifiable external proof)

---

## Quality bars by document type

Structure is necessary but not sufficient. These bars define what separates compelling content from template-filling.

### Resume

**Impact formula**: Action + Scope + Measurable Result + Business Outcome. Every bullet must answer "what did I do, at what scale, with what result, and why did it matter?"

| Avoid | Use |
|---|---|
| "Worked on backend services" | "Redesigned order pipeline serving 2M daily txns, cut p99 latency from 800ms to 120ms, saved $340K/yr in infra costs" |
| "Led a team to deliver features" | "Led 5-engineer squad that shipped real-time collaboration (3-month timeline), adopted by 40% of enterprise accounts within one quarter" |
| "Improved performance" | "Reduced cold-start time 62% across 14 Lambda functions by replacing runtime init with pre-baked layers, cutting median API response from 1.2s to 0.45s" |

**Rules**:
1. Start every bullet with a strong past-tense verb (designed, led, reduced, migrated). Never "Responsible for" or "Helped with"
2. Every bullet needs at least one number. If no hard metric exists, use scope (team size, user count, codebase size)
3. Connect technical work to business outcomes: revenue, cost, reliability, user retention, time-to-market
4. Include before/after pairs when possible: "from X to Y" is more credible than "improved by Z%"
5. Use precise numbers over round ones: "$280K" reads as measured, "$300K" reads as estimated
6. Distinguish ownership: "owned" vs "contributed to" vs "coordinated". Inflating scope is the fastest way to lose credibility in an interview

**Senior vs junior**: junior resumes show execution ("built X"). Senior resumes show judgment ("evaluated 3 approaches, chose Y because of tradeoff Z") and multiplier effect ("mentored 4 engineers, 2 promoted within 12 months")

### Portfolio

**Core rule**: open every case study with the problem and its stakes, not with your role or the project name.

| Avoid | Use |
|---|---|
| "I redesigned the dashboard" | "Enterprise users abandoned the analytics dashboard at 73% rate within the first session. I led the redesign that cut abandonment to 31%." |
| "The client was happy" | "Task completion time dropped from 4.2 min to 1.8 min. NPS increased from 22 to 51 over 3 months." |

**Rules**:
1. Show 2-3 decision points where you chose between alternatives. Explain the tradeoff, not just the winner
2. Three-layer outcomes: quantitative metric (conversion rate +80%) + qualitative evidence (user quote) + business context ($1.2M additional annual revenue)
3. State your exact role and scope: "I designed" vs "I led" vs "I contributed to" are very different signals
4. 3-5 deep case studies beats 12 shallow ones. Depth is credibility
5. Always close the loop: every problem introduced must have a measured resolution
6. Prefer final product screenshots over mockups. If product images are missing, mark the gap rather than filling with unrelated imagery

### Slides

**Core rule**: every slide title should be a full declarative sentence (an assertion), not a topic label. The body provides one piece of evidence supporting the assertion.

| Avoid | Use |
|---|---|
| Title: "Q3 Performance" | Title: "Q3 revenue grew 23% because enterprise deals closed 2x faster" |
| 7 bullet fragments per slide | One chart proving the assertion |
| "Key Takeaways" slide with 8 points | One clear ask or recommendation |

**Rules**:
1. 20-40 words per slide maximum. If a slide has more than 40 words, split it or convert text to a visual
2. 5 items per list maximum (working memory capacity)
3. Three-act structure: Setup (slides 1-4, establish stakes) -> Evidence (slides 5-12, build the case) -> Resolution (slides 13-16, deliver the payoff)
4. Reading just the slide titles in sequence should tell the full argument
5. Include a "so what" moment every 3-4 slides to re-anchor the audience
6. End with one clear ask, not a bullet list of "key points"

**Eyebrow vs title non-duplication**: the eyebrow is a stable, cross-slide section label ("Growth / Q3 Results"). The title is a page-unique declarative claim ("Revenue grew 23% because enterprise deals closed 2x faster"). They must never say the same thing in different words. If removing the eyebrow would make the title ambiguous, the title is too weak. If reading the title makes the eyebrow redundant, the eyebrow is a topic label masquerading as context.

**Deck rhythm (>=12 slides)**: before writing any slide, sketch a layout-type sequence. Rules: every 4-6 slides must include a `chapter_slide` (ink-blue full-bleed divider); never run more than 5 consecutive `content_slides` without a divider; the deck must include at least one `quote_slide` or `metrics_slide` to vary density. Monotony is a structure failure, not a content one.

**Term consistency self-check**: after drafting, list every domain term that appears 3 or more times (product names, feature names, roles, metrics). Confirm there is exactly one spelling and capitalization for each. Inconsistent casing ("LLM" vs "llm" vs "large language model") signals an unreviewed draft.

### Equity Report

**Core rule**: lead with the variant perception (what you see that the market doesn't) and tie every thesis driver to a measurable financial impact.

| Avoid | Use |
|---|---|
| "Strong management team" | "Management delivered 23% revenue CAGR over 5 years while keeping debt-to-equity below 0.4" |
| "Massive opportunity" | "We estimate 25% upside to $X based on DCF with 12% WACC and 3% terminal growth" |
| Vague "risks include competition" | "BYD's 35% unit cost advantage in the $20-30K segment threatens 15% of addressable volume by 2027" |

**Rules**:
1. Investment thesis on page 1, above the fold. Rating + price target (if applicable) + 3-5 bullet thesis drivers
2. Every claim backed by a number or a source. No unquantified superlatives
3. At least two valuation methods with sensitivity ranges. Single-method valuation is amateur
4. Catalysts must have dates and expected magnitude: "Robotaxi launch in Dallas, June 2025, adding estimated $X to revenue run-rate by Q4"
5. Competitive positioning with market share numbers, not narrative: "23% share of the $45B market, up from 18% in 2022"
6. Risk factors quantified and connected to the financial model, not generic disclaimers
7. Professional tone: "we estimate" / "our base case" / "we see upside to". Never "this stock will moon" or "buy the dip"
8. Acknowledge counter-arguments before dismissing them. One-sided analysis signals bias, not conviction
9. Separate GAAP from non-GAAP clearly. Flag one-time items (warranty reserves, tax benefits, restructuring charges)

### Long Document

**Core rule**: each chapter's claim paragraph must survive the "so what?" test. If the reader asks "why should I care?", the first paragraph must have the answer.

**Rules**:
1. Evidence density: at least one data point per paragraph. A paragraph with zero numbers is an opinion paragraph and should be rare
2. Callout or figure after every 3-4 paragraphs of dense text. Long unbroken prose causes eye fatigue in print
3. Counter-arguments addressed before they become reader objections. If you can predict the pushback, address it proactively
4. Source cues preserved inline: "(Gartner, 2025)" or "according to the company's 10-K" so readers can distinguish fact from inference
5. Each chapter should stand alone as a mini-essay with its own arc: claim -> evidence -> conclusion

### One-Pager

**Core rule**: the reader grasps the point in 30 seconds. Every element that doesn't serve 30-second comprehension is bloat.

**Rules**:
1. Metrics are the headline, not supporting evidence. If your 4 metric cards don't tell the story, the metrics are wrong
2. The lead paragraph must contain the single sharpest claim, not context-setting
3. Bullet points should be evidence, not restated arguments. Each bullet: fact + number + "so what"
4. Footer is for contact and classification, not for squeezing in one more argument

### Letter

**Core rule**: first paragraph states purpose in one sentence. Last paragraph states the specific ask or next step. Everything in between is evidence.

**Rules**:
1. One point per middle paragraph, each with its own evidence
2. Tone calibration per use case: resignation (grateful + clear), recommendation (specific + enthusiastic), proposal (value-first + concrete), personal statement (authentic + structured)
3. Sign-off matches formality: "Sincerely" for formal, "Best regards" for professional-warm, "Warm regards" for personal
4. Under no circumstances exceed one page. If you need two pages, it's a memo or a proposal, not a letter

### Changelog

**Core rule**: one sentence per change, verb-led, user-facing language. If the user cannot understand the change from the sentence alone, rewrite it.

| Avoid | Use |
|---|---|
| "Refactored internal state management module" | "Fix crash when switching tabs rapidly on iPad" |
| "Updated dependencies" | "Upgrade OpenSSL to 3.2.1 (patches CVE-2026-1234)" |

**Rules**:
1. Breaking changes always first, with migration path ("Replace `config.old` with `config.new`; run `migrate.sh` to convert")
2. 5-8 items per section. If more, this is probably 2 releases
3. Group by user impact (Breaking / Features / Fixes), not by component or file
4. No internal jargon. "Fix memory leak in image decoder" is clear. "Fix retain cycle in UIImageDecoderBridge" is not

---

## Diagrams and infographics

Words inside a diagram or infographic (title, eyebrow, node label, summary line) follow tighter rules than body prose. Readers process them in seconds, so rhetorical flourish reads as noise. This applies whether the figure is a Kami SVG embedded in a long-doc or an external image rendered elsewhere.

### Avoid colloquial slogan-words

| Avoid | Why it fails |
|---|---|
| 白搭 / 立住 / 才顺 / 回炉 / 闸 | Slang verbs; rewrite as a literal claim |
| 必看 / 一图看懂 / 彻底搞懂 | Bait phrasing; state the figure's actual content |
| 爆款 / 神器 | Marketing tone in an engineering figure |
| 飞轮 / 闭环 (unless audience is fluent) | Use 数据循环 / 持续改进 / 四类入口 |

### Avoid product-specific judgments that date fast

Pinning a category to a current product name ages the figure within a quarter, because tools evolve faster than diagrams ship. Frame the paradigm and let the reader map products themselves: prefer 「上一代工具范式 / 新一代执行范式」over 「Cursor 是副驾驶 / Claude Code 是自动驾驶」.

### Slogan to neutral rewrites

| Before (slogan) | After (neutral) |
|---|---|
| 没对完，不算完成 | 交付前，过三遍 |
| 任一不过则回炉 | 任一步不通过，回到修改 |
| 交付前最后一道闸 | 交付前最后检查 |
| 订阅前先把这些习惯立住 | 订阅前的基础检查 |

The principle: a diagram caption is engineering documentation, not marketing copy. Restraint reads as competence; bravado reads as filler.

---

## Coupling rules (layout × content)

### Emphasis rhythm

Across any document:
- ≤ 2 emphasized items per line
- Emphasis must be a **quantifiable number** or a **distinctive phrase**
- Do not emphasize adjectives

### Number formatting

| Use | Avoid |
|---|---|
| 5,000+ | 5000+ (missing thousands separator) |
| 5,000+ | 5，000+ (full-width comma in a metric) |
| 90% | 90 % (space before percent) |
| ~$10M | $9,876,543 (false precision reads fake) |
| 2026.04 | 2026年4月 / April 2026 (when horizontal space is tight) |
| -> | → |

### Language-specific punctuation

Chinese documents:
- Prefer `「」` for quoted prose, not straight double quotes
- Keep numbers, commas, percent signs, and dates half-width in metric-heavy areas
- Add spaces between Chinese text and Latin product names when it improves readability

English documents:
- Use straight quotes in source text unless the document already has a typographic quote convention
- Prefer compact date forms (`2026.04`) in dense layouts and natural dates (`April 2026`) in prose

### Emphasis is not bold

Use `color: var(--brand)` alone - don't also add `font-weight: bold`. Bold breaks the single-weight design language.

---

## Pre-ship checklist

Run through before every draft:

- [ ] Any jargon like "leverage / unlock / embrace / pioneer"? Cut.
- [ ] Any Chinese filler like "拥抱 / 打造 / 赋能 / 重构"? Rewrite in plain language.
- [ ] Does every paragraph's first sentence stand alone? If not, that paragraph has no claim.
- [ ] Are all numbers verifiable? If asked "where did this come from", can you answer?
- [ ] Are current facts, versions, launch dates, funding, financials, and specs backed by reliable sources?
- [ ] Does every branded document have logo, product image, or UI screenshot coverage? Are missing materials clearly marked?
- [ ] At least one **distinctive phrase** (not industry boilerplate)?
- [ ] Every emphasized (brand-colored) span is either a number or a distinctive phrase? If not, remove the emphasis.
- [ ] Paragraph lengths even? No paragraph over 5 lines?
- [ ] Number format consistent (commas, percent signs, arrows)?
- [ ] Chinese punctuation and Chinese / Latin spacing consistent where applicable?
- [ ] Page count within the document's constraint (resume 2, one-pager 1, letter 1)?

---

## Writing references

- **Paul Graham's essays** - short, direct, judgmental. The gold standard for essayistic writing.
- **Stripe Press books** - print-grade typography paired with deep content. Where to learn the craft of the single sentence.
- **Minto's Pyramid Principle** - conclusion first, evidence below. The shape of every one-pager and exec summary.
- **Ben Horowitz's blog** - how to write technical and business judgment in prose ordinary people can read. The template for long-doc voice.

None are required, but reading any one of them will move the dial on both your writing and your judgment.
