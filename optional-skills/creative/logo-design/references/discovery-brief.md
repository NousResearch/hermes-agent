# Discovery & the Design Brief

Design is a strategic response to a business need, not decoration. Without clarity about the need, the work
lacks direction no matter how beautiful it is. This file covers how to gather that clarity quickly and turn it
into a brief that anchors every later decision.

## Contents
1. Why discovery first
2. The fast path (when the user wants speed)
3. The full question bank
4. Research you do yourself
5. From answers to a brief (template)
6. Word mapping — turning language into symbols
7. Warning signs in a brief

---

## 1. Why discovery first

- Uncover and define the problem before attempting to solve it visually. Rushing into drawing raises the odds
  of missing the mark, and a missed mark is expensive for everyone.
- The brief is a *contract of clarity*: a shared, written record of what success looks like. When
  disagreements come (they will), return to the brief instead of arguing taste. That doesn't mean never
  adjusting — it means adjustments are purposeful, not arbitrary.
- A strong brief doesn't constrain creativity; it channels it.
- The designer is hired as the expert who guides, creates and solves — not as a technician executing a
  pre-drawn idea. If a client already has the exact design in mind, clarify the relationship early and honestly.

---

## 2. The fast path (when the user wants speed)

Ask **at most five** questions in one message, then proceed. If the user can't or won't answer, infer sensible
answers, **state your assumptions explicitly**, and continue — you can revise after they react to concepts.

1. **Name & what you do** — exact spelling/casing of the name, one-line description, and any tagline.
2. **Audience** — who must notice and trust this (age, profession, mindset)? Where will they meet the logo?
3. **Three to five adjectives** the brand should feel like (e.g. calm, precise, rebellious, warm, premium).
   Adjectives are the single most useful input for concepting.
4. **Competitors or look-alikes to avoid** — and any marks the user admires (and why).
5. **Constraints** — colours to keep/avoid, existing equity to preserve, where it must work (app icon,
   embroidery, signage, favicon), deadline for decisions.

Optional sixth when relevant: *Who makes the final decision?* (Always know who signs off.)

---

## 3. The full question bank

Use for substantial projects. Every business is different — adapt, don't recite. Allow time for answers;
off-topic remarks often open new avenues, but keep the conversation focused on outcomes.

### The essential four
- Who are you?
- Who needs to know?
- How will they find out?
- Why should they care?

### The business
- **Positioning** — compared with alternatives, where do you sit? What is your price point? How big is the
  business and where is it heading (projected scale gives context for positioning)?
- **Purpose & mission** — beyond the economics, why is this worth doing?
- **Composition** — how is the organisation structured (parent/sub-brands, divisions, products)?
- **Culture** — what shared behaviours best express the mission?
- **Personality** — what style and manner? If the brand were a person, how would they speak and dress?
- **Goals** — five key goals over the next year / five years.
- **Growth** — where are the biggest opportunities for the business and its image?
- **Promises** — what does the organisation promise its audiences? (Promises synthesise what it stands for.)

### The audience
- **Current audience** — who, where, when, why?
- **Desired audience** — a different or wider group? Demographics (age, gender, nationality, profession)
  and psychographics (values, lifestyle).
- **Current perception** — how does the audience see the brand today?
- **Desired perception** — how should they see it?
- **Desired response** — what should they feel, think or do after encountering the brand?

### The competition
- Who are the main competitors? How are you different? What visual cues, colours and typefaces dominate the
  category? What marketing is gaining traction?

### The project
- What are the goals for the new identity? What specific deliverables are needed?
- Why now? What's motivating the project and driving the timeline?
- Who is involved, with what roles? Any outside agencies or partners?
- What concerns or obstacles do you anticipate? Anything in the culture, structure or past experiences that
  could make this easier or harder?
- Budget range? Are other designers being considered, and when will a decision be made?
- **Who makes the final decision?** Involve that person when presenting.
- Existing equity: current logo, colours, symbols people already recognise? What must survive a redesign?
- Applications: where will it appear first — and where might it appear in five years? Any production
  constraints (embossing dies, embroidery, laser etching, signage regulations, single-colour printing)?
- Naming: is the name final? Is there a shorter name people already use? (A nickname the public already
  uses can be a stronger communicative name than the legal one.)

### After gathering answers, ask yourself
- What are the client's real concerns?
- What does the organisation want to emphasise?
- What is it *really* selling? (A tap sells convenience and kitchen pride; a gym sells confidence.)
- How does it want to be perceived in the market? Stylish identities win awards; relevant ones win market share.

---

## 4. Research you do yourself

- **The organisation's history**, its current identity, past identity efforts and how they shaped perception.
  Previous logos often reveal useful patterns or pitfalls.
- **The competitive landscape**: collect competitor marks, note colours, shapes, type, and what they all share.
  Map them (e.g. on axes like *traditional ↔ modern*, *playful ↔ serious*) to find open territory.
- **Category conventions** via the bundled library. Examples:
  ```bash
  python3 scripts/search_library.py --industry payments-fintech --format table
  python3 scripts/search_library.py --subject shield --format table
  python3 scripts/search_library.py --industry database-data --summary
  ```
  `--summary` shows the dominant mark types, colours and techniques in a category — i.e. what to avoid
  repeating and what cues signal "belonging".
- **Cultural checks**: meanings of candidate colours, symbols and gestures in the audience's cultures;
  whether a symbol is sacred or owned by a community.
- **Name mechanics**: letter shapes (which letters have rich structure for a monogram — `a`, `s`, `w`, `g`, `k`
  combine well; `i`, `l` offer little), ligature opportunities, symmetry, how the word sounds spoken.

---

## 5. From answers to a brief (template)

Write this before designing and show it to the user. Keep it short, jargon-free and specific — designers
must be editors too.

```markdown
## Design brief — <Name>

**Summary**: <one sentence: who they are, for whom, and why they're different>
**Audience**: <primary / secondary; where they will meet the mark>
**Brand adjectives**: <3–5 words>   **Avoid feeling**: <2–3 words>
**Promise / big idea**: <one line>
**Competitive landscape**: <what the category looks like; what to avoid; open territory>
**Mark type hypothesis**: <wordmark / combination / letterform ... and why> (see mark-types.md)
**Must work on**: <favicon/app icon, social avatar, print, signage, embroidery, dark mode, motion ...>
**Constraints**: <colours, equity to keep, legal, production>
**Success criteria**: <e.g. recognisable at 16 px; distinct from X and Y; reads as "precise but warm">
**Decision-maker**: <name/role>   **Deliverables**: <files, lockups, mini-guidelines>
**Assumptions I made**: <list — invite correction>
```

---

## 6. Word mapping — turning language into symbols

Good designers translate language into symbols. A word map bridges strategy and form.

1. Write the core words: the name, the offering, the adjectives, the promise.
2. Branch from each word into **nouns** (things you can draw), then into associations, metaphors, opposites,
   sounds, places, materials, processes, historical references.
3. Circle the intersections — where two branches meet is where distinctive ideas live (e.g. "trust" → shield;
   name begins with T → a T that is also a shield; the name's meaning is a woodworking joint → precise fit).
4. Turn the best 6–10 intersections into one-sentence concept statements before drawing anything.

Example (abridged):
```
Brand: "Harbor" — a savings app. Adjectives: safe, calm, patient.
  harbor → boat, anchor, lighthouse, breakwater, shelter, bay-shape, tide
  safe   → shield, lock, nest, roof, cupped hands, enclosure
  calm   → horizon line, still water, circle, slow wave
  patient→ growth rings, stacking, tide rising, sunrise
Intersections:
  • An "H" whose crossbar is a calm horizon/water line
  • Stacked horizontal waves forming a rising bar chart (tide = growth)
  • A bay-shaped enclosure (negative space) that shelters a coin
```

If neither the name nor the activity offers visual clues, use abstract keywords that do translate to form:
*connection, speed, unity, network, care, stability, balance, growth, openness, precision*.

---

## 7. Warning signs in a brief

- **"Total creative freedom."** Usually means the client has no vision and expects the designer to supply one
  — which takes many more rounds. Ask for stylistic references (marks they like/dislike and why) to calibrate.
- **"It should show everything we do."** Explain *identify, don't explain*; offer a system (icons, imagery,
  copy) to carry the detail instead of the mark.
- **"Make it pop / modern / clean."** Ask what those words mean to them; show references.
- **No decision-maker identified.** Presentations to the wrong person get re-litigated later.
- **The real problem isn't the logo.** If the product, service or message is broken, say so; a logo won't fix it.
- **Design by committee.** Plan to present to the smallest group that includes the decision-maker; many voices
  produce an illusion of consensus and averaged, forgettable marks. See `presentation-delivery.md`.
