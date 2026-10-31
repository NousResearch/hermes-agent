# Critiquing a Logo

Use when the user asks for feedback on a logo (theirs or anyone's), wants to compare options, or needs a diagnosis
before a redesign. Be specific, kind and useful: name what works, what doesn't, why, and exactly how to fix it.

## Method

1. **Understand the context first** — what the organisation does, for whom, what it should feel like. A logo can only
   be judged against its purpose. If unknown, ask one question or state assumptions.
2. **First impression (2 seconds)** — what do you see, feel, and remember? What would a stranger call it?
3. **Technical pass** — if an SVG is available, run `scripts/svg_audit.py` and `scripts/preview_sheet.py`, then look at
   the preview. For a raster image, reason visually about the same checks.
4. **Principles pass** — score each dimension below 1–5 with one-line evidence.
5. **Prioritise** — the 3 changes with the biggest impact, most important first. Make each actionable.
6. **Optionally demonstrate** — sketch the fix as an SVG revision when the user wants it.

## Scorecard

| Dimension | Question | Score (1–5) | Evidence |
|---|---|---|---|
| Idea | Is there one clear, relevant idea? Can it be said in a sentence? | | |
| Simplicity | Is anything unnecessary? Does it survive 16 px and a squint? | | |
| Distinction | Does it stand apart from competitors and famous marks? | | |
| Memorability | Is there one defining feature you'd recall tomorrow? | | |
| Relevance / tone | Does its character (shape, weight, colour, type) match the brand? | | |
| Craft | Clean geometry, optical corrections, consistent weights, good spacing/kerning? | | |
| Versatility | One colour, reversed, small, large, embroidery, app icon, lockups? | | |
| Longevity | Conceptual rather than trendy? Will it date soon? | | |
| Colour | Few, purposeful, ownable, accessible, reproducible? | | |
| Typography | Appropriate, customised, legible, well spaced, ≤ 2 families? | | |

Total /50 is a rough guide only; one fatal flaw (illegible at small size, looks like a competitor, offensive reading)
outweighs a high total.

## Common diagnoses and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Turns to mush at small sizes | Too many elements, thin lines, small gaps | Remove details, thicken strokes, open gaps; make a small-size version |
| Looks generic / like a template | Stock typeface, cliché symbol (globe, swoosh, lightbulb) | Customise letters, find an ownable twist, derive the idea from the name or promise |
| Looks dated | Effects (gradients, bevels, shadows), trendy fonts | Flatten, simplify, choose a more timeless type direction |
| Busy | Two or three ideas competing | Pick one idea; move the others into the identity system |
| Symbol and type feel unrelated | Different geometry/weight logic | Share radii, stroke weights, angles; rebalance sizes |
| Feels unstable | Accidental tilt, top-heavy mass | Widen the base, correct axes, check optical centre |
| Circle looks small next to letters | Missing overshoot | Enlarge round forms 1–3 % |
| Rounded rectangle looks pinched | Bone effect | Smooth curvature transitions |
| Reversed version looks bold | Irradiation | Thin the white version slightly |
| Fails on photos | No graphic device | Add an outline/container version |
| Hidden meaning nobody sees | Too subtle or too complex | Make it a bonus, not the point — or simplify until it reads |
| Unfortunate reading | Designer blindness after long exposure | Change the offending shape; test with fresh eyes |

## Output format

```markdown
## Logo critique — <name>
**First impression:** <1–2 sentences>
**What works:** <2–3 bullets>
**Scorecard:** <table above, filled>
**Top 3 changes (in priority order):**
1. <change> — why, and how (specific: "open the counter of the e from 6 to 12 units")
2. …
3. …
**Optional next step:** <offer a revised SVG / alternative directions>
```

Tone: critique the work, never the person. Balance honesty with encouragement; the goal is a better mark.
