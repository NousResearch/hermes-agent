# Validated Content Brief Format

This is the EXACT format produced by the 2026-06-12 kimi-k2.6 run that scored 62/62 on validation. Use this as the template for all content brief output.

## File Naming

```
~/.hermes/content-briefs/{YYYYMMDD}-{arxiv-id}-{short-tag}.md
```

Example: `20260612-2606.09659-lclms.md`

## YAML Frontmatter (mandatory)

```yaml
---
paper_id: 2606.09659
date: 2026-06-12
paper_title: "End-to-End Context Compression at Scale (LCLMs)"
target_voices: [sahil-twitter, sahil-linkedin]
score: 5
---
```

Fields:
- `paper_id` — arXiv ID (no version suffix)
- `date` — YYYY-MM-DD of brief generation
- `paper_title` — full paper title in quotes
- `target_voices` — array of brand voice skill names to load
- `score` — final v2 score (1-5)

## Body Sections (all mandatory)

### Paper Summary
1-2 sentences. No bullet points. Include key metrics if available (e.g., model sizes, benchmark scores, compression ratios).

### Content Angles

#### Twitter/X
- 2-4 tweetable hooks. Each 200-280 chars. Include paper URL.
- Style: forward-looking, insightful, opinionated.

#### LinkedIn
- 1 paragraph describing the post angle (not the post itself).
- Style: strategic, research-oriented, or educational depending on paper.

#### Blog
- 1 paragraph describing the article angle. Include suggested length and sections.
- Always mention: KENSEI integration point if applicable.

### Target Voice
Per-platform voice direction. Use skill names: `sahil-twitter`, `sahil-linkedin`.

### Suggested Schedule
Concrete dates (DD/MM format) for each platform. Blog should be 5-14 days out to allow drafting time.

## Validation Checklist

Before the cron writes a content brief, verify:
- [ ] Frontmatter has all 5 fields
- [ ] Twitter section has at least 2 hooks with paper URLs
- [ ] LinkedIn section mentions a specific angle, not generic "write about this"
- [ ] Blog section includes suggested length and sections
- [ ] Schedule has concrete dates, not "next week" or "soon"

## Anti-Patterns

- ❌ Generic LinkedIn angle: "Write a post about this paper"
- ✅ Specific angle: "Why prior KV cache methods fell short, and how adaptive expansion enables long-horizon agents"

- ❌ No schedule or vague schedule
- ✅ Concrete: "Twitter: 15/06, LinkedIn: 17/06, Blog: 22/06"

- ❌ Missing paper URL in Twitter hooks
- ✅ Include `arxiv.org/abs/ID` in at least one tweet

## Verified Output (2026-06-12)

6 briefs produced, all 6 passed validation across all fields. Vetted briefs:
- `20260612-2606.09659-lclms.md` — LCLMs paper, score 5
- `20260612-2606.09730-searchswarm.md` — SearchSwarm paper, score 5
- `20260612-2606.12320-five-plane.md` — Five-Plane Governance, score 5
- `20260612-2606.12674-evoflux.md` — Evoflux, score 5
- `20260612-2606.13177-memrefine.md` — MemRefine, score 5
- `20260612-2606.13317-skillcat.md` — SkillCAT, score 5
