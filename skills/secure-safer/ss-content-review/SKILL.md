---
name: ss-content-review
description: "Content review skill for Secure Safer — adapts code-review methodology (severity labels, 4-phase process) for insurance content. Reviews articles, social posts, and landing pages for accuracy, compliance, tone, SEO, and brand consistency."
version: 1.0.0
author: Hermes Agent via Rafiul
license: MIT
metadata:
  hermes:
    tags: [secure-safer, content-review, editing, compliance, seo, brand]
    related_skills: [copy-editing, ss-content-generation, code-review, ai-seo]
---

# Secure Safer Content Review

Adapts the **code-review skill** methodology for insurance content. Same structured process, severity labels, and collaborative tone — applied to articles, social posts, and landing pages.

## Severity Labels (from code-review-skill)

| Label | Meaning | Content Example |
|-------|---------|----------------|
| 🔴 blocking | Must fix before publish | Regulatory error, wrong state law cited, missing disclaimer |
| 🟡 important | Should fix | Weak CTA, unclear headline, missing source link |
| 🟢 nit | Nice to have | Slight wording preference, formatting |
| 💡 suggestion | Alternative approach | Different angle or structure to consider |
| 📚 learning | Educational context | "NY DFS actually updated this rule in March" |
| 🎉 praise | Good work | Strong analogy, great local angle, clear explanation |

## Review Process (4 Phases)

### Phase 1: Context Gathering (2 min)
1. Read the research brief — what's the topic, concern, audience?
2. Check word temperature labels — does the content follow the arc?
3. Identify content type — article, social post, or landing page?
4. Note SEO keywords and AEO target question

### Phase 2: High-Level Review (5 min)
1. **Accuracy** — Are all claims sourced? Is state-specific info correct?
2. **Compliance** — Any regulatory red flags? Disclaimer present?
3. **Brand Voice** — Does it sound like Secure Safer? "Energetic precision"?
4. **Structure** — Does it follow the template? AEO answer → problem → solution → FAQ → CTA?

### Phase 3: Line-by-Line (10 min)
1. **Word Temperature Check** — Arc correct? Warm → hot → cold?
2. **Readability** — Sentences 14-18 words avg? No jargon walls?
3. **SEO** — Primary keyword in H1 + first 60 words? AEO answer present?
4. **Sources** — Every claim has a link? (.gov > .edu > trade > blog)
5. **CTA** — Specific, low-friction, one action?

### Phase 4: Summary (2 min)
1. List blocking issues first
2. Highlight what works (praise)
3. Clear verdict: Approve / Edits Needed / Needs Rewrite

## Content-Specific Checklists

### Article Checklist
- [ ] H1 contains primary SEO keyword + benefit hook
- [ ] AEO answer in first 60 words (targets featured snippet)
- [ ] Temperature arc: warm (problem) → hot (risk) → cold (solution)
- [ ] FAQ section with 3-5 schema-ready Q&A
- [ ] CTA is specific and low-friction
- [ ] All claims have source links (.gov preferred)
- [ ] Meta description (150-160 chars) includes keyword
- [ ] No regulatory red flags (no "guaranteed," no medical advice)

### Social Post Checklist
- [ ] Platform-appropriate length (LinkedIn: 150-300, FB: 80-200, X: <280)
- [ ] Hook in first sentence
- [ ] Visual concept described for image gen
- [ ] Brand colors referenced (#fc820c orange, clean white)
- [ ] CTA matches platform norms

### Landing Page Checklist
- [ ] Headline = benefit + keyword
- [ ] Subheadline = specificity + objection handler
- [ ] Problem → solution → proof → action flow
- [ ] 3-5 selling points from research brief
- [ ] Social proof near decision points
- [ ] Schema markup applicable? (FAQ, HowTo, LocalBusiness)

## Word Temperature Compliance
- **HOT words** (fined, denied, lawsuit): max 2 per piece
- **WARM words** (worried, confused): 60% of emotional language
- **NEUTRAL words** (coverage, premium): backbone of content
- **COLD words** (protected, compliant): resolution zone only

## Brand Compliance
- Primary orange #fc820c for highlights/accents
- Plus Jakarta Sans for headlines, Manrope for body
- Clean white space, minimal clutter
- "Energetic precision" — sharp but warm

## References
- `code-review` skill for the original methodology
- `copy-editing` skill for 5-pass editing
- `ai-seo` skill for AEO optimization
- `ss-content-generation` skill for the content pipeline
- `Brand Design System.md` in vault for brand specs
