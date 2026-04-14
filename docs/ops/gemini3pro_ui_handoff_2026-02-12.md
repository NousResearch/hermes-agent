# Gemini 3 Pro UI Handoff — Twitter Digest (WalletDB/NGMI)

## Objective
Redesign the Twitter digest UI/UX so output is decision-ready, low-noise, and operator-grade.

---

## Prompt to paste into Gemini 3 Pro

You are a senior product designer + frontend architect.

Context:
- We have a Telegram-delivered “Twitter Signal Digest” for crypto/AI trend tracking.
- Current output quality is poor: generic phrase fragments, shadow junk, weak prioritization.
- We need a **UI/UX-first redesign** that is operator-grade and decision-ready.
- Surface is primarily Telegram messages/cards, but design should be portable to web panel later.

What I need from you:

## 1) Design Spec (v2)
Create a concrete UI spec for this digest with:
- Information hierarchy
- Section order
- Visual grammar (badges, confidence, momentum, novelty)
- Compact vs detailed mode
- Mobile-first scan behavior (3–5 sec glance)
- Failure/sparse-window behavior (never useless/empty)

Required sections:
1. Snapshot
2. Top Signals
3. Acceleration Board
4. Novel/Unknown Context
5. Action Queue (what to check next)

## 2) Ranking + Quality Rules (product logic, not model training)
Define deterministic display rules:
- How to suppress low-information phrase junk
- How to demote low-evidence shadow items
- Minimum evidence for each section
- Tie-break logic
- “Do not show” policy examples
- Fallback policy when data is sparse

## 3) Output Templates
Provide exact message templates for:
- 4h recall mode
- 12h strict mode
- Novel alert message
- Empty/sparse run message (still useful)

Keep templates production-ready with clean markdown.

## 4) Acceptance Checklist
Provide a QA checklist with pass/fail criteria:
- readability
- actionability
- noise rate
- consistency across runs
- novelty quality

## 5) Migration Plan
Give a phased rollout:
- v2.0 hotfix
- v2.1 stabilization
- v2.2 polish
with expected impact and risks.

Output format:
- Use headings and bullet points
- Include concrete examples
- Avoid generic advice
- Be strict and opinionated

Important constraints:
- Deterministic metrics remain authoritative.
- LLM text is assistive only.
- Avoid exposing internal debug fields in default output.

At the end, include:
A) “Top 10 immediate changes”
B) “One-page final template ready to ship”

---

## Current output example (problematic)

```md
# 📊 Twitter Signal Digest — Top 40
_Example Mode (12h) • View: Compact_

## Snapshot
- Topics to review: **20** | Active in run: **11** | Scored: **11**
- Coverage (last 24h ingest): **1538** posts seen | **92** dropped for age
- Window: **12h** short / **24h** long

## 1) Top Signals (Top 40)
- 01. **child trafficking** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 2 feeds
- 02. **using claude** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 2 feeds
- 03. **playing around** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 04. **something coming** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 05. **chatgpt moment** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 06. **early access** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 07. **ai agents** — New | Momentum 1.20x | Mentions 12/22 | Sources 12 authors / 2 feeds
- 08. **agent teams** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 2 feeds
- 09. **money while** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 10. **40b active** — New | Momentum 3.33x | Mentions 2/2 | Sources 2 authors / 1 feeds
- 11. **$BTC** — Emerging (Shadow rescue) | Momentum 0.00x | Mentions 1/1 | Sources 0 authors / 0 feeds
- 12. **$COIN** — Emerging (Shadow rescue) | Momentum 0.00x | Mentions 1/1 | Sources 0 authors / 0 feeds
- 13. **$HYUNDAI** — Emerging (Shadow rescue) | Momentum 0.00x | Mentions 1/1 | Sources 0 authors / 0 feeds
- 14. **claude code** — Emerging (Shadow rescue) | Momentum 0.00x | Mentions 5/6 | Sources 0 authors / 0 feeds

## 2) Acceleration Board
- 01. **40b active** | Formula 3.33 (momentum 3.33x + Δmentions 1.00 vs baseline 1.00)
- 02. **agent teams** | Formula 3.33 (momentum 3.33x + Δmentions 1.00 vs baseline 1.00)
- 03. **chatgpt moment** | Formula 3.33 (momentum 3.33x + Δmentions 1.00 vs baseline 1.00)

## 3) NOVEL / UNKNOWN
- 01. **child trafficking** — new conversation cluster is appearing faster than baseline (...)
- 02. **agent teams** — new conversation cluster is appearing faster than baseline (...)
```

### Known issues in this example
- Generic phrase fragments dominate (“playing around”, “something coming”, “money while”).
- Shadow-rescue rows with `0 authors / 0 feeds` still appear in high-visibility positions.
- Weak interpretability and low actionability despite non-empty output.
- Novel section includes topics that are not decision-useful.

---

## Expected deliverable from Gemini
- strict UI spec
- deterministic display policy
- clean templates (4h/12h/novel/sparse)
- acceptance checklist
- phased migration plan
- top 10 immediate changes
- one-page ready-to-ship format
