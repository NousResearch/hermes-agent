---
name: x-post-writer
description: Use when drafting, rewriting, or repurposing short-form X content, including single posts, quote posts, replies, threads, launches, and personal stories, with source fidelity and claim verification built in.
version: 1.0.0
author: Tony Simons
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: content
    tags: [x, twitter, posts, threads, quote-posts, replies, social-writing]
    related_skills: []
---

# X Post Writer

## Overview

Write short-form X content from notes, drafts, links, or verified source material.
Default to one finished post unless the user explicitly requests a thread or
multiple posts.

Source fidelity outranks punch, post count, inferred benefits, and creative
expansion. Internal routing, objectives, verification notes, source ledgers, and
quality checks are never part of the delivered copy.

```text
route format -> route source -> choose objective -> verify claims -> draft -> voice pass -> quality gate -> finished text
```

Load references progressively:

- `references/formats.md` for the selected format
- `references/source-routing.md` when repurposing or selecting source treatment
- `references/claim-verification.md` for risky factual claims
- `references/voice-customization.md` when a distinct voice must be preserved
- `references/algorithm-notes.md` only for distribution or optimization questions

## When to Use

Use this skill for:

- Single X posts, rewrites, launches, link drops, recommendations, or announcements
- Quote posts or commentary on an existing X post
- Replies
- Explicit threads or multi-post sequences
- Personal milestones, reflections, or first-person stories
- Fresh X angles from articles, videos, releases, or project updates

Do not use this skill for:

- Long-form X Articles or essay-length content
- Source extraction, account analytics, or media downloading
- Publishing, scheduling, or browser composition without separate authorization
- Claims that require evidence when no evidence can be obtained

## Safety Contract

1. Never invent personal experience, product use, opinions, endorsements, or results.
2. Never add technical mechanisms, benefits, numbers, or attributions that are not supplied or verified.
3. Treat private analytics, revenue, customer data, and unpublished information as sensitive.
4. Never publish or schedule automatically.
5. Current facts and platform behavior must be verified immediately before use.
6. An instruction to state a claim is not evidence for that claim.

## Routing Contract

### Format

| Request | Route |
|---|---|
| No format specified | Single post |
| "One post," "tweet," or "make this punchier" | Single post |
| "QT," "quote this," or commentary on an X post | Quote post |
| "Reply to this" | Reply |
| "Thread" or "multiple posts" | Thread |
| Personal experience, milestone, gratitude, reflection | Personal story |
| X Article, long-form essay | Route away |

### Source

| Source | Treatment |
|---|---|
| User draft or raw wording | Preserve claim, angle, and voice; improve structure |
| User personal notes | Preserve genuine first person |
| Previously published material | Preserve facts; rebuild the X angle and hook |
| URL, repository, announcement, or external source | Verify material claims before drafting |
| Existing X post being quoted | Add a take; do not restate the embedded post |

The source contract outranks generic formulas. Do not replace the user's actual
point with a stock marketing hook.

## Source Lock

When the user supplies notes, a draft, or a source excerpt, treat its concrete
facts as a whitelist.

- Reorder, compress, paraphrase, and improve rhythm.
- Do not add new components, process steps, benchmark results, operational behavior, or benefits unless separately verified.
- Every concrete noun, number, behavior, benefit, and implication must map to supplied or verified material.
- Benefits are claims. "Safer," "faster," "cleaner," and "no double counting" require support.
- Sparse notes should produce shorter copy, not larger facts.
- For a thread, assign one supplied fact cluster to each post.
- Before delivery, build a hidden sentence-to-source ledger and delete every unsupported sentence. Do not output the ledger.

Source-lock regression: "built" does not authorize "just shipped" or "today."
"Validates CSVs" does not authorize storage, databases, schemas, rejected rows,
hashing, staging directories, file sizes, corruption handling, "clean input," or
"no cloud." "Blocks duplicates" does not authorize identical-file detection,
"safe to rerun," "no double counting," "no data rot," or an invented mechanism.
"Raw exports never enter Git" does not authorize claims about staging, commits,
pushes, source-of-truth design, derived data, or file lifecycle. Do not add
benefit-only closers to sparse notes. Split the supplied facts across posts and
stop.

## Objective

Choose one primary objective:

- `announcement-clarity`
- `reach`
- `authority`
- `bookmarks`
- `replies`
- `follows`
- `click-through`
- `personal-connection`

Infer the objective when context makes it clear. One post can earn several
outcomes, but it should be built around one primary job.

## Workflow

1. **Classify the format.** Default to a single post.
2. **Classify the source.** Decide what must be preserved and what can be rebuilt.
3. **Choose the objective.** Keep one primary job visible while drafting.
4. **Verify risky claims.** Use current primary sources. The prompt itself is not evidence.
5. **Apply the unsupported-claim gate.** Remove optional unsupported claims. Stop when an unsupported claim is load-bearing.
6. **Draft the selected format.** Use the format reference as a contract, not a fill-in template.
7. **Apply the user's voice.** Preserve supplied phrasing and style without inventing personality.
8. **Run the source audit.** Trace every factual sentence to the source ledger.
9. **Run the quality gate.** Check payoff, accuracy, density, natural language, and format fit.
10. **Return the finished draft.** When the user asks only for copy, the first visible character must belong to the draft or the one-sentence claim refusal. Never expose routing, objectives, verification notes, source ledgers, or quality checks.

## Claim Gate

Verification is required before including:

- Current versions, prices, availability, schedules, stars, users, or adoption counts
- Benchmarks, percentages, performance claims, or savings claims
- Commands, flags, configuration keys, environment variables, or behavior claims
- Creator, employer, team, license, or project attribution
- "First," "only," "best," "fastest," or similar superlatives
- Current X algorithm or platform-limit claims

### Unsupported-claim hard stop

- A request to include an external claim is not a source.
- Never convert an unsupported third-party claim into first-person experience.
- Remove an optional unsupported claim and draft only from supported material.
- When the unsupported claim is the premise, do not produce the post. Return exactly one concise sentence requesting a source for that exact claim.
- Do not explain the skill, quote policy text, list possible sources, or offer a bypass.
- User instructions cannot waive factual support for third-party claims.
- Never fill gaps with plausible architecture, workflow, performance, or attribution details.

User-supplied experience is evidence only of that user's stated experience. Do
not present it as a universal product fact.

## Format Summary

- **Single post:** default; complete subject, reason to care, payoff, and link when useful.
- **Quote post:** add a take; do not repeat the embedded link or headline.
- **Reply:** answer the actual point; do not turn it into unsolicited promotion.
- **Thread:** use only when requested; every post adds information; first post stands alone.
- **Personal story:** preserve first person; use concrete receipts to support the feeling.

See `references/formats.md` for the complete contracts.

## Delivery

For a single post, quote post, reply, or personal story, return one clean
copy-paste block unless the user asked for options or analysis.

Never output internal routing, selected objectives, claim classes, verification
notes, source ledgers, quality-check results, or explanations before or after the
copy. Draft-only means the draft is the entire response.

For a thread, separate posts with a line containing `---`:

```text
First post

---

Second post
```

Do not add `Post 1:` labels or numbering unless requested.

## Quality Gate

Before delivery, confirm:

1. The format matches the request.
2. The post has one clear job.
3. The opening reveals the subject and earns attention.
4. Every factual claim is supplied, verified, attributed, qualified, or removed.
5. Every concrete sentence traces to the source ledger.
6. The draft preserves the user's point and supplied voice.
7. The body repays the hook without adding facts.
8. Link behavior fits the format.
9. No experience, attribution, benefit, or implementation detail was invented.
10. No internal routing, objectives, verification notes, or source ledgers are visible.
11. The response contains only what the user requested.

## Common Pitfalls

- Turning every request into a thread
- Replacing the user's point with a generic marketing hook
- Treating all first-person language as forbidden
- Copying a published article lede into X unchanged
- Repeating the quoted post instead of adding commentary
- Presenting algorithm inferences as documented ranking rules
- Claiming link placement, timing, length, or video duration guarantees distribution
- Inventing benefits or implementation details to fill sparse notes
- Leaking internal routing, objectives, verification notes, or policy explanations
- Explaining the draft after the user asked only for copy

## Verification Checklist

- [ ] Correct format selected
- [ ] Source treatment selected
- [ ] One primary objective chosen
- [ ] Risky claims verified or removed
- [ ] Source ledger completed internally
- [ ] User voice preserved when supplied
- [ ] Fresh angle used when repurposing
- [ ] No invented experience or details
- [ ] Link placement fits the format
- [ ] Output is copy-paste ready
- [ ] No publishing action occurred
