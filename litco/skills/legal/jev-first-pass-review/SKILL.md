---
name: jev-first-pass-review
description: Decide on and run a Jev first pass before review.
version: 0.1.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litkit, ediscovery, review, jev, typesafe, screening]
    related_skills: [litkit-review-and-tag, litkit-corpus-pull]
---

# Jev First-Pass Review Skill

Jev is TypeSafe's System One model. It answers typed questions about a text with calibrated probabilities, and it writes no prose. Ana uses it to decide which documents the full review reads first and which it sets aside for now. Jev never writes a tag, a finding, or a privilege call; the full LLM review and the lawyer do that.

## When to Use

Read this skill before any review that reads more than a handful of documents, and decide whether a first pass applies. The decision is a required step of `litkit-review-and-tag`.

A first pass helps when both of these hold:

- The scope is broad: more than a few hundred documents.
- The criteria are topical: responsiveness to numbered requests, or relevance to named issues. A yes/no on "is this about rebates in 2023?" is what Jev answers well.

A first pass does not help, and Ana proposes the run without one (`firstPass: {enabled: false}`), when any of these holds:

- The criteria turn on nuance a yes/no cannot carry, such as "evidence of intent," "inconsistent with the witness's testimony," or "admission against interest." Jev would set aside documents whose relevance only a careful reader sees.
- The scope is small (a few dozen documents). The full review costs little, and the screen saves nothing.
- Privilege review is the goal. Jev may flag a privilege signal, and that signal keeps a document in the full review. Jev never decides privilege.
- The corpus is mostly not English. Jev's accuracy is lower outside English. Say so to the lawyer, and if a first pass still runs, lower `low` (0.05 to 0.10) so fewer documents are set aside.

## Prerequisites

- The `litkit` toolset on a matter host, with `TYPESAFE_API_KEY` in the host's environment. Without the key, `litkit_jev` returns "Jev not configured on this host"; tell the lawyer, and propose the run without a first pass.
- Jev reads text only. A document with no extracted text is never screened; it goes to the full review.

## How to Run

| Scope | Tool | What happens |
|---|---|---|
| A review run at any size | `litkit_review` action=create with `firstPass` | LitKit screens every document on the server before the full review. The set-aside list, with each document's probabilities and a Promote action, appears on the job page. |
| A quick look at up to 200 documents | `litkit_jev` action=screen | One Jev request per document; counts and per-document rows come back in the turn; every probability is saved under `jev/`. |
| One document, any typed question | `litkit_jev` action=ask | Jev's raw answers to up to 20 questions (noul, choice, score), with confidence on choice and score answers. |

## Quick Reference

The cull rule runs in code, never in Jev. A document is set aside only when all three hold:

- `responsive` ≤ `low`;
- every criterion (`c_0`, `c_1`, ...) ≤ `low`;
- `privileged_signal` ≤ `priv`.

The defaults are `low` 0.15 and `priv` 0.30. Every other document is read in full, including any document Jev could not read or that returned an error. `junk` is reported for triage and never sets a document aside by itself.

A document is uncertain when a move of 0.1 in its probabilities would flip the decision. `litkit_jev screen` counts these as `uncertain` and `uncertain_set_aside`.

Jev's price is $0.042 per million input tokens; output is free. A 4,000-document screen at about 8,000 tokens a document is about 32 million tokens, roughly $1.35, before the full review's own cost.

## Procedure

1. **Decide.** Apply When to Use to the lawyer's request. Note the reason in one sentence; you will give it in the proposal.
2. **Pick the thresholds.** Start from 0.15 and 0.30. Thresholds scale with the stakes of a wrong set-aside. For a production deadline where a missed responsive document is a discovery violation, lower `low` to 0.05–0.10. For an internal issue scan where speed matters more, 0.20 is defensible. Never raise `priv` above 0.30.
3. **At scale, propose the run with the first pass.** Call `litkit_review` action=create as `litkit-review-and-tag` describes, with `firstPass: {enabled: true, thresholds: {low, priv}}`. Omit `firstPass` only when the server default (on when Jev is configured) is what you decided. Name the thresholds in the proposal: "First pass: Jev, set aside at responsive ≤ 0.15 and privilege signal ≤ 0.30; then full review."
4. **For a quick look, screen.** Call `litkit_jev` action=screen with `documentIds`, `bates`, or `workSetId`, and `criteria` or `criteriaSetId`. Over 200 documents, the tool refuses and points to step 3; follow it.
5. **Report the counts.** After a screen, or when a first-pass run finishes, give the lawyer the counts in one sentence: "Jev set aside 1,240 of 4,000; 2,760 read in full; 85 uncertain, and I kept the uncertain ones in the full review."
6. **Route by band.** Treat the three bands differently:
   - Clear set-aside: set it aside for now. The set-aside list stays on the job page, and the lawyer can promote any document back.
   - Uncertain: read it in full, or put it in front of the lawyer. Do not let a close call decide what goes unread.
   - Clear read: read it in full.
7. **Use `ask` for one document's triage.** Batch every question about that document into one call: "Does `document` seek or give legal advice?" (noul), "Which custodian group wrote `document`?" (choice), "How hostile is the tone of `document`?" (score). On choice and score answers, a confidence below 0.5 means Jev does not know; read the document yourself.
8. **Rank when the lawyer wants the best first.** Sort the screen's rows by `responsive` and the top criterion probability to choose reading order. The ranking orders reading; it decides nothing.

## Pitfalls

- **Jev's numbers are not findings.** Never report "Jev found 412 responsive documents." Report what the full review found, and describe Jev only as the screen that chose what it read.
- **Set aside is not deleted.** A set-aside document is unread, not non-responsive. Say "set aside" and never "excluded" or "non-responsive."
- **Structure the state.** The document goes under a named key (`document`, with its Bates, custodian, date, and text), and criteria go in named fields of the question's `instructions` object. A bare string of document plus criteria blurs what Jev is judging.
- **Never send more than Jev takes.** Jev's limit is 32,000 tokens for the state plus the longest question. The tools cut long documents to their head and tail within 30,000 and flag `truncated: true`. A truncated document that is set aside deserves a second look if its middle may matter, as in a long contract.
- **Batch questions.** Questions about one document go in one call. TypeSafe measured one batched call as about 12 times cheaper than separate calls, and 10 times faster.
- **No Jev loops in a turn.** Never call `litkit_jev` over thousands of documents in a loop. Propose a review run with `firstPass`; the server screens at scale with its own rate handling.
- **Privilege stays human.** A high `privileged_signal` keeps a document in the full review. A low one does not clear a document for production.

## Verification

- The proposal or the report names the thresholds and gives the three counts: set aside, read in full, and uncertain.
- No sentence to the lawyer presents a Jev probability as a finding, a tag, or a privilege call.
- Every document with no text, an error, or an uncertain decision went to the full review.

## Worked Example

The lawyer asks: "Run Part 11 items 1–6 over the 4,000 Crowder emails." The scope is broad, and the six requests are topical (pricing, rebates, distributor terms), so a first pass applies. Ana registers the tags and the criteria set, then proposes the run with `firstPass: {enabled: true, thresholds: {low: 0.15, priv: 0.30}}`. She tells the lawyer: "Proposed: Part 11 v1 over the Crowder emails (4,000 documents), first pass with Jev at responsive ≤ 0.15 and privilege signal ≤ 0.30, then full review; estimate on the card." When the run finishes, she reports: "Jev set aside 1,240 of 4,000; 2,760 were read in full, including 85 uncertain. The full review tagged 912 as responsive: item 1, 404; item 2, 288; and so on. The set-aside list is on the job page if you want any promoted."
