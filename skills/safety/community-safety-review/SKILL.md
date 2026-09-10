---
name: community-safety-review
description: Helps human moderators review suspicious claims, unsafe links, impersonation signs, unclear ownership, and misleading messages in a Hermes Agent community. Produces neutral safety notes only. Never accuses anyone and never enforces. Human review is always required.
---

# Community Safety Review Skill for Hermes

## Overview

This skill helps a Hermes Agent assist **human moderators** and community
managers when they need to review a potentially suspicious message, project
claim, or link inside a community space.

The skill does **not** make decisions. It produces a structured, neutral
**safety note** that a human can read, verify, and act on. Every output ends
with a reminder that human review is required.

The goal is to make community safety review more organized, calmer, and
evidence-based — not faster accusations.

## Problem statement

Communities receive many messages every day. Some of them include:

- fake official claims
- misleading project names
- unsafe or suspicious links
- impersonation signs
- unclear ownership
- pressure-style wording
- missing verification sources
- mismatched usernames, domains, or branding
- claims that cannot be verified from public sources

Moderators have limited time and often review these messages in a hurry. A
hurried review can either miss a real problem or, worse, label an innocent
person unfairly. Both outcomes hurt the community.

This skill gives moderators a consistent, neutral structure so they can review
suspicious messages carefully and keep a written record of *why* something
looked unclear — without jumping to conclusions.

## When to use this skill

Use this skill when a human moderator wants help to:

- review a message that claims to be from an official source
- check whether a link or domain looks suspicious or mismatched
- look for impersonation signs (copied names, avatars, or branding)
- organize notes about a project whose ownership is unclear
- summarize why a message feels misleading, using observable evidence
- prepare a neutral handoff note for other moderators

## When not to use this skill

Do **not** use this skill to:

- automatically ban, mute, kick, or punish anyone
- make a final decision about whether a person is malicious
- declare someone a "scammer" or any other accusation
- replace human judgment or community rules
- review private personal data that the moderator is not authorized to handle
- act as legal, security, or identity-verification authority

This skill is an assistant for human review. It is **not** an enforcement tool.

## Expected inputs

A moderator may provide some or all of the following:

- the message text being reviewed (or a neutral description of it)
- the claimed identity or role (for example, "claims to be an admin")
- any links, domains, or handles included in the message
- any public verification sources the moderator already checked
- context about where and when the message appeared

Inputs may be incomplete. Missing information is expected and should be recorded
as missing, not guessed.

## Expected outputs

A single, neutral **safety note** based on the template in
`templates/safety-review.md`. The note contains:

- what is being reviewed
- the claimed identity
- the links provided
- the verification sources checked
- any suspicious signals observed (with the evidence for each)
- missing information
- a risk level: LOW / MEDIUM / HIGH / UNKNOWN
- a suggested moderator action (a review suggestion, never an enforcement order)
- a short neutral summary
- a human review disclaimer

The output is meant to be pasted into a moderator note, not posted publicly as
an accusation.

## Review workflow

1. **Collect** the available inputs from the moderator. Record what is present
   and what is missing.
2. **Restate** the claimed identity and the message neutrally, without judgment.
3. **List the links** exactly as provided. Note any mismatched domains,
   look-alike spellings, or branding that does not match the claimed source.
4. **Check verification sources.** Record which public sources were checked and
   what they showed. If none were checked, say so.
5. **Scan for suspicious signals** using the checklist below. For each signal
   found, write the specific observable evidence next to it.
6. **Record missing information** that a human would need to reach a conclusion.
7. **Assign a risk level** using the framework below, based only on observable
   evidence.
8. **Suggest a moderator action** framed as a review step, never as a
   punishment.
9. **Write a neutral summary** and attach the human review disclaimer.

## Suspicious signal checklist

Look for, and record evidence for, any of the following:

- **Fake official claims** — claims to be staff, admin, or an official source
  without a verifiable basis.
- **Misleading project names** — a name that imitates a known, trusted project.
- **Unsafe or suspicious links** — shortened, obfuscated, or look-alike URLs.
- **Impersonation signs** — copied display names, avatars, or branding.
- **Unclear ownership** — no public, verifiable owner or maintainer.
- **Pressure-style wording** — urgency, secrecy, or "act now" framing.
- **Missing verification sources** — no public evidence supports the claim.
- **Mismatched usernames, domains, or branding** — the handle, domain, and
  stated identity do not line up.
- **Unverifiable claims** — statements that cannot be confirmed from public
  sources.

A signal should only be recorded when there is observable evidence for it.
Do not record a signal based on a feeling alone.

## Risk level framework

- **LOW** — No obvious suspicious signal was found, but human review is still
  recommended.
- **MEDIUM** — Some information is unclear or missing. Manual verification is
  needed before any conclusion.
- **HIGH** — Strong suspicious signals exist, such as impersonation signs,
  unsafe links, fake official claims, or pressure-style wording. Human review is
  strongly recommended before any action.
- **UNKNOWN** — There is not enough information to make a useful safety note.

A risk level describes *how much careful review is recommended*. It is never a
verdict about a person.

## Moderator handoff notes

The safety note is designed to be handed to another human. When writing the
handoff:

- keep the tone calm and factual
- separate observed evidence from interpretation
- always list what is still missing
- suggest a next review step, not a punishment
- make it easy for the next moderator to disagree and re-check

## Safety notes

- This skill never accuses anyone. It describes observable signals only.
- It never recommends an automatic ban or enforcement action.
- It always states that human review is required.
- When evidence is absent, it records UNKNOWN instead of guessing.
- It does not invent verification sources or links that were not provided.
- It treats the reviewed person as innocent until a human, using community
  rules, decides otherwise.

## Output style rules

- Stay neutral. Never call anyone a "scammer" or any similar label.
- Prefer soft, evidence-based phrasing such as "suspicious signal detected"
  or "this claim could not be verified from public sources."
- Always include "human review required."
- If there is no evidence, do not speculate.
- If information is missing, write UNKNOWN rather than filling the gap.
- Describe the message and the claim; do not pass judgment on the person.

## Examples of safe output

Safe, neutral phrasing:

- "Suspicious signal detected: the link domain does not match the claimed
  official source. Human review required."
- "Claimed identity could not be verified from public sources. Risk level:
  MEDIUM. Manual verification is recommended."
- "No obvious suspicious signal was found, but human review is still
  recommended. Risk level: LOW."
- "Not enough information was provided to write a useful safety note. Risk
  level: UNKNOWN."

Phrasing to avoid:

- "This person is a scammer." (accusation)
- "Ban this user." (enforcement)
- "This is definitely fake." (conclusion without human review)

## Project safety review helper (optional script)

In addition to the message-review workflow above, the skill ships a small
optional helper for reviewing a **Web3/GameFi project** from its public
information. It is a lightweight, dependency-free Python script that reads a
JSON description of a project and produces the same kind of neutral safety
note — automatically detecting public risk signals and trust signals.

The helper keeps every principle of this skill: public signals only, no private
data, no network calls, no accusations, no enforcement, and a manual-verification
reminder on every report. The review level it assigns describes *how much
careful human review is recommended* — it is never a verdict and never claims a
project is a scam.

### Files

- `scripts/safety_review.py` — the analyzer (Python 3.8+, standard library only).
- `scripts/sample-project.json` — a sample input.
- `templates/project-safety-review.md` — a blank note to fill in by hand.
- `reports/sample-report.md` — a committed example of generated output.

### How to run

```bash
cd scripts
python safety_review.py sample-project.json            # print the note
python safety_review.py sample-project.json --report   # also save to ../reports/
cat project.json | python safety_review.py -           # read from stdin
```

### Input

A single JSON object. All fields are optional — provide what is public and
known. Free-text fields (`description`, `marketing_text`, `documentation`,
`roadmap`) are scanned for risky language; structured fields record presence or
clarity. The `reviewer_flags` list lets a human record signals that text
scanning cannot see (`broken_links`, `copied_content`, `fake_social_proof`,
`fake_partnership`, `impersonation`). See the header of `safety_review.py` for
the full field list.

### Risk signals it checks

Anonymous/unclear team, no public documentation, no working product/demo,
unrealistic reward claims, guaranteed-profit language, aggressive referral
focus, unverified partnership claims, suspicious token-sale wording, unclear
tokenomics, no/inactive GitHub, copied content (reviewer flag), broken links
(reviewer flag), no audit/security information, pressure tactics, and possible
fabricated social proof.

### Trust signals it checks

Documentation available, working demo/product, public team or named
contributors, GitHub provided/active, community activity described, transparent
tokenomics, audit/security notes, roadmap provided, and absence of
guaranteed-profit language.

### Review levels

- **LOW** — few/no risk signals; human review still recommended.
- **MEDIUM** — some signals or gaps; manual verification needed.
- **HIGH** — multiple notable signals; careful human review strongly recommended.
- **HUMAN REVIEW REQUIRED** — not enough public information to assess.

## Future automation ideas

Further ideas that would keep the same principles:

- An optional helper that highlights look-alike domains for a reviewer to inspect.
- An optional checklist prompt that walks a reviewer through the signals.
- An optional log format so a community can keep a neutral history of past
  safety notes for transparency.

Any future automation must keep the same principles: neutral output, no
accusations, no enforcement, and human review always required.
