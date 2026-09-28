---
name: expanded-legal-letter-redlines
description: Expand a legal letter with tracked-change redlines.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, letter, redline, tracked-changes, docx]
    related_skills: [discovery-letter-brief, litkit-deliverable, legal-cite-check]
---

# Expanded Legal-Letter Redlines Skill

When a lawyer asks to "beef up" or strengthen a letter, deliver a substantively expanded version as Word tracked changes on their own draft. Expansion means developed facts and legal reasoning, not repetition.

## When to Use

- A request to strengthen, expand, or "beef up" a letter, with the lawyer's .docx attached or in the matter's files.

## Matter style note

Some matter teams welcome longer, substantively expanded redlines in this setting. Before drafting, check the matter memory (`litkit_recall` query="redline style") for the team's stated preference and follow it. The preference applies only to expanded letter redlines, never to chat replies or other assignments. When a lawyer states a preference, save it as a matter note (`litkit_remember` kind=style) without naming the person, or privately with scope=user if it is theirs alone.

## Procedure

1. Fetch the draft (`litkit_attachment` or `litkit_files` read) and preserve the baseline: an untouched copy plus its sha256.
2. Develop supporting facts from the record (produced documents via `litkit_search` and `litkit_text`, the matter's files, the correspondence) and legal reasoning from authority read on Westlaw, Lexis, or LitLex. Every added fact carries its source; a request for a stronger letter does not authorize unsupported allegations.
3. Apply the changes as real tracked changes (the redline recipe in `discovery-letter-brief`): rewritten sentences as one deletion plus one insertion, existing comments kept, doubts as comments on the affected sentence.
4. Validate accept-all and reject-all programmatically, render, and inspect.
5. Quote-check and deliver with `litkit-deliverable`: the redline, the clean copy, a PDF, and a note listing each addition with its source.

## Pitfalls

- Longer is not better by itself: cut filler, repetition, and restatement.
- Facts that could not be verified this session are softened and flagged in a comment, never asserted.

## Verification

- Accept-all equals the intended clean text; reject-all equals the baseline.
- Every new fact and citation has a source listed in the note.
