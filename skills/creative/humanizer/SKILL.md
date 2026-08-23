---
name: humanizer
description: Use when humanizing, de-AIing, or editing text to remove LLM tells while preserving meaning, audience, and the writer's voice.
version: 2.5.2
author: Siqi Chen (@blader), ported by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  author: Siqi Chen (@blader), ported by Hermes Agent
  tags: [writing, editing, humanize, anti-ai-slop, voice]
  category: creative
  homepage: https://github.com/blader/humanizer
  related_skills: [songwriting-and-ai-music]
---

## Purpose

Make supplied prose sound like a specific person rather than a language model. Preserve facts, intent, audience, and requested tone; remove formulaic AI language without sanding away personality. Apply the same pass to user-facing summaries, docs, release notes, and PR descriptions you write yourself.

## When to use this skill

Load it for requests to humanize, de-AI, de-slop, or un-ChatGPT text; rewrite a draft naturally; match a voice sample; or review prose for AI tells before publishing. If no text was supplied, ask for the text and, when useful, a voice sample. Do not pretend an edit occurred.

## Inputs and prerequisites

- Inline text: rewrite in the same turn.
- File path: read the complete file first, then use a targeted patch or full write and show the changed section/diff.
- Voice sample: read it before editing and note sentence rhythm, vocabulary, paragraph openings, punctuation, recurring phrases, and transitions.
- No API key or special dependency is required. Keep named facts, studies, quotes, and citations only when supplied or verified; do not invent plausible examples.

## Execution gate

1. Read the source carefully and state the intended audience and tone internally.
2. Scan for the pattern groups in `references/full-humanizer-patterns.md`: significance inflation, media/notability padding, promotional language, vague attribution, superficial `-ing` clauses, AI vocabulary, copula avoidance, negative parallelism, rule-of-three lists, synonym cycling, false ranges, passive fragments, em-dash/bold/emoji/header habits, chatbot artifacts, hedging, generic conclusions, metaphors, dramatic fragments, rhetorical questions, opener tics, and reassurance kickers.
3. Replace each problem with the plain, specific sentence the source actually supports. Prefer active subjects and simple `is/are/has` constructions. Delete filler rather than replacing it with a synonym.
4. Add personality only where appropriate: vary rhythm, include a grounded opinion or first person when the source warrants it, acknowledge uncertainty or mixed feelings, and keep a few natural asides. Do not manufacture quirks.
5. Preserve meaning and evidence. Do not strengthen claims, change chronology, add sources, or turn placeholders into facts.
6. Read the result aloud mentally. Look for repeated sentence shapes, tidy paragraph symmetry, slogans, unexplained metaphors, and remaining chatbot language.
7. Revise once more after answering: “What still makes this obviously AI-generated?”

## Validation and error handling

Before returning, compare the rewrite with the source for omitted claims, altered numbers, changed names, unsupported certainty, and audience mismatch. If the source is incomplete, contradictory, or citation-dependent, flag the gap instead of guessing. If a voice sample conflicts with explicit tone instructions, follow the user's current instruction and say so briefly. For file edits, verify the file exists and show a diff or changed section; do not silently overwrite.

## Output format

Return the final rewrite first. For a substantive edit, also provide:

1. a brief list of material changes;
2. a short “What still makes this obviously AI generated?” audit, if anything remains;
3. the final revised version after that audit.

For a simple inline sentence, skip ceremony and return the improved text plus one short note if needed.

## Examples

### Inline rewrite

```text
User: Humanize: “Additionally, this pivotal update showcases a seamless, powerful workflow.”
Assistant: “This update makes the workflow easier to use.”
Changes: removed filler, significance inflation, and promotional wording.
```

### Voice calibration

```text
Read the user's sample first. If it uses short sentences and plain words, keep that rhythm. Do not upgrade “stuff” to “components” or add polished transitions the sample never uses.
```

### Safe file edit

```text
Read the file, identify the affected paragraphs, patch only those paragraphs, verify the file remains readable, and show the diff. Keep citations and factual claims unchanged unless the user asks for fact-checking.
```

## References

The complete 34-pattern catalog, before/after examples, full worked example, attribution, and detailed source material are preserved in `references/full-humanizer-patterns.md`. Consult it for a full audit; this top-level file keeps the execution gate concise.

## Limitations

Humanization is editorial judgment, not proof that text was written by a person. It cannot verify facts or identify an author's private voice without source material. Review the result before publication.
