---
name: large-context-rlm
description: "Use when a task involves large files, massive diffs, long documents, many files, or corpus-scale review. Orchestrate recursive_context with search/read/map/delegate/synthesis instead of pasting huge inputs into model context."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [large-context, rlm, recursive-context, corpus, research, delegation]
    category: research
    related_skills: [subagent-driven-development, requesting-code-review, test-driven-development]
---

# Large-Context RLM Workflow

## Overview

Use `recursive_context` as the default large-context substrate. The point is not
"read a huge thing and hope the model remembers it." The point is to externalize
large source material into a durable corpus, navigate it with bounded line reads,
parallelize chunk work when useful, then synthesize only evidence-backed claims.

This skill turns `recursive_context` from a power tool into a repeatable RLM-style
workflow: create corpus -> search first -> read bounded windows -> map chunks ->
`delegate_task` -> synthesize -> verify citations.

## When to Use

Load and follow this skill when the user asks to work with any of these:

- large files that would exceed normal `read_file` comfort
- massive diffs or broad repository reviews
- long documents, transcripts, PDFs converted to text, logs, exports, or reports
- many files that need combined analysis
- "review all of this", "summarize the corpus", "find themes", "extract claims",
  "audit these docs", "compare sources", or similar corpus-scale tasks
- any task where direct context stuffing would be tempting

Do not use this skill for tiny inputs that fit cleanly in a single `read_file`
window. Do not paste giant content into the conversation. Use the corpus.

## Core Rule: Do Not Paste the Ocean

If source material is large, do not paste it into prompts, subagent contexts, or
final answers. Store it once with `recursive_context.create`, then pass corpus IDs
and line ranges.

Bad:

```text
Here are 80,000 lines. Summarize them...
```

Good:

```text
Corpus: annual-logs-a1b2c3d4e5f6
Read with recursive_context(action="read", corpus_id="annual-logs-a1b2c3d4e5f6", start_line=1200, line_count=80)
Return claims with corpus lines and source-line citations.
```

## Standard Orchestration Loop

### 1. Create corpus

For raw text:

```python
recursive_context(action="create", name="task-slug", text=large_text, chunk_lines=200)
```

For files:

```python
recursive_context(action="create", name="task-slug", paths=["/abs/path/a.md", "/abs/path/b.log"], chunk_lines=200)
```

Record the returned `corpus_id`, total lines, sources, and chunk count.

### 2. Search first

Before reading broadly, search for likely anchors:

```python
recursive_context(action="search", corpus_id=corpus_id, query="topic terms", limit=20, context_lines=2)
```

Use several focused searches rather than one vague query. If search misses but
the task requires full coverage, continue to mapping instead of pretending search
is exhaustive.

### 3. Read bounded windows

Read only the regions needed for reasoning:

```python
recursive_context(action="read", corpus_id=corpus_id, start_line=hit_line - 20, line_count=80)
```

Rules:

- Keep windows small enough to reason about.
- Expand windows only when the local context is insufficient.
- Preserve corpus line, source path, and source_line in notes.
- Never cite a claim from memory if you did not read the supporting lines.

### 4. Map chunks

For full-corpus tasks, ask the tool to emit chunk prompts:

```python
recursive_context(
    action="map",
    corpus_id=corpus_id,
    task="Extract factual claims, contradictions, risks, and unresolved questions. Return source-line citations.",
    max_chunks=24,
)
```

Each mapped prompt must direct the worker to call `recursive_context.read` for its
assigned chunk. The chunk prompt is not the evidence; the read result is.

### 5. Delegate chunk work

Use `delegate_task` for independent chunks when the corpus is too large for one
agent pass. Give each worker only:

- the corpus ID
- its start/end lines
- the exact analysis task
- the citation contract below
- toolsets: usually `file` plus any task-specific tools; add `terminal` only if needed

Worker instructions:

```text
Read your assigned range with recursive_context(action="read", ...).
Do not infer beyond the lines you read.
Return bullets as:
- claim: ...
  evidence: corpus lines X-Y; source <path>:<source_line_start>-<source_line_end>
  confidence: high|medium|low
  caveat: ...
```

### 6. Synthesize

Merge worker outputs into a claim ledger. Group duplicates, contradictions, and
open questions. Prefer precise, cited statements over broad vibes.

Synthesis order:

1. List major findings.
2. Attach evidence for each finding.
3. Identify contradictions and weak evidence.
4. Separate source-backed facts from interpretation.
5. Produce the requested final artifact.

### 7. Verify citations

Before finalizing, sample or fully verify citations using `recursive_context.read`.
For high-stakes outputs, verify every material claim. For low-stakes summaries,
verify at least each major section and any surprising claim.

If a citation does not support the claim, fix the claim or remove it. No orphan
claims.

## Citation Contract

Every non-trivial claim derived from the corpus must carry at least one citation:

```text
[corpus lines 120-134; source /path/file.md:44-58]
```

When source files are unavailable (raw text input), cite corpus lines only:

```text
[corpus lines 120-134]
```

Citation levels:

- **Required:** facts, quotes/paraphrases, dates, numbers, specific assertions,
  contradictions, recommendations based on source text.
- **Optional:** section headers, obvious connective prose, user's own requested
  framing.
- **Forbidden:** citing a line that was not read, citing a chunk prompt instead
  of a read result, or citing a source line that does not actually support the
  claim.

## Claim Ledger

Use a temporary claim ledger for synthesis-heavy work:

```text
Finding: concise claim
Evidence:
- corpus lines A-B; source path:source_lines
- corpus lines C-D; source path:source_lines
Confidence: high|medium|low
Conflicts: corpus lines E-F, if any
Notes: interpretation boundary / caveat
```

The final answer can be cleaner than the ledger, but the ledger disciplines the
reasoning.

## Coverage Check

Before declaring completion, run a coverage check appropriate to the task:

- Search coverage: did you search the obvious synonyms and entities?
- Chunk coverage: if the user asked for whole-corpus review, did mapped chunks
  cover the corpus or did you intentionally sample?
- Source coverage: if multiple files/sources were ingested, did each relevant
  source get considered?
- Citation coverage: do major claims cite corpus-line and source-line evidence?
- Negative coverage: if you say something was absent, did you search/read enough
  to justify absence?

Do not claim exhaustive review unless the chunks actually covered every relevant
line or you explicitly state the sampling boundary.

## Quality Gates

### Pre-flight gate

Before corpus creation:

- Confirm input source is authorized to read.
- Prefer file paths over pasted text when files exist.
- Choose a descriptive corpus name.
- Choose `chunk_lines` based on source density: 100 for dense/legal/code, 200 for
  normal prose, 400 for sparse logs.

### Revision gate

After search/read/map:

- If results are thin, broaden search terms or increase coverage.
- If workers return uncited claims, send them back or discard those claims.
- If sources conflict, preserve the conflict instead of flattening it.

### Final gate

Before final answer:

- Verify citations.
- Remove unsupported claims.
- Separate facts from interpretation.
- Mention corpus/sampling limits honestly.

## Failure Modes

1. **Context stuffing relapse.** You start pasting whole documents into prompts.
   Stop, create a corpus, and pass line ranges.

2. **Search tunnel vision.** You only read search hits and miss full-corpus themes.
   Use `map` for coverage when the task is corpus-wide.

3. **Citation laundering.** A worker cites a range but the range does not support
   the claim. Verify citations before finalizing.

4. **Chunk hallucination.** A worker summarizes from the chunk prompt instead of
   calling `recursive_context.read`. Reject outputs without read-backed evidence.

5. **False exhaustiveness.** You say "no evidence" after one search. Either run
   broader searches/read coverage or say "not found in sampled/searched regions."

6. **Redaction surprise.** `recursive_context` redacts likely secrets before
   durable storage. If exact credential values are the task, stop and ask for a
   safe workflow; do not try to bypass redaction.

7. **Stale corpus assumption.** A corpus is a snapshot. If source files changed,
   recreate the corpus rather than assuming it updated itself.

## Output Patterns

### Concise answer

```text
Finding: ... [corpus lines 10-14; source /tmp/a.md:3-7]
Finding: ... [corpus lines 80-91; source /tmp/b.md:12-23]
Caveat: coverage was limited to mapped chunks 1-6 of 10.
```

### Audit report

```text
Scope:
- Corpus: <corpus_id>
- Sources: <count>
- Coverage: chunks 1-N / all lines M

Findings:
1. ...
   Evidence: corpus lines ...; source ...
   Severity: high|medium|low
   Fix: ...

Unresolved:
- ...
```

### Research synthesis

```text
Thesis:
- ... [evidence]

Supporting evidence:
- ... [evidence]

Contradictions:
- Source A says ... [evidence]
- Source B says ... [evidence]

What would change my mind:
- ...
```

## Verification Checklist

- [ ] Created or reused the correct corpus snapshot.
- [ ] Used search first for targeted questions.
- [ ] Used map chunks for whole-corpus tasks.
- [ ] Used `delegate_task` when chunk work is independent and large enough.
- [ ] Maintained a claim ledger for synthesis-heavy work.
- [ ] Verified citations against `recursive_context.read`.
- [ ] Included corpus-line citations and source-line citations where available.
- [ ] Stated sampling/coverage limits honestly.
- [ ] Did not paste huge source material into prompts or final output.
