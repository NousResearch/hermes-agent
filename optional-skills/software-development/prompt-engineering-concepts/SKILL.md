---
name: prompt-engineering-concepts
description: Pick the right prompting technique for a task.
version: 1.0.0
author: Joerg Peetz (JPeetz), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [prompting, chain-of-thought, few-shot, react, prompt-selection, evaluation]
    category: software-development
    related_skills: [grill-me]
when_to_use:
  - Writing or rewriting a system prompt, a skill body, or a long task prompt
  - A prompt returns vague, wrong-shaped, or unreliable output and you need the specific fix
  - Choosing between zero-shot, few-shot, chain-of-thought, tree-of-thought, and ReAct
  - Designing tool-use loops, multi-stage workflows, or parseable output contracts
when_not_to_use:
  - Prompt-injection or safety hardening (delimiter wrapping, refusal rules, untrusted-content policy)
  - Judge design, rubric calibration, and eval harness plumbing
  - Single-turn factual questions where no prompt is being authored
---

# Prompt Engineering Concepts Skill

Maps the 20 core prompting techniques to the situations where each one earns its
tokens, and gives the exact move for each. It is a selection reference: it helps
choose and correct a technique. It does not cover injection defense, judge
calibration, or eval-harness design.

## When to Use

- Authoring or repairing a system prompt, skill body, or long task prompt
- Diagnosing one specific failure: vague output, wrong shape, unreliable
  reasoning, wrong tone, token bloat, or unparseable JSON
- Deciding between zero-shot, few-shot, chain-of-thought, tree-of-thought, ReAct
- Splitting an overloaded prompt into stages, or choosing what to sample and what
  to make deterministic

Don't use for: injection/safety hardening, judge calibration (that is eval work),
or as a substitute for running the prompt and reading the output.

## Prerequisites

None. No scripts, no network, no credentials — the skill is a reference table plus
a procedure. It applies to any provider or model; technique choice is independent
of vendor.

## How to Run

Load it with `skill_view(name="prompt-engineering-concepts")` when a prompt is
being written or debugged, then follow the procedure below. There is nothing to
execute: the deliverable is the revised prompt and the evidence from running it.

## Quick Reference

| Reach for | When |
|---|---|
| Zero-shot | Common, unambiguous, one-shot task |
| Few-shot | Format or judgment is easier to show than describe (1–5 pairs) |
| Chain-of-thought | Math, logic, multi-step reasoning, planning |
| Role prompting | Tone, register, or domain expertise matters |
| Instruction prompting | Output is vague because the ask is vague |
| Output formatting | Output is parsed, or must be consistent |
| Temperature control | Consistency vs. creativity — not a prompt fix |
| Prompt chaining | Distinct phases with different goals |
| Self-consistency | One sample is unreliable on reasoning |
| RAG | Answer must come from specific documents |
| Prompt templates | Same shape reused across inputs |
| Guardrails | User-facing, untrusted input, or side effects |
| Dynamic prompting | One prompt must serve beginner/expert/pro |
| ReAct | Real tasks needing tools or environment feedback |
| Tree-of-thought | Many plausible paths, one-shot reasoning stalls |
| Prompt compression | Long context, repeated calls, token bills |
| Multi-modal | The answer depends on an image or diagram |
| JSON mode | Output feeds code |
| Prompt evaluation | You cannot tell whether a change helped |
| Prompt optimization | It works but is not good enough |

## Procedure

1. **Name the failure mode** — vague output, wrong format, unreliable reasoning,
   wrong tone, token bloat, or unparsed JSON. Each maps to a different technique.
2. **Pick the smallest technique set** — one or two concepts, not six. Every added
   technique costs tokens and hides the cause of a regression.
3. **Write the prompt** with the task stated plainly and all never-rules grouped
   together, ahead of the task instructions.
4. **Run it on at least three real inputs** and read the outputs. Completion
   criterion: three inputs, three inspected outputs.
5. **Change one variable at a time**, re-run, keep the better version. Completion
   criterion: one recorded change per iteration, with the reason.
6. **Lock the shape** of anything machine-read by validating it in code — schema
   check or parse-and-catch — never by eye. Completion criterion: the validator
   accepts the output or the prompt is rejected.

## Pitfalls

1. **Few-shot examples that disagree with the instructions** — the examples win.
   Audit them (wrong label, wrong format) before blaming the model.
2. **Chain-of-thought on tasks that do not need it** — costs latency and tokens on
   lookups and translation, and flattens creative voice.
3. **Guardrails buried at the bottom** — constraints must precede the task
   instructions, or later content overrides them.
4. **"Output JSON" with no schema** — the model invents keys. Ship the exact shape
   plus one example.
5. **Self-consistency on a non-reasoning task** — N× the cost for noise; it only
   pays where independent samples can genuinely disagree.
6. **Chaining without a contract between stages** — each stage needs a defined
   output shape, or errors compound silently.
7. **Changing several prompt elements at once** — no attribution, no learning.
8. **Trusting a "better" prompt without running it** — a prompt is a hypothesis
   until outputs are scored on real inputs.

## Verification

- Three real inputs ran, and their outputs were read
- Parsed output validated programmatically where the prompt feeds code
- Format compliance checked against the requested spec
- Token cost compared to the previous version after compressing or chaining
- Where reliability matters: N runs, success rate recorded, one variable per pass

## Reference

- `references/20-concepts-cheatsheet.md` — the source poster's 20 techniques
  transcribed with their use cases, plus the diagram note for each.
