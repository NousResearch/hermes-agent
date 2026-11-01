---
name: request-scope-resolution
description: "Resolve vague requests via JEV, memory, then user."
version: 0.1.0
author: Joerg Peetz (JPeetz), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Scope, Disambiguation, Jev, Routing, Clarify]
    related_skills: [hermes-agent]
    homepage: https://docs.typesafe.ai/concepts/how-to-build-with-system-one
---

# Request Scope Resolution Skill

Resolve an ambiguous or under-defined request through layered evidence before acting, escalating to the user only after the decision and memory layers are exhausted. Turn vague intent into a clearly scoped, executable brief. Never proceed on a guessed scope when ambiguity would change the outcome — and never ask prematurely when a decision model plus the configured memory surfaces can resolve it.

## When to Use
- The request's goal, deliverable, boundary, or success criterion is unclear.
- More than one reasonable interpretation changes what would be built or done.
- You need a decision on how to route, and are unsure whether to consult memory, escalate, or just act.
- A task could belong to several workflows (heavy-dev / stuck / fresh-angle / simple / code-review / adversarial).
- Use it at the START of any task with scope ambiguity, before dumping tokens into an expensive generation pipeline.

Don't use for: fully-specified, low-risk tasks (act directly); decisions where guessing is cheap and reversible.

## Prerequisites
- `OPENROUTER_API_KEY` set in the environment (decision-model backend). Get one at https://openrouter.ai/keys.
- A System One decision model reachable via OpenRouter — the reference model is `typesafe/jev-1.13`
  (see https://docs.typesafe.ai/concepts/system-one). Any typed-decision model works; swap `MODEL` in
  `scripts/scope_resolver.py`. Ship the resolve as pure calls the agent invokes through the `terminal` tool.
- Your configured memory surfaces readable — Hermes `memory` tool, MeMex Zero RAG, HermesVault, or any
  MCP memory server in `# MCP Servers`. The ladder searches these for prior context on the topic.

## How to Run

Given a vague request, classify it and read the gate:

```
python3 <skill>/scripts/scope_resolver.py "user request text"
python3 <skill>/scripts/scope_resolver.py "text" --context "extra context bytes"
```

The first form returns `{category, confidence, complexity, verdict}`. Feed any memory excerpts you
surface back in via `--context`, which resets the `state` Jev evaluates.

## Quick Reference
- `scope_resolver.py "<request>"` → category + confidence + verdict (RESOLVED / NEEDS_CONTEXT / ASK_USER).
- `scope_resolver.py "<request>" --context "..."` → re-classify with injected memory context.
- Categories: `heavy-dev`, `stuck`, `fresh-angle`, `code-review`, `adversarial`, `simple`.
- Confidence gates: `>= 0.85` RESOLVED · `0.6–0.85` NEEDS_CONTEXT · `< 0.6` ASK_USER.

## Procedure
1. Capture the raw request text.
2. Run `scope_resolver.py` once (category Choice + complexity Score in a single Jev call).
3. Read `confidence`; branch: `>= 0.85` → route to execution (Claude authoring → DeepSeek/`execute_code`
   for heavy-dev); `0.6–0.85` and `ASK_USER` proceed to memory.
4. Search memory surfaces in order: `memory` tool, MeMex, HermesVault, then any MCP memory server.
   If salient context is found, re-run `scope_resolver.py` with it in `--context`.
5. Re-gate. Still `ASK_USER` after memory → ask the user ONE or TWO bounded clarifying questions —
   the minimum that actually changes the build, never an open-ended "what do you want?".
6. On resolution, hand the scoped brief downstream.

## Hard Rules
- Never guess when ambiguity changes the outcome. A wrong-scope heavy-dev task can waste a full
  authoring/execution cycle before the mistake is visible. Ask — but only after the decision model
  and memory are exhausted.
- Never ask prematurely. Resolve via decision model → memory FIRST; asking what those layers could
  have answered is a failure.
- Evidence, not habit, triggers the ask. Escalate on a low-confidence number, not because a task
  "feels" unclear. The confidence value is the authority.
- Use the decision model's `confidence`, not the bare `choice` or `category`. A choice at 0.5 confidence
  is indistinguishable from a guess; gate on confidence.
- Batch the questions. One Jev call with a Choice + Score (and optionally a Noul) is ~10x cheaper than
  sequential calls; the script already does this.
- Keep `related_skills` and paths to in-repo surfaces only; never reference user-local skills.

## Pitfalls
- A System One model is not an LLM: it returns typed decisions + probabilities, never prose. It classifies;
  it cannot write the detailed execution prompt (a capable LLM does that) and it cannot recall across time
  (your memory surfaces do that).
- For Jev specifically: it uses the `/api/alpha/decisions` endpoint, not `/chat/completions`; Score uses
  `criteria` (ordered array), the boolean primitive is `noul`, and Score output must be normalized by
  `(len(criteria) - 1)` before comparing thresholds.
- Confidence `1.0` ≠ correctness. Jev is calibrated across groups of predictions; a confident answer can
  still be wrong. For irreversible or high-stakes scopes, require memory confirmation or one user check.
- Don't re-run Jev in a loop praying for higher confidence. Two attempts with real added context is the
  cap; beyond that, escalate — the model is telling you the input is ambiguous.

## Verification
- A clear request returns `{category, verdict: "RESOLVED"}` with confidence `>= 0.85`.
- A genuinely ambiguous request (no memory hits, confidence stays `< 0.6` after re-ask) produces a
  targeted user question — not a guess and not paralysis.
- Order was honored: decision model → memory → ask user (NOT ask → memory).
