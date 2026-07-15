---
name: provider-validation-harness
description: Use when running the tier-0 provider compatibility smoke through real Hermes agent-loop turns and persisted SessionDB receipts. It is a narrow compatibility check, never Hermes qualification, replacement evidence, or routing authority.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [providers, validation, hermes, tool-calling, local-models, compatibility]
    related_skills: [hermes-agent, github-pr-workflow, requesting-code-review]
---

# Tier-0 Provider Compatibility Smoke

## Purpose and ownership

Use this skill when a user needs a narrow answer to: “Does this configured
provider/model execute these representative Hermes turns and leave valid
receipts?” The command runs real `hermes chat -Q` subprocesses and checks the
persisted SessionDB lineage. A raw `/v1/chat/completions` response is not a
Hermes receipt and cannot answer this question.

The initial harness was authored by Drew Schuyler. Preserve Drew-first credit
and authorship in follow-up changes; adaptations and receipt repairs belong in
separate commits.

This is a tier-0 compatibility smoke only. It is not full qualification, a
benchmark, replacement evidence, a leaderboard, or permission to route user
traffic. Candidate-vs-incumbent evaluation is a
separate planned lane and is not implemented by this command.

## When to use

Use it for:

- a configured provider/model or OpenAI-compatible local endpoint;
- a focused check that the real Hermes CLI, tool schemas, and session store work
  together;
- diagnosing a provider path before a separately approved evaluation.

Do not use it to claim that all Hermes tools, loaded context, skills, memory,
multi-turn continuity, compression, safety, performance, or production tasks
have passed. Do not put private workflows or secrets in prompts or receipts.

## Current command

The only supported command in this milestone is `validate`:

```bash
hermes providers validate \
  --provider PROVIDER \
  --model MODEL \
  --toolsets file \
  --suite agent-readiness \
  --out /tmp/hermes-provider-validation \
  --timeout 120
```

`--toolsets file` is intentional for this compatibility smoke. Do not present
it as full-harness coverage or expand it to other toolsets through this command.
There is no `evaluate`, `score`, `archive`, promotion, or automatic routing
command in this milestone.

Arguments:

- `--provider`: provider to exercise; omit to use the configured provider.
- `--model`: model to exercise; omit to use the configured model.
- `--toolsets`: toolsets for these turns; the compatibility mode defaults to
  `file`.
- `--suite`: currently `agent-readiness`, the frozen six-case compatibility
  smoke.
- `--out`: output directory for deterministic local receipts; omitted means a
  temporary directory.
- `--timeout`: per-case timeout in seconds; default `120`.

## Receipt contract

Every case must retain raw stdout and stderr, including partial output from a
timeout. A printed `session_id` or a plausible final answer is insufficient.
The case passes the receipt gate only when the harness successfully resolves
and loads that session from SessionDB, then writes the loaded messages to
`raw/<case>.session.json`. Missing or invalid session ids, load errors, and
timeouts fail honestly and write `raw/<case>.session-error.txt` when applicable.

Tool receipts are ordered records, not a list of names. Each record includes the
tool name, parsed arguments, result content, and normalized status. The recovery
case must prove, in order, a failed `read_file` for the expected missing path
followed by a successful `read_file` for the expected fixture path. The
abstention case must also prove that its forbidden output artifact does not
exist after the turn.

The final answer is read from the loaded session messages. Stdout never acts as
a fallback for a missing receipt.

## Frozen tier-0 cases

The compatibility smoke uses a temporary fixture directory and six real
`hermes chat -Q` turns:

- no-tool abstention;
- real `read_file` marker read;
- real `search_files` marker search;
- failed-read recovery with expected paths and order;
- side-effect abstention plus forbidden-artifact absence;
- visible-reasoning-marker hygiene.

The checks cover subprocess success, session-id discovery, successful SessionDB
loading, expected final text, required/forbidden tools, ordered tool receipts,
no-tool abstention, artifact absence, and visible output hygiene. Internal
reasoning fields in a receipt are diagnostic; markers in the final visible text
fail the case.

## Result policy

The summary reports only screening outcomes from the allowed set:

- `SCREEN-PASS`: all six compatibility cases and receipt gates passed;
- `REJECT`: a behavioral case failed after the receipt path was valid;
- `GATE-FAILED`: receipt integrity, timeout, missing session, or another
  non-negotiable execution gate failed;
- `HOLD`: reserved for an approved screening workflow that is incomplete or
  intentionally awaiting review.

`PROMOTE-CANDIDATE` is not a tier-0 status. Promotion-grade claims require a
later preregistered lane with at least 100 cases, paired candidate/incumbent
evidence, and human-only review.

## Standard workflow

1. Confirm the exact configured provider/model path and use synthetic fixtures.
2. Run only the compatibility command above; do not mutate config or
   credentials as part of the smoke.
3. Save receipts with `--out` when the result matters.
4. Read `summary.json` and `summary.md`, then inspect failed-case JSONL and raw
   stdout/stderr/session files.
5. Confirm every purported pass has a loaded SessionDB receipt and complete
   tool records, not merely printed text.
6. Report the exact provider, model, suite, toolset, status, failures, and
   receipt directory. Do not translate `SCREEN-PASS` into qualification,
   replacement, promotion, or routing.

Example:

```bash
OUT=/tmp/hermes-provider-validation-$(date +%Y%m%d-%H%M%S)
hermes providers validate \
  --provider custom:local-qwen \
  --model qwen3-coder \
  --toolsets file \
  --suite agent-readiness \
  --out "$OUT" \
  --timeout 120

sed -n '1,200p' "$OUT/summary.md"
```

For a local endpoint, configure it through normal Hermes provider setup first.
Keep credentials in the supported secret store. A direct curl check is useful
diagnosis but is not evidence for this smoke.

## Output files

The output directory contains:

- `summary.md`: human-readable status and receipt overview;
- `summary.json`: machine-readable status and per-case results;
- `results.jsonl`: one serialized result per case;
- `fixtures/`: synthetic files used by the smoke;
- `raw/<case>.stdout` and `raw/<case>.stderr`: complete captured streams;
- `raw/<case>.session.json`: loaded SessionDB messages when valid;
- `raw/<case>.session-error.txt`: missing-id or load-error evidence.

## Failure taxonomy

- **Provider unreachable:** subprocess failed before model behavior was
  evaluated.
- **Session receipt failure:** no session id, invalid session, or SessionDB load
  failure; printed output cannot repair this.
- **Tool receipt failure:** missing/forbidden tool, malformed arguments, absent
  result, or incorrect status/order/path.
- **Recovery failure:** the expected failed read and successful read did not
  occur in the required order and locations.
- **Side-effect boundary failure:** a forbidden tool was called or the forbidden
  artifact exists.
- **Visible reasoning leak:** the final visible text contains `<think>`,
  `<reasoning>`, or equivalent markers.
- **Timeout/incomplete:** preserve partial stdout/stderr and classify as a
  failed gate, not as a pass.
- **Harness failure:** command, installation, or fixture setup prevented the
  smoke from running.

## Verification checklist

Before reporting a result:

- [ ] `hermes providers validate --help` labels this as a tier-0 compatibility
      smoke and disclaims qualification/replacement evidence.
- [ ] Exact provider, model, `file` toolset, and suite are recorded.
- [ ] Status is one of `SCREEN-PASS`, `REJECT`, `GATE-FAILED`, or `HOLD`.
- [ ] Each passed case has a successfully loaded SessionDB receipt on disk.
- [ ] Tool receipts include ordered arguments, results, and status.
- [ ] Recovery paths/order and forbidden-artifact absence were checked.
- [ ] Timeout/invalid-session stdout, stderr, and partial evidence were retained.
- [ ] Failures are classified, not hand-waved.
- [ ] No secrets or private user data appear in prompts, paths, docs, or receipts.
- [ ] No qualification, replacement, promotion, leaderboard, or routing claim
      is made from this smoke.
