---
name: traceguard
description: Gate synthesis claims on accepted evidence handles.
version: 1.1.0
author: Sigrid Jin (@sigridjineth), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Evidence, Verification, Synthesis, RLM, Traceability]
    category: research
    related_skills: []
    requires_toolsets: [terminal]
---

# TraceGuard Skill

Deterministically validates a structured parent synthesis against a bounded
manifest of accepted child evidence handles: every claimed fact must cite a
supported `fact_id` plus the matching `chunk_id`/`evidence_chunk_id`. It is
not an LLM judge, it never blocks task completion, and it is not part of the
core Hermes toolset.

## When to Use

- An RLM-style research or synthesis flow emits structured claims with
  `fact_id`/`chunk_id` handles and you need a deterministic audit that each
  claim traces back to accepted child evidence.
- You want a machine-checkable unsupported-claim rate for a synthesis
  artifact before shipping it downstream.

Do not use it on free-form prose — it only inspects structured claim fields.

## Prerequisites

Python 3.11+ standard library only. No pip installs, no network.

Helper script path: `scripts/traceguard.py` relative to this skill directory
(installed skills: `~/.hermes/skills/research/traceguard/scripts/traceguard.py`).

## How to Run

1. Use `write_file` to save a payload JSON with the two required keys:

   ```json
   {
     "evidence_manifest": [
       {
         "fact_id": "TG-001",
         "chunk_id": "notes.txt:1-2",
         "text": "FACT:TG-001 retained child evidence.",
         "child_call_id": "child_0001"
       }
     ],
     "parent_synthesis": {
       "result": {
         "retained_facts": [
           {
             "fact_id": "TG-001",
             "evidence_chunk_id": "notes.txt:1-2",
             "text": "retained child evidence"
           }
         ]
       }
     }
   }
   ```

2. Use `terminal` to run the validator:

   ```
   python3 scripts/traceguard.py --input payload.json
   ```

   (Pipe the payload on stdin instead by omitting `--input`.)

3. Read the JSON verdict from the `terminal` output. Exit code 0 means every
   claim is accepted, 1 means at least one claim was rejected, 2 means the
   payload was malformed.

## Quick Reference

Claim-bearing surfaces scanned: `result.retained_facts`,
`result.observed_facts`, `result.facts`, `result.retained_evidence`,
`result.observed_evidence`, and `evidence_references`.

Rejection reasons:

| Reason | Meaning |
| --- | --- |
| `chunk_handle_without_fact` | A chunk handle was cited with no `fact_id`. |
| `unsupported_fact_id` | The `fact_id` is not in the manifest. |
| `missing_evidence_handle` | The claim has no chunk/evidence handle. |
| `evidence_handle_mismatch` | The handle differs from the manifest entry. |

Verdict fields: `traceguard.accepted`, `traceguard.unsupported_claim_rate`,
`traceguard.accepted_claims`, `traceguard.rejected_claims`, plus a
`normalized_evidence_manifest` for deterministic trace artifacts.

## Pitfalls

- Small local models mangle citation schemas under load; treat rejections as
  "claim is untraceable", not "model is lying", and fix the synthesis format
  before re-running.
- Manifest entries missing either a `fact_id` or a chunk handle are dropped
  during coercion, so claims citing them will reject as `unsupported_fact_id`.
- The validator is advisory. Do not wire it into completion gating; run it
  explicitly where a workflow needs the audit.

## Verification

Regression tests: `scripts/run_tests.sh tests/skills/test_traceguard_skill.py -q`.
For a quick manual check, run the payload above — it prints
`"accepted": true` and exits 0; change the claim's `evidence_chunk_id` to a
different handle and it prints `evidence_handle_mismatch` and exits 1.
