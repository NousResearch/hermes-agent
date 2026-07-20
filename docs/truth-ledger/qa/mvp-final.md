# Truth Ledger reduced-MVP final acceptance

Date: 2026-07-20
Branch: `feat/truth-ledger-option-2`
Worktree: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Verdict: **PASS**

## Acceptance evidence

### Live sanitized extraction measurement

Artifacts generated at `2026-07-20T22:16:56.560412Z`:

- Corpus: `docs/truth-ledger/evaluation/corpus.jsonl`
- Results: `docs/truth-ledger/evaluation/results.json`
- Report: `docs/truth-ledger/evaluation/report.md`

Observed results:

- 145/145 turns measured
- 145/145 terminal extractor calls observed on `openai-codex/gpt-5.6-sol`
- Provenance mismatches: 0
- Precision: 1.0000
- Recall: 1.0000
- No-fact abstention: 1.0000
- Overall accuracy: 1.0000
- Leakage rate: 0.0000
- Overall verdict: PASS

The evaluator conservatively rejects missing or mismatched provenance, unresolved retries, strict fact-field mismatches, and spurious additional admitted facts.

### Tests and diff hygiene

Command:

```bash
git diff --check && scripts/run_tests.sh \
  tests/plugins/truth_ledger/ \
  tests/plugins/test_truth_ledger_reconciliation.py \
  tests/plugins/test_truth_ledger_admission_redaction.py \
  tests/agent/test_turn_finalizer_post_llm_call_metadata.py \
  tests/agent/test_turn_finalizer_interrupt_alternation.py \
  tests/agent/test_turn_finalizer_final_response_persistence.py \
  tests/agent/test_turn_finalizer_cleanup_guard.py -q
```

Observed result: 16 files, 133 tests passed, 0 failed; `git diff --check` passed.

### Disposable canary

A fresh canary ran against a temporary `HERMES_HOME` and passed:

- source envelope schema valid
- synthetic secret absent after sanitization
- duplicate hook invocation suppressed
- subagent turn skipped
- only the temporary home was used

No live default-profile configuration or gateway was changed.

### Independent review

Independent read-only review `deleg_b2f45edb` returned **APPROVE**. It independently verified:

- per-turn provider/model provenance and conservative acceptance gating
- complete multi-fact precision accounting
- schema-valid reconciliation and manual-retraction events
- rejection of invalid ledger events
- payload rollback after pending-record failure
- strict matching, retry incompleteness handling, and complete leakage scanning

## Constraint interpretation

The user prohibition applies to Codex CLI, skill, and coding-agent lanes. None were used. The separately required extraction route was exercised through the Hermes runtime `PluginLlm` path using OpenAI-Codex OAuth and `gpt-5.6-sol`.

## Safety state

- Plugin remains opt-in and disabled by default.
- No live configuration was changed.
- No gateway restart occurred.
- No USER.md, MEMORY.md, Memory, or GBrain writes occurred.
- No merge, push, publish, or release occurred.
- Evaluation data is sanitized and synthetic.
