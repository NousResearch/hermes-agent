# Hermes GPT-5.5 Prompt Baseline Refactor

Status: review
Started: 2026-04-28
Repo: /Users/jeffphoon/.hermes/hermes-agent
Branch: feat/restore-hermes-memory-provider-after-update

## Goal
Refactor Hermes prompt guidance toward a GPT-5.5-friendly baseline: outcome-first, modular, explicit success/verification/stop rules, fewer broad absolute rules, while preserving hard invariants and tool-use discipline.

## Source Context
- User requested research on The Decoder article about OpenAI GPT-5.5 prompt guidance.
- OpenAI guide emphasizes fresh baseline, short outcome-first prompts, role/personality/collaboration structure, retrieval budgets, preambles, validation, and stop rules.

## Acceptance Criteria
- Prompt builder includes explicit Role, Personality, Collaboration Style, Success Criteria, Constraints, Tool Use, Retrieval Budget, Verification, and Stop Rules sections.
- Hard invariants remain strict: secrets, production/destructive authorization, mandatory tool-use for live/system/file/git/math/current facts.
- Broad judgment rules are expressed as decision rules where possible.
- Existing tests are updated or extended to guard required prompt blocks.
- Run targeted prompt tests and prompt caching tests.

## Plan
1. Inspect current prompt builder and tests.
2. Add or update tests for GPT-5.5 baseline sections.
3. Refactor guidance constants in `agent/prompt_builder.py`.
4. Run targeted tests.
5. Commit changes.

## Notes
- This repo is not under `/Users/jeffphoon/Documents/GitHub/<repo>`, but we still keep a local `.hermes/` ledger for traceability.
- Do not store secrets in this ledger.

## Implementation Summary
- Refactored OpenAI/GPT execution guidance into modular GPT-5.5-style blocks:
  - `BASE_ROLE_GUIDANCE`
  - `PERSONALITY_AND_COLLABORATION_GUIDANCE`
  - `SUCCESS_CRITERIA_GUIDANCE`
  - `CONSTRAINTS_GUIDANCE`
  - `TOOL_USE_GUIDANCE`
  - `RETRIEVAL_BUDGET_GUIDANCE`
  - `EXECUTION_DISCIPLINE_GUIDANCE`
  - `VERIFICATION_AND_STOP_RULES_GUIDANCE`
- Kept strict hard invariants for mandatory tool-use cases, secrets, and production/destructive authorization.
- Softened broad tool-use enforcement wording into decision-rule style while preserving immediate tool-backed action requirements.
- Added prompt-builder regression tests for baseline sections, composition, retrieval budget, preamble/collaboration style, hard invariants, and stop rules.

## Verification
```bash
cd /Users/jeffphoon/.hermes/hermes-agent
python -m pytest tests/agent/test_prompt_builder.py tests/agent/test_prompt_caching.py -q
# 136 passed, 1 skipped
```

## Status
Ready for review/commit.
