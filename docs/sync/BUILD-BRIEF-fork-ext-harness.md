# BUILD BRIEF: refactor_equiv harness + rank-5 shakedown extraction (fork-mergeability spec, Deliverable 0 + first extraction)

You are in a git worktree on branch `feat/fork-ext-harness` (off fork/main). Implement
**Deliverable 0 + the rank-5 shakedown extraction ONLY** from
`docs/sync/2026-07-16-fork-mergeability-refactor-SPEC.md` (v0.3, Momus AWC-folded). Read the
spec FULLY first. Ranks 1-4 and 6 are LATER PRs — do not touch them.

## Part A — the harness: `scripts/refactor_equiv/`
Build the golden-transcript differential runner as a small package:
- `runner.py`: capture / verify modes per the spec (corpus in, serialized observable outputs
  out, golden JSON keyed by input hash; verify = byte-identical or fail loud w/ exact field).
- `determinism.py`: the seam kit — frozen wall clock AND time.monotonic (monkeypatch), seeded
  uuid4/token counter, fixed temp HERMES_HOME + fixed seed DB, sqlite3.connect wrapper
  registering deterministic datetime/strftime overrides for SQL-layer clocks.
  Pre-flight fn: grep target module + schema for monotonic|CURRENT_TIMESTAMP|datetime('now')|
  AUTOINCREMENT and error if any hit lacks a named seam.
- `equiv_normalize.py`: the reviewed allowlist (starts near-empty). Self-lint: any allowlist
  entry matching /time|date|_at$|_ts$|id$/ fails the harness itself.
- `mutate.py`: the mutation self-test CLI (`mutate --module <m>`): applies each registered
  mutation (min 3, one per output class: return-value / message-emit / DB-write), replays,
  asserts RED, reverts; exit nonzero on any missed mutation.
- Branch-coverage gate: run the corpus under `coverage run --branch` scoped to the extracted
  module; fail under 100%.

## Part B — rank-5 shakedown: extract relay lane headers
Target: the fork-only relay-pool header block in `agent/chat_completion_helpers.py`
(~lines 980-1100: `_pool_lane`, `_pool_lane_src`, the lane classifier, and the function
building `x-hermes-session`/`x-hermes-lane`/`x-hermes-lane-src` headers).
Follow the spec's per-extraction ritual IN ORDER:
1. Golden-capture on the UNTOUCHED tree first: corpus = realistic (agent, aux_task) input
   matrix covering every branch of the classifier (interactive/background/headless/cron
   principals, aux tasks, missing session ids). Real entrypoint = call the actual header-builder
   fn, capture returned header dicts.
2. Extract to `agent/fork_ext/relay_headers.py` via git mv-style move (NO logic edits);
   1-line call sites in chat_completion_helpers.py.
3. Golden-replay → byte-identical.
4. Mutation self-test → all 3 classes RED, then revert mutations.
5. Run the relay/lane test suites: tests/agent/ greps for x-hermes-lane tests + neighbors.
6. fork-features.json: ADD a new entry for the relay-headers feature with paths
   ["agent/fork_ext/relay_headers.py"] and its guard tests.
7. NET-addition check: budget = 10 + 2×call_sites (measure call sites by grep).
8. Import-order audit per the spec.

## Test requirements
- tests/scripts/test_refactor_equiv.py: harness unit tests — determinism seams actually freeze
  (two capture runs byte-identical), allowlist self-lint RED on a time-named field, mutation
  harness exits nonzero when a mutation is NOT detected (feed it a serializer that ignores one
  output class), 100%-branch gate RED on an uncovered branch.
- The rank-5 extraction's golden + corpus + mutations COMMITTED under
  tests/golden/relay_headers/ per the spec's RC-B (mechanical enforcement).

## Constraints
- Python 3.11, stdlib + existing venv deps only (coverage is available; freezegun may not be —
  check, else monkeypatch by hand).
- Suite: HOME=/Users/alexgierczyk ~/.hermes/hermes-agent/venv/bin/python -m pytest
  tests/scripts/test_refactor_equiv.py <relay tests> -q -o addopts="" -p no:randomly → GREEN.
- py_compile everything you touch. Commit per-part locally. DO NOT push, DO NOT open a PR.
  STOP when green + committed; the orchestrator verifies and lands.
