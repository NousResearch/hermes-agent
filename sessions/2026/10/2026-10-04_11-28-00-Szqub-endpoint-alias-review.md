---
session_id: 2026-10-04-endpoint-credential-alias-review
writer: Szqub
role: implementer
started_at: 2026-10-04T11:28:00+02:00
timezone: Europe/Warsaw
status: completed
updated_at: 2026-10-04T11:40:28.6181506+02:00
ended_at: 2026-10-04T11:40:28.6181506+02:00
repo: Szqub/hermes-agent
target: NousResearch/hermes-agent#132598
branch: review/agent-provider-config
base_sha: e76ee5f9c8a503f6ec54df32db0ef450dc7e725d
coordination_mode: github-target-claim
claim_id: https://github.com/NousResearch/hermes-agent/pull/132598#issuecomment-5978513445
claim_status: done
claim_released_at: 2026-10-04T11:40:28.6181506+02:00
release_evidence: https://github.com/NousResearch/hermes-agent/pull/132598#issuecomment-5978599857
rules:
  global_ref: ByteTech-PL/agents-global-hub/.rulesync/rules/AGENTS.md
  global_revision: 9aab2d7f3a556eefa93fc0f18be51e6213d32fd1
  project_ref: AGENTS.md
  project_revision: e76ee5f9c8a503f6ec54df32db0ef450dc7e725d
touched_paths:
  - hermes_cli/web_routers/config_env.py
  - tests/hermes_cli/test_web_server.py
---

Full review read before editing. Detach/delete and display omit api_key_env.
Preserve/clear coverage will verify real edits, raw persistence, normalization,
and runtime behavior. Existing main checkout was clean and equal to origin/main.
Task-specific attribution instructions restrict this record to operator identity.

## Verified findings and implementation

- Confirmed stale model.api_key_env after endpoint deletion. The review overstates
  immediate auth reuse: _model_level_key_env also checks model.provider, which
  deletion clears. The regression verifies raw removal and post-delete runtime.
- Confirmed false has_api_key and missing display for alias-only provider entries.
- Confirmed alias-only pointers were not copied by activate, make-default, or
  main-model assignment. Each corresponding regression failed before its fix.
- Completed these pre-existing alias semantics with five local fallback/cleanup
  changes; no auth refactor, new representation, or process-environment shortcut.
- Expanded scope: hermes_cli/web_server_config.py and
  tests/hermes_cli/test_model_assignment_env_key_mirror.py.
- Omitted-key tests now assert a persisted real edit, raw reference retention,
  normalized reference, runtime resolution, and environment preservation.
  Clear tests cover all three credential fields and shared versus dedicated slots.
- Normalization canonicalizes a copied runtime view; raw persistence is checked
  separately so normalization cannot hide stale references.

## Validation (Windows, Python 3.14.7, official scripts/run_tests.sh)

- Reviewed-head delete/display matrix: 10 passed, 2 failed as expected.
- Reviewed-head pointer-copy matrix: 3 passed, 3 failed as expected.
- Original clear behavior restored temporarily: 4 passed, 4 failed as expected.
- Final focused regression matrix: 18 passed.
- Full affected 12-file set: 354 passed, 5 skipped. Full test_web_server.py:
  206 passed, 5 skipped (POSIX timezone and PTY functionality unavailable on Windows).
- Additional 7-file set: 273 passed, 2 failed. Failures are unchanged
  test_config.py::TestEnvWriteDenylist::test_non_exec_near_misses_still_writable
  cases git_config_parameters and ld_preload; Windows env names are case-insensitive.
  Repeated with reviewed-head production files: 4 passed, same 2 failures.
- ruff check . passed; git diff --check passed. An initial lint invocation
  traversed the disposable PM runtime's standard library (44 irrelevant errors);
  excluding only that untracked local tool store corrected the invocation.
- Test interpreter built through pm.build_env with dev and test groups.
  No full repository suite or Linux run claimed.

## Documentation review

README and project/area rules reviewed. This upstream checkout has no STATUS.md,
MEMORY.md, PLAN.md, or prior sessions; avoid unrelated lifecycle-file migration
in this narrow PR. This record and the PR carry current status and validation.
The canonical alias contract already exists in config_providers.py; the display
helper documentation now names both spellings. No new future work was introduced.
durable_memory_promoted: false — existing documented contract completed, no new
configuration concept; regression tests preserve the verified lifecycle behavior.

## Coordination

Existing origin/main remained bd0affe5e5f723579df8902852f5d0c47795f355 and the
remote PR branch remained e76ee5f9c8a503f6ec54df32db0ef450dc7e725d after fresh fetch.
Only this worktree owns implementation; the original checkout remains unchanged.

## Closure

Implementation checkpoint: 8f260487173d22184cb2928adfd365cba26af8d3, pushed to
review/agent-provider-config. Existing PR description updated and factual review
reply published. Claim explicitly released by the linked RUN_END / RELEASE.
Final documentation closure follows that release; no technical changes remain.
Known unrelated Windows test failures remain disclosed, not concealed.

