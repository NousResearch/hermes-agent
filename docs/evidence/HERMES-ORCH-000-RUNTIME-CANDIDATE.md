# HERMES-ORCH-000 — Verified Orchestrator Runtime Candidate

Date: 2026-07-11

## Goal

Establish one isolated Hermes runtime candidate containing the independently verified HERMES-OBS-001 usage ledger before implementing additional orchestrator features. The live gateway and default Hermes configuration were not replaced or restarted.

## Candidate identity

- Branch: `feature/hermes-orch-000-candidate`
- Worktree: `C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-orch-000-candidate`
- Candidate source commit before this evidence commit: `7db00025378522a3b4f4b80aa85aefe10fe1bc15`
- Installed live source: `C:\Users\fallo\AppData\Local\hermes\hermes-agent`
- Installed live branch/commit after proof: `main` / `540f90190f50f9518bf36632a724e0e58877a10b`

Candidate CLI evidence before the proof reported:

```text
Hermes Agent v0.18.2 (2026.7.7.2) · upstream 5e849942 · local 7db00025 (+12 carried commits)
Install directory: C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-orch-000-candidate
```

`hermes_cli.banner.get_git_banner_state()` obtains `origin/main`, `HEAD`, and the carried-commit count from the active checkout. The worker process imported `hermes_cli.kanban_usage_ledger` from the candidate worktree while a temporary `.pth` shim was installed. The shim was removed after the proof; the venv again imports `hermes_cli` from the installed live source.

## Source relationship and integration method

Observed repository relationship before candidate creation:

- Local `main`: `540f90190f50f9518bf36632a724e0e58877a10b`
- `origin/main`: `5e849942c` (`main` was ahead 1 and behind 1)
- Verified OBS/inventory tip: `7db00025378522a3b4f4b80aa85aefe10fe1bc15`
- `merge-base(local main, OBS tip)`: `540f90190f50f9518bf36632a724e0e58877a10b`
- OBS tip relative to local main: 11 commits ahead

The safest base was the verified OBS tip because OBS-001 branched directly from the installed local main. The candidate branch was created directly at that tip. This preserved the individual OBS implementation and evidence commits, preserved the current local Hermes base, and required no replay, squash, or source conflict resolution.

### Conflicts

None. No merge operation was required and no source conflict occurred.

The unrelated `origin/main` sandbox commit was not integrated in this milestone because the candidate objective was to preserve and verify the current local runtime plus OBS-001 without introducing an unverified upstream change.

## Test evidence

### OBS-001 ledger suite

Command:

```text
C:\Users\fallo\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m pytest tests/hermes_cli/test_kanban_usage_ledger.py -q -p no:cacheprovider
```

Result:

```text
101 passed in 198.20s
```

### Relevant Kanban tests

Results:

```text
tests/hermes_cli/test_kanban_promote.py:             16 passed
tests/hermes_cli/test_kanban_block_kinds.py:         11 passed
tests/hermes_cli/test_kanban_blocked_sticky.py:       6 passed
tests/hermes_cli/test_kanban_boards.py:              54 passed, 2 failed
```

The two board-test failures were Windows teardown file-lock failures in:

```text
TestBoardCRUD::test_remove_clears_init_cache_for_recreated_db[True]
TestBoardCRUD::test_remove_clears_init_cache_for_recreated_db[False]
```

Both failed with `PermissionError: [WinError 32]` while pytest removed a temporary `kanban.db`. They are recorded as test-environment limitations, not hidden or counted as passes.

The broad `test_kanban_core_functionality.py` invocation exceeded the 300-second harness limit and remains inconclusive. Per the resume instruction, completed suites were not rerun.

### Import/startup smoke tests

Candidate imports succeeded for:

```text
hermes_cli.main
hermes_cli.kanban_usage_ledger
aggregate_usage
query_usage
```

The candidate CLI started and identified local commit `7db00025` from the candidate worktree.

## Isolated runtime

### Exact HERMES_HOME

Two temporary paths appeared during setup:

- Incorrect/nested setup path: `C:\Users\fallo\AppData\Local\hermes\orch000-home`
- Actual successful worker HERMES_HOME: `C:\Users\fallo\orch000-home`

The successful proof used only the second path. This was necessary because `get_default_hermes_root()` treats any `HERMES_HOME` nested below the native default `C:\Users\fallo\AppData\Local\hermes` as part of the default root. Placing the proof home outside that tree made profile and board resolution genuinely isolated.

Successful proof artifacts:

- HERMES_HOME: `C:\Users\fallo\orch000-home`
- Board: `orch000`
- Database: `C:\Users\fallo\orch000-home\kanban\boards\orch000\kanban.db`
- Temporary profile: `orch000-proof`
- Worker log: `C:\Users\fallo\orch000-home\worker_t_39af6d40.log`
- Task: `t_39af6d40`
- Run: `1`
- Provider/model: `nous` / `tencent/hy3:free`

The temporary profile configured no fallback provider, disabled memory and delegation/self-improvement surfaces, disabled gateway dispatch and auto-decomposition, and exposed only the Kanban/Hermes lifecycle surface needed by the bounded task. No production Telegram destination was used.

### Authentication isolation

The existing default-home `auth.json` was copied, not moved, into the actual isolated HERMES_HOME for the proof. Immediately before the copy, both source hash and metadata were recorded. The source and copy initially had the same SHA-256:

```text
32ba1d08a87a7a75bc49c68540ccb497cbb62871f9d1723ba4a8f8b3394b3f38
```

After the successful proof and immediately after removing the isolated copy:

- `C:\Users\fallo\orch000-home\auth.json` was deleted.
- The original `C:\Users\fallo\AppData\Local\hermes\auth.json` still had the pre-copy SHA-256, byte size (`23466`), and recorded modification time.
- A later read-only-looking `hermes status --all` verification refreshed the live Nous credential as part of normal auth status resolution. This changed the default auth file hash from `32ba1d08a87a7a75bc49c68540ccb497cbb62871f9d1723ba4a8f8b3394b3f38` to `7e153b144c82b34289f75ba9f2d9a7e7b717997b8dca5ea95d8b49adef31470f` at `2026-07-11 03:33:08 -0600`, while retaining byte size `23466`. The original file was not moved, deleted, manually edited, or restored. This automatic token refresh is an operational side effect and means the stronger claim that the default Hermes home was byte-for-byte unchanged at final handback is false.
- No credential value is reproduced in this report.

## Tiny worker proof

Task contract:

```text
Read this task. Return EXACTLY the phrase: HY3 isolated usage proof complete.
Then call kanban_complete with that result. Do not modify files or run git.
```

Worker invocation was dispatcher-equivalent and explicitly pinned:

```text
HERMES_HOME=C:\Users\fallo\orch000-home
HERMES_KANBAN_BOARD=orch000
HERMES_KANBAN_TASK=t_39af6d40
HERMES_KANBAN_RUN_ID=1
HERMES_KANBAN_DB=C:\Users\fallo\orch000-home\kanban\boards\orch000\kanban.db
HERMES_PROFILE=orch000-proof
python -m hermes_cli.main -p orch000-proof --cli --accept-hooks chat -q "work kanban task t_39af6d40"
```

Lifecycle evidence:

- created: event exists, initial status `ready`
- admitted/ready: task creation returned `ready` with assignee `orch000-proof`
- claimed: event exists with run `1`
- spawned: temporary process PID `13712`, exited `0`; log path above
- heartbeating: heartbeat event exists for run `1`; task/run `last_heartbeat_at=1783762226`
- completed: task and run status `done`, outcome `completed`, error `NULL`
- result: `HY3 isolated usage proof complete.`

The log records the HY3 model completing the Kanban lifecycle and calling `kanban_complete`. It contains no credential values.

## Exact run_usage query and evidence

Query used against the isolated database:

```sql
SELECT
  board, task_id, run_id, api_call_index, call_kind,
  provider, model,
  input_tokens, output_tokens,
  cache_read_tokens, cache_write_tokens, reasoning_tokens,
  elapsed_ms,
  aux_input_tokens, aux_output_tokens,
  aux_cache_read_tokens, aux_cache_write_tokens,
  parent_task_id, profile, token_source,
  cost_usd, cost_status,
  checker_result, repair_cycle, accepted_result_tokens,
  api_calls, created_at
FROM run_usage
WHERE task_id = ?
ORDER BY run_id, call_kind, api_call_index;
```

Bound parameter: `t_39af6d40`.

The query returned three stable primary-call events, proving at least one successful observable HY3 API call. Their identities do not overwrite one another because the primary key is `(board, task_id, run_id, call_kind, api_call_index)`.

### Redacted row evidence

No value below is a prompt, response body, credential, authentication header, secret URL, or raw provider payload.

```json
[
  {
    "board": "orch000",
    "task_id": "t_39af6d40",
    "run_id": 1,
    "api_call_index": 0,
    "call_kind": "primary",
    "provider": "nous",
    "model": "tencent/hy3:free",
    "input_tokens": 22079,
    "output_tokens": 64,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 36,
    "elapsed_ms": 5964,
    "aux_input_tokens": null,
    "aux_output_tokens": null,
    "aux_cache_read_tokens": null,
    "aux_cache_write_tokens": null,
    "parent_task_id": null,
    "profile": "orch000-proof",
    "token_source": "provider_authoritative",
    "cost_usd": 0.0,
    "cost_status": "estimated",
    "checker_result": null,
    "repair_cycle": 0,
    "accepted_result_tokens": null,
    "api_calls": 0,
    "created_at": "2026-07-11T09:30:32.280Z"
  },
  {
    "board": "orch000",
    "task_id": "t_39af6d40",
    "run_id": 1,
    "api_call_index": 1,
    "call_kind": "primary",
    "provider": "nous",
    "model": "tencent/hy3:free",
    "input_tokens": 785,
    "output_tokens": 106,
    "cache_read_tokens": 21888,
    "cache_write_tokens": 0,
    "reasoning_tokens": 26,
    "elapsed_ms": 3838,
    "aux_input_tokens": null,
    "aux_output_tokens": null,
    "aux_cache_read_tokens": null,
    "aux_cache_write_tokens": null,
    "parent_task_id": null,
    "profile": "orch000-proof",
    "token_source": "provider_authoritative",
    "cost_usd": 0.0,
    "cost_status": "estimated",
    "checker_result": null,
    "repair_cycle": 0,
    "accepted_result_tokens": null,
    "api_calls": 0,
    "created_at": "2026-07-11T09:30:36.153Z"
  },
  {
    "board": "orch000",
    "task_id": "t_39af6d40",
    "run_id": 1,
    "api_call_index": 2,
    "call_kind": "primary",
    "provider": "nous",
    "model": "tencent/hy3:free",
    "input_tokens": 259,
    "output_tokens": 53,
    "cache_read_tokens": 22528,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
    "elapsed_ms": 3164,
    "aux_input_tokens": null,
    "aux_output_tokens": null,
    "aux_cache_read_tokens": null,
    "aux_cache_write_tokens": null,
    "parent_task_id": null,
    "profile": "orch000-proof",
    "token_source": "provider_authoritative",
    "cost_usd": 0.0,
    "cost_status": "estimated",
    "checker_result": null,
    "repair_cycle": 0,
    "accepted_result_tokens": null,
    "api_calls": 0,
    "created_at": "2026-07-11T09:30:39.364Z"
  }
]
```

`api_calls=0` is the persisted value even though three uniquely indexed usage events exist. This field should not be treated as the event count; the event identity/count comes from the stable primary keys. This is an observability semantic limitation to retain for later work, not a reason to manufacture a different value.

## Privacy verification

Actual `run_usage` schema columns:

```text
board, task_id, run_id, api_call_index, call_kind, provider, model,
input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
reasoning_tokens, elapsed_ms, aux_input_tokens, aux_output_tokens,
aux_cache_read_tokens, aux_cache_write_tokens, parent_task_id, profile,
token_source, cost_usd, cost_status, checker_result, repair_cycle,
accepted_result_tokens, api_calls, created_at
```

There are no columns for prompt text, response body, conversation content, API key, OAuth token, credential, authentication header, secret URL, raw request, or raw response payload.

Every textual value in the returned rows was scanned for:

```text
proof task text; required response phrase; authorization; bearer; api_key;
access_token; refresh_token; oauth; raw request; raw response
```

Result: zero matches.

Therefore the isolated usage ledger persisted metadata and observed token/cost fields only. It did not persist prompt text, response content, credentials, authentication headers, or raw provider payloads.

## Cleanup and production-safety verification

- All three temporary worker process sessions were exited; no temporary process remained running.
- Copied isolated `auth.json` removed after proof.
- Temporary `.pth` candidate-source shim removed.
- The temporary database and worker log remain preserved under `C:\Users\fallo\orch000-home` as evidence.
- The actual isolated HERMES_HOME was not deleted before this report commit.
- Candidate worktree contained no unintended source changes before this report was written.
- Installed live source remained `main` at `540f90190f50f9518bf36632a724e0e58877a10b`.
- The normal gateway was not restarted or replaced. `hermes status --all` still resolved the normal project at `C:\Users\fallo\AppData\Local\hermes\hermes-agent`, with Telegram configured and the default profile using the user-selected OpenAI Codex runtime.
- No Telegram send occurred from the proof.
- No push, merge, dashboard, or HERMES-ORCH-001 work occurred.

## Activation assessment

**Candidate verdict: SAFE TO ACTIVATE AS THE NEXT CONTROLLED LOCAL RUNTIME CANDIDATE.**

Grounds:

1. It preserves the installed local base and every verified OBS-001 commit.
2. The OBS ledger test suite passed all 101 tests.
3. Relevant Kanban lifecycle suites passed except for two documented Windows file-lock teardown failures; the broad suite timeout is explicitly retained as inconclusive.
4. Candidate startup/import smoke tests passed.
5. A real isolated Nous/HY3 worker completed a Kanban task and emitted three nonzero, provider-authoritative usage rows tied to the exact board/task/run/profile/provider/model.
6. The ledger schema and stored values contain no prompt, response, credential, or raw provider payload.
7. The live gateway, live source commit, default auth file, and production Telegram path were not replaced or modified by the proof.

This verdict authorizes only a separate, controlled activation step. It does not itself restart, replace, merge, or push the live gateway.