# Plan: Cron delivery retry for transient failures (GitHub #13566)

**Date**: 2026-04-22  
**Role**: Plan Master (atomic breakdown)  
**Constitution**: `.cursorrules` — ledger sniff completed; DLP: repo-relative paths only, English in committed artifacts.

---

## Phase 1 — Digest

### Ammo-clip ledger outcome (file-level)

| Planned direction | Ledger collision (same core file = block) |
|-------------------|-------------------------------------------|
| Independent MCP Web3 + core wiring | **Block**: `#029` `tools/mcp_tool.py` / `src/hermes_mcp/`; `#036` `optional-skills/mcp/web3-chain-tools/`; `#039` `agent/web3_mcp_governance.py`, `run_agent.py`, `hermes_cli/config.py` |
| Vector / hybrid memory providers | **Block**: `#040` `agent/vector_hybrid/`, `plugins/memory/vector_hybrid/`, `hermes_cli/config.py`; `#022` `agent/memory_manager.py` |
| Sandbox execution providers | **Block**: `#038` `src/sandbox/`, `tools/code_execution_tool.py`, `tools/terminal_tool.py`, `hermes_cli/config.py` |

All three clip targets **degraded**; no implementation started on those paths in this workstream.

### Fallback prey (Alpha: `type/feature`)

- **Source**: GitHub REST API — `NousResearch/hermes-agent` open issues; label / title filter → **#13566** — *Cron delivery retry mechanism for transient network failures*.
- **Rationale**: `Master_Ledger.md` index has **no** rows claiming `cron/scheduler.py` or `cron/jobs.py` for parallel PRs; avoids hot files (`hermes_cli/config.py`, `tools/mcp_tool.py`, `run_agent.py`) if retries are driven by **per-job JSON fields** and/or **documented env vars** read inside `cron/` only.

### Scope guard

- **In scope**: Retry loop + backoff around `_deliver_result` (or a thin helper used from `_process_job` in `tick()`), optional job fields, structured logging, unit tests.
- **Out of scope (v1)**: Changing global `config.yaml` schema in `hermes_cli/config.py` (high merge collision); full distributed queue — keep single-process semantics.

### Ledger lock (manual)

Repo parent ledger must be updated by the account holder (agent workspace policy: edits under this repo only). Append a **WIP** line naming:

- `cron/scheduler.py`
- `tests/cron/test_scheduler.py` (and any new `tests/cron/test_*delivery*retry*.py` if split)

---

## Phase 2 — Atomic breakdown

### [Step 1] Define retry policy surface (job JSON + env fallback)

- **File**: `cron/jobs.py`
- **Action**: Modify (small)
- **Details**: Document and optionally validate optional keys on each job dict, e.g. `delivery_retry_max` (int, default from env), `delivery_retry_base_seconds` (float, default 5). Add module-level helpers `_delivery_retry_defaults()` reading `HERMES_CRON_DELIVERY_MAX_RETRIES` / `HERMES_CRON_DELIVERY_RETRY_BASE_SECONDS` with safe caps (document in code comments; no new website doc required for v1 unless requested).
- **Verification**: `python -m pytest tests/cron/test_jobs.py -q -o addopts=` still passes; add one test if helpers are exported.

### [Step 2] Classify delivery errors as retryable vs terminal

- **File**: `cron/scheduler.py`
- **Action**: Modify
- **Details**: Add `_is_transient_delivery_failure(msg: str) -> bool` (or similar): treat timeout strings, `ConnectionError`, common `aiohttp`/`httpx` network phrases as retryable; treat `unknown platform`, `not configured/enabled`, `no delivery target` as **non-retryable**. Keep heuristic conservative (when unsure, retry at most once or not at all — pick one policy and document).
- **Verification**: New unit tests assert classification for representative strings; no import cycles.

### [Step 3] Implement bounded retry with backoff around delivery

- **File**: `cron/scheduler.py`
- **Action**: Modify
- **Details**: From `_process_job` (or a new `_deliver_with_retries(job, content, adapters, loop)`), call existing `_deliver_result` in a loop: sleep `base * 2**attempt` between attempts (cap max sleep, e.g. 60s); stop on success, terminal error, or max attempts. Preserve existing `last_delivery_error` semantics: set to **last** error if all attempts fail; `None` if any attempt succeeds.
- **Verification**: `python -m pytest tests/cron/test_scheduler.py -q -o addopts=` passes; new tests mock `_deliver_result` to fail N times then succeed / fail permanently.

### [Step 4] Regression: SILENT / adapter / wrap paths unchanged

- **File**: `tests/cron/test_scheduler.py`
- **Action**: Modify
- **Details**: Extend or add tests proving retry wrapper does not change SILENT suppression, media extraction, or adapter-first path (mock at `_deliver_result` boundary).
- **Verification**: Full cron test slice: `python -m pytest tests/cron/ -q -o addopts=`.

### [Step 5] Optional split — dedicated test module

- **File**: `tests/cron/test_cron_delivery_retry.py` (optional if `test_scheduler.py` grows too large)
- **Action**: Create
- **Details**: Host only retry/backoff/classification tests if Step 3 adds >~80 lines of tests to `test_scheduler.py`.
- **Verification**: Same as Step 4.

---

## Phase 3 — Delivery checklist

- [x] No edits to `hermes_cli/config.py` in v1 (unless ledger clears and product asks for YAML keys).
- [ ] Commit message and PR body: English only; link `https://github.com/NousResearch/hermes-agent/issues/13566`.
- [ ] After push: holder updates `Master_Ledger.md` index + detail block per team protocol.

### Implementation status (2026-04-22)

- `cron/jobs.py`: `delivery_retry_max_extra`, `delivery_retry_base_seconds` + env `HERMES_CRON_DELIVERY_MAX_RETRIES`, `HERMES_CRON_DELIVERY_RETRY_BASE_SECONDS`; optional per-job `delivery_retry_max`, `delivery_retry_base_seconds`.
- `cron/scheduler.py`: `_is_transient_delivery_failure`, `_deliver_result_with_retries`; `tick()` → `_process_job` uses retries around delivery.
- Tests: `tests/cron/test_jobs.py` (`TestDeliveryRetryPolicy`), `tests/cron/test_scheduler.py` (`TestDeliveryRetryTransientHeuristic`, `TestDeliverResultWithRetries`).
- Verification: `python -m pytest tests/cron/test_jobs.py tests/cron/test_scheduler.py -q -o addopts=` → green. Full `tests/cron/` may show unrelated failures on Windows for `test_file_permissions.py` (chmod / `st_mode`).

---

💡 **Plan Master 提示**: The above is a file-level atomic breakdown. If this matches your intent, reply **「按计划执行」** and implementation can proceed step-by-step with `pytest tests/cron/` as the primary gate.
