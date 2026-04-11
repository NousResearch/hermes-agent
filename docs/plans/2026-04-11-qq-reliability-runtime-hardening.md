# QQ Reliability And Runtime Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make QQ group/private interactions reliable under load by hardening message intake, busy/queued visibility, empty-response fallbacks, ordered provider fallback chains, provider-specific vision payloads, and deploy/restart verification.

**Architecture:** Keep a single QQ-facing bot flow, but make every stage explicit and observable: intake trigger decisions in the QQ adapter, response state and busy status in the gateway runner, ordered runtime failover in `AIAgent`, provider-family-specific vision payload shaping in `tools/vision_tools.py`, and a repeatable deploy/health-check script. Avoid broad refactors; prefer surgical compatibility layers and regression tests around current behavior.

**Tech Stack:** Python, asyncio, pytest, Hermes gateway/session/runtime config, QQ NapCat adapter, auxiliary client router, shell deploy helper.

---

### Task 1: Harden QQ trigger and batching decisions

**Files:**
- Modify: `gateway/platforms/qq_napcat.py`
- Test: `tests/gateway/test_qq_napcat.py`

**Step 1: Add failing tests**

Cover:
- alias mentions `马嘎/马噶/马哥/...` trigger group dispatch consistently
- follow-up windows remain group-shared in project mode
- low-signal batches are skipped with a concrete reason
- explicit progress-check messages dispatch even without raw `@` mention

**Step 2: Run target tests**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_qq_napcat.py -q
```

**Step 3: Implement minimal adapter changes**

Add:
- normalized alias detection helper
- structured dispatch/skip reason helper
- stronger “explicit progress check” trigger path
- observability fields that the gateway can reuse

**Step 4: Re-run tests**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_qq_napcat.py -q
```

**Step 5: Commit**

```bash
git add gateway/platforms/qq_napcat.py tests/gateway/test_qq_napcat.py
git commit -m "feat: harden qq trigger and batching decisions"
```

### Task 2: Make busy, queued, empty, and silent turns visible in the gateway

**Files:**
- Modify: `gateway/run.py`
- Test: `tests/gateway/test_no_reply_marker.py`
- Test: `tests/gateway/test_busy_input_mode.py`
- Test: `tests/gateway/test_command_bypass_active_session.py`
- Test: `tests/gateway/test_queue_consumption.py`

**Step 1: Add failing tests**

Cover:
- explicit QQ mentions that end as `(empty)` or `[[NO_REPLY]]` send a fallback line
- queued/busy turns return a short visible acknowledgement instead of silent `None`
- normal low-signal group `[[NO_REPLY]]` remains silent
- model identity / status queries still return the live reply

**Step 2: Run target tests**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tests/gateway/test_no_reply_marker.py \
  tests/gateway/test_busy_input_mode.py \
  tests/gateway/test_command_bypass_active_session.py \
  tests/gateway/test_queue_consumption.py -q
```

**Step 3: Implement gateway changes**

Add:
- QQ explicit-address fallback for both `(empty)` and `[[NO_REPLY]]`
- short busy/queued acknowledgement when a live session is already running
- structured response state logging with reason labels

**Step 4: Re-run tests**

Run the same pytest command and ensure all pass.

**Step 5: Commit**

```bash
git add gateway/run.py tests/gateway/test_no_reply_marker.py \
  tests/gateway/test_busy_input_mode.py tests/gateway/test_command_bypass_active_session.py \
  tests/gateway/test_queue_consumption.py
git commit -m "feat: surface qq busy and silent-turn states"
```

### Task 3: Upgrade provider fallback chain behavior

**Files:**
- Modify: `run_agent.py`
- Modify: `agent/auxiliary_client.py`
- Test: `tests/run_agent/test_fallback_model.py`
- Test: `tests/run_agent/test_primary_runtime_restore.py`
- Test: `tests/run_agent/test_run_agent.py`

**Step 1: Add failing tests**

Cover:
- list-style fallback chains skip duplicate/failed endpoints cleanly
- 401/403/429/empty-response fallback routing records useful status
- fallback entries can carry `context_length`, `api_mode`, and `name`
- fresh turns restore the primary runtime and reset chain state

**Step 2: Run target tests**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tests/run_agent/test_fallback_model.py \
  tests/run_agent/test_primary_runtime_restore.py \
  tests/run_agent/test_run_agent.py -q
```

**Step 3: Implement runtime changes**

Add:
- fallback entry normalization helper
- endpoint dedupe / skip logic for same failed target
- clearer emitted statuses for auth/rate-limit/empty-response transitions
- preserve compressor/runtime context length when swapping providers

**Step 4: Re-run tests**

Run the same pytest command and ensure all pass.

**Step 5: Commit**

```bash
git add run_agent.py agent/auxiliary_client.py \
  tests/run_agent/test_fallback_model.py tests/run_agent/test_primary_runtime_restore.py \
  tests/run_agent/test_run_agent.py
git commit -m "feat: harden ordered provider fallback chain"
```

### Task 4: Finish provider-family-specific vision handling

**Files:**
- Modify: `tools/vision_tools.py`
- Modify: `agent/auxiliary_client.py`
- Test: `tests/tools/test_vision_tools.py`

**Step 1: Add failing tests**

Cover:
- OpenAI-compatible `/v1/chat/completions` payload family
- Anthropic `/anthropic` payload family
- Gemini-compatible OpenAI image payload family
- local file inputs and downloaded files behave identically
- auxiliary fallback kicks in when the first vision backend rejects vision

**Step 2: Run target tests**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/tools/test_vision_tools.py -q
```

**Step 3: Implement minimal vision changes**

Add:
- explicit payload-family routing helper
- retry/fallback metadata in debug logs
- provider-family-aware detail normalization

**Step 4: Re-run tests**

Run the same pytest command and ensure all pass.

**Step 5: Commit**

```bash
git add tools/vision_tools.py agent/auxiliary_client.py tests/tools/test_vision_tools.py
git commit -m "feat: harden provider-specific vision routing"
```

### Task 5: Add deploy and health-check helper

**Files:**
- Create: `scripts/deploy_gateway.sh`
- Optional Doc: `website/docs/developer-guide/provider-runtime.md`

**Step 1: Add helper script**

Script responsibilities:
- print target host and working tree
- optionally rsync/sync selected files
- back up config/runtime files
- restart gateway with `--replace`
- verify `gateway_state.json`
- tail gateway log

**Step 2: Smoke-test locally**

Run:
```bash
bash scripts/hermes-gateway/deploy_gateway.sh --help
```

**Step 3: Update existing wrapper if needed**

Keep the existing `scripts/hermes-gateway` service manager unchanged; the new helper is for targeted sync + restart + health verification.

**Step 4: Commit**

```bash
git add scripts/deploy_gateway.sh
git commit -m "chore: add gateway deploy and health-check helper"
```

### Task 6: Verify the integrated path

**Files:**
- No new code required; verification only

**Step 1: Run focused regression suite**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tests/gateway/test_no_reply_marker.py \
  tests/gateway/test_busy_input_mode.py \
  tests/gateway/test_command_bypass_active_session.py \
  tests/gateway/test_queue_consumption.py \
  tests/gateway/test_qq_napcat.py \
  tests/run_agent/test_fallback_model.py \
  tests/run_agent/test_primary_runtime_restore.py \
  tests/run_agent/test_run_agent.py \
  tests/tools/test_vision_tools.py -q
```

**Step 2: Capture deployment notes**

Record:
- files changed
- config assumptions
- remote restart command
- smoke-test prompts (`@马嘎 在?`, image upload, fallback chain failure simulation)

**Step 3: Final commit**

```bash
git add docs/plans/2026-04-11-qq-reliability-runtime-hardening.md
git commit -m "docs: add qq reliability hardening plan"
```
