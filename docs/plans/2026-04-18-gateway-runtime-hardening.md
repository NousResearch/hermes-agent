# Gateway Runtime Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce gateway runtime drift by hardening direct-control parsing, removing code-level auto-background keyword coupling, and unifying the foreground vision preparation path around the actual production flow.

**Architecture:** Keep the existing gateway execution order, but move brittle keyword lists out of Python code and add a thin normalization layer for admin oral-control text before matching. For vision, treat `gateway/run.py` auto-vision as the source of truth and either route the shared helper through that logic or remove the dead divergent path so tests cover the real runtime behavior.

**Tech Stack:** Python, pytest, gateway runtime services, JSON-backed lexicon/config data.

---

### Task 1: Data-Drive Auto-Background Lexicon

**Files:**
- Create: `gateway/data/auto_background_intents.json`
- Modify: `gateway/auto_background_runtime_service.py`
- Test: `tests/gateway/test_auto_background_runtime_service.py`

**Step 1: Write the failing tests**

Add coverage for:
- loading shortcuts/action/domain terms from data instead of module constants
- safe fallback when the data file is missing or malformed
- preserving existing routing behavior for bare `继续` and obvious work requests

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_auto_background_runtime_service.py -q`
Expected: FAIL once tests assert JSON-backed loading behavior that does not exist yet.

**Step 3: Write minimal implementation**

Implement a JSON loader in `gateway/auto_background_runtime_service.py` similar to `gateway/qq_intents.py`, replacing code-level hardcoded term tuples with loaded data while keeping the public helper API unchanged.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_auto_background_runtime_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/data/auto_background_intents.json gateway/auto_background_runtime_service.py tests/gateway/test_auto_background_runtime_service.py
git commit -m "Data-drive auto background intents"
```

### Task 2: Normalize Admin Direct-Control Text Before Matching

**Files:**
- Modify: `gateway/direct_control_event_runtime_service.py`
- Modify: `gateway/group_control_requests.py`
- Modify: `gateway/qq_intel_control_requests.py`
- Test: `tests/gateway/test_direct_control_event_runtime_service.py`
- Test: `tests/gateway/test_group_control_requests.py`
- Test: `tests/gateway/test_qq_intel_control_requests.py`
- Test: `tests/gateway/test_auto_background_jobs.py`

**Step 1: Write the failing tests**

Add coverage for:
- leading conversational wrappers like `我让你`, `你现在`, `帮我把` not polluting direct-control matching
- group-control requests still winning over intel parsing when worker context is not explicit
- non-admin or non-targeted text still returning `None`

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_direct_control_event_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py -q`
Expected: FAIL once tests assert normalization behavior that is not implemented yet.

**Step 3: Write minimal implementation**

Add a small shared normalizer in `gateway/direct_control_event_runtime_service.py` that preserves meaning but strips obvious address/politeness wrappers before downstream matchers run. Keep matcher APIs the same, and only use the normalized body for shortcut parsing.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_direct_control_event_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/direct_control_event_runtime_service.py gateway/group_control_requests.py gateway/qq_intel_control_requests.py tests/gateway/test_direct_control_event_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py
git commit -m "Harden direct control text normalization"
```

### Task 3: Unify Foreground Vision Preparation Around Production Auto-Vision Flow

**Files:**
- Modify: `gateway/foreground_turn_runtime_service.py`
- Modify: `gateway/vision_orchestrator.py`
- Test: `tests/gateway/test_vision_orchestrator.py`
- Test: `tests/gateway/test_auto_vision_enrichment.py`

**Step 1: Write the failing tests**

Add coverage for:
- image-only turns using the same degraded-note semantics as production auto-vision
- no direct user-facing “重发一下/补一句你要我看什么” fallback from the unused path
- shared helper behavior matching the real foreground enrichment contract

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_vision_orchestrator.py tests/gateway/test_auto_vision_enrichment.py -q`
Expected: FAIL once tests assert runtime-consistent foreground behavior.

**Step 3: Write minimal implementation**

Refactor `gateway/vision_orchestrator.py` to match the real foreground pipeline contract instead of maintaining a separate divergent behavior. Use the shared degraded-note wording and remove divergent direct-reply fallbacks for image-only turns unless the caller explicitly requests them.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_vision_orchestrator.py tests/gateway/test_auto_vision_enrichment.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/foreground_turn_runtime_service.py gateway/vision_orchestrator.py tests/gateway/test_vision_orchestrator.py tests/gateway/test_auto_vision_enrichment.py
git commit -m "Align gateway foreground vision behavior"
```

### Task 4: Final Verification Batch

**Files:**
- Modify: none unless regressions appear
- Test: `tests/gateway/test_auto_background_runtime_service.py`
- Test: `tests/gateway/test_direct_control_event_runtime_service.py`
- Test: `tests/gateway/test_group_control_requests.py`
- Test: `tests/gateway/test_qq_intel_control_requests.py`
- Test: `tests/gateway/test_send_runtime_service.py`
- Test: `tests/gateway/test_direct_shortcuts.py`
- Test: `tests/gateway/test_auto_background_jobs.py`
- Test: `tests/gateway/test_auto_vision_enrichment.py`
- Test: `tests/gateway/test_vision_orchestrator.py`

**Step 1: Run the focused regression suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tests/gateway/test_auto_background_runtime_service.py \
  tests/gateway/test_direct_control_event_runtime_service.py \
  tests/gateway/test_group_control_requests.py \
  tests/gateway/test_qq_intel_control_requests.py \
  tests/gateway/test_send_runtime_service.py \
  tests/gateway/test_direct_shortcuts.py \
  tests/gateway/test_auto_background_jobs.py \
  tests/gateway/test_auto_vision_enrichment.py \
  tests/gateway/test_vision_orchestrator.py -q
```

Expected: PASS

**Step 2: Run lightweight syntax verification**

Run:

```bash
source .venv/bin/activate && python -m compileall \
  gateway/auto_background_runtime_service.py \
  gateway/direct_control_event_runtime_service.py \
  gateway/group_control_requests.py \
  gateway/qq_intel_control_requests.py \
  gateway/vision_orchestrator.py \
  gateway/foreground_turn_runtime_service.py
```

Expected: PASS

**Step 3: Commit**

```bash
git add gateway/auto_background_runtime_service.py gateway/direct_control_event_runtime_service.py gateway/group_control_requests.py gateway/qq_intel_control_requests.py gateway/vision_orchestrator.py gateway/foreground_turn_runtime_service.py gateway/data/auto_background_intents.json tests/gateway/test_auto_background_runtime_service.py tests/gateway/test_direct_control_event_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py tests/gateway/test_auto_vision_enrichment.py tests/gateway/test_vision_orchestrator.py
git commit -m "Harden gateway runtime routing and vision prep"
```
