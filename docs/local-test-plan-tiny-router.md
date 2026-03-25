# Local Test Plan: Tiny Router Integration

This runbook is for validating tiny-router integration locally before opening a public PR.

## Goal

Confirm that tiny-router classification is:

1. generated per turn
2. consumed by routing/policy logic
3. persisted in SQLite metadata
4. safe under shadow rollout and predictable under active rollout

## Prerequisites

From repo root:

```bash
source venv/bin/activate
```

If your environment does not have all optional test deps installed:

```bash
python -m pip install fire firecrawl-py fal-client
```

## A. Automated Regression (required)

Run:

```bash
python -m pytest -o addopts='' \
  tests/agent/test_tiny_router.py \
  tests/agent/test_smart_model_routing.py \
  tests/test_cli_provider_resolution.py \
  tests/test_hermes_state.py \
  tests/tools/test_approval.py \
  tests/agent/test_context_compressor.py \
  tests/gateway/test_background_command.py \
  tests/gateway/test_transcript_offset.py \
  -q
```

Expected:

- all tests pass
- only known warnings (if any), no new failures

Optional full suite:

```bash
python -m pytest tests/ -q
```

## B. Config Validation (required)

Set tiny-router in `shadow` mode first.

Example (heuristic-only backend for portability):

```yaml
smart_model_routing:
  tiny_router:
    enabled: true
    backend: heuristic
    behavior_mode: shadow
    fallback_mode: heuristic
    apply_approval_posture: true
```

Start CLI:

```bash
python cli.py
```

Expected:

- startup succeeds
- no config validation error for tiny-router

For subprocess backend, configure:

```yaml
smart_model_routing:
  tiny_router:
    enabled: true
    backend: subprocess
    repo_root: /abs/path/to/tiny-router
    model_dir: /abs/path/to/tiny-router/artifacts/tiny-router
    pinned_commit: 9d6b2a718a205d90ebe85e9a28f9b8a1f20801e4
    enforce_pinned_commit: true
    source_revision_file: REVISION
```

Expected:

- startup fails fast with clear error if path/script/model is invalid
- startup fails fast if source revision does not match `pinned_commit` (git HEAD first, then `source_revision_file` fallback)
- startup succeeds when valid

## C. Manual Functional Scenarios (required)

## Scenario 1: Shadow mode is non-disruptive

1. Keep `behavior_mode: shadow`
2. Send simple prompt and complex prompt
3. Verify normal model behavior still feels consistent

Expected:

- no surprising route changes
- no regressions in normal agent execution

## Scenario 2: Active mode route influence

1. Set `behavior_mode: active`
2. Configure `smart_model_routing.cheap_model`
   - Optional: define `smart_model_routing.routes` and `tier_routes` so
     low/medium/high tiers map to named profiles (for example: local-fast,
     local-strong, remote-strong)
   - Optional: set `max_high_tier_calls_per_session` to validate expensive-turn caps
3. Send:
   - low-urgency/simple prompt
   - high-urgency/action prompt

Expected:

- simple prompt can route to cheap model
- high-urgency/action prompt tends to route to a high-tier profile
- after budget cap is hit, high-tier routes fall back to configured medium tier

## Scenario 3: Approval posture

1. Use a turn likely to classify as review
2. Execute a command that hits guardrails

Expected:

- approval flow is triggered/escalated rather than silently executing

## Scenario 4: Memory/compression behavior

1. Trigger a high-actionability + high-urgency turn
2. Force a long-enough context to approach compression behavior

Expected:

- memory flush bias appears on urgent/action-heavy turns
- retention-priority turns are less likely to be dropped during compression boundary logic

## D. Persistence Validation (required)

Run a few CLI turns with tiny-router enabled, then inspect SQLite:

```bash
python - <<'PY'
from hermes_state import SessionDB
db = SessionDB()
sessions = db.search_sessions(limit=1)
if not sessions:
    print("No sessions found")
else:
    sid = sessions[0]["id"]
    msgs = db.get_messages(sid)
    hits = [m for m in msgs if isinstance(m.get("metadata"), dict) and "tiny_router" in m["metadata"]]
    print("session:", sid)
    print("messages:", len(msgs))
    print("tiny_router_metadata_messages:", len(hits))
db.close()
PY
```

Expected:

- at least one user message has `metadata.tiny_router`
- JSON structure includes all four heads and source/confidence fields

## E. Gateway Smoke (recommended)

Run gateway-related tests:

```bash
python -m pytest -o addopts='' \
  tests/gateway/test_background_command.py \
  tests/gateway/test_transcript_offset.py \
  -q
```

Expected:

- no failures
- background task flow handles tiny-router classification cleanly

## F. Sign-off Checklist

- [ ] Targeted regression suite passes
- [ ] Shadow mode manual pass complete
- [ ] Active mode route behavior verified
- [ ] Approval posture scenario verified
- [ ] SQLite metadata persistence verified
- [ ] Gateway smoke tests pass
- [ ] No new lint/type issues in touched files

