# Tiny Router Integration (Hermes)

This document describes the tiny-router integration added to Hermes for per-turn message classification and policy routing.

Repository for the classifier: <https://github.com/UdaraJay/tiny-router>

## What This Adds

Hermes now classifies each user turn into four heads before execution:

- `relation_to_previous`: `new | follow_up | correction | confirmation | cancellation | closure`
- `actionability`: `none | review | act`
- `retention`: `ephemeral | useful | remember`
- `urgency`: `low | medium | high`

The classification is consumed by routing and policy components, and persisted as message metadata.

## Core Behavior

### 1) Per-turn routing signal

- New module: `agent/tiny_router.py`
- Supports:
  - subprocess backend (runs tiny-router `scripts.predict`)
  - deterministic heuristic fallback
  - multimodal-safe input normalization (extracts text from list-style message parts)
- Used by:
  - CLI flow (`cli.py`)
  - gateway flow (`gateway/run.py`)
  - `AIAgent.run_conversation(...)` (`run_agent.py`)

### 2) Route selection integration

- `agent/smart_model_routing.py` now accepts tiny-router output.
- `smart_model_routing.tiny_router.behavior_mode`:
  - `shadow`: classify + persist, but do not strongly alter model route decisions
  - `active`: let tiny-router drive route-policy decisions when confidence/policy conditions match
- Routing policy is centralized in `smart_model_routing`:
  - simple-turn heuristics, tiny-router low/high-stakes decisions, and named route profiles are evaluated in one place
  - `smart_model_routing.routes` can define extra profiles (for example, a local model lane) beyond `cheap_model`
  - `smart_model_routing.tier_routes` maps `low|medium|high` quality tiers to route names (`cheap`, `primary`, or custom routes)
  - `smart_model_routing.max_high_tier_calls_per_session` can cap expensive turns per session

### 3) Policy integration

- Memory behavior in `run_agent.py`:
  - `retention=remember` can boost memory review behavior
  - high urgency + actionable turns can trigger aggressive pre-persist memory flushing
- Context compression in `agent/context_compressor.py`:
  - `retention=useful|remember` turns are favored near compression boundaries
- Approval posture in `tools/approval.py`:
  - review-classified turns can escalate risky command execution to explicit approval

### 4) Persistence

- SQLite schema version bumped in `hermes_state.py`:
  - `messages.metadata` added as JSON text
- tiny-router output is stored under message metadata:
  - `metadata.tiny_router.*`

## Config

New config section in `hermes_cli/config.py`:

```yaml
smart_model_routing:
  tier_routes:
    low: cheap
    medium: primary
    high: primary
  max_high_tier_calls_per_session: 0
  tiny_router:
    enabled: false
    backend: subprocess          # subprocess | heuristic
    repo_root: ${TINY_ROUTER_REPO_ROOT}
    model_dir: ${TINY_ROUTER_MODEL_DIR}
    pinned_commit: 9d6b2a718a205d90ebe85e9a28f9b8a1f20801e4
    enforce_pinned_commit: true
    source_revision_file: REVISION
    predict_timeout_seconds: 30
    fallback_mode: heuristic     # heuristic | none
    behavior_mode: shadow        # shadow | active
    apply_approval_posture: true
    confidence_thresholds:
      overall: 0.45
      relation_to_previous: 0.5
      actionability: 0.5
      retention: 0.5
      urgency: 0.5
```

New optional env vars:

- `TINY_ROUTER_REPO_ROOT`
- `TINY_ROUTER_MODEL_DIR`

## Version pinning model

tiny-router currently does not publish GitHub release tags in the upstream repo.
Hermes therefore pins to an immutable upstream commit SHA (`pinned_commit`) for
reproducible behavior.

If `enforce_pinned_commit: true`, Hermes startup validation fails when the local
source revision does not match `pinned_commit`.

Validation order:

1. read `git rev-parse HEAD` under `repo_root` (normal checkout), or
2. if git metadata is unavailable, read `source_revision_file` (default: `REVISION`)

This supports both git checkouts and non-git source bundles while keeping the
pin deterministic. Updating this pin is a normal Hermes maintenance task.

## Rollout Guidance

1. Enable in `shadow` mode first.
2. Validate output quality + operational impact.
3. Move to `active` mode only after confidence tuning.
4. Keep a rollback path by setting `smart_model_routing.tiny_router.enabled: false`.

## Rollback

Fast rollback:

- set `smart_model_routing.tiny_router.enabled: false`

This returns Hermes to pre-integration routing behavior without schema rollback.

## Files Changed (high level)

- `agent/tiny_router.py`
- `agent/smart_model_routing.py`
- `run_agent.py`
- `cli.py`
- `gateway/run.py`
- `tools/approval.py`
- `agent/context_compressor.py`
- `hermes_state.py`
- `hermes_cli/config.py`
- tests:
  - `tests/agent/test_tiny_router.py`
  - `tests/agent/test_smart_model_routing.py`
  - `tests/test_cli_provider_resolution.py`
  - `tests/test_hermes_state.py`

