# XO-909 Execution Backend Adapter Contract (v1)

**Slice:** `XO-909` · **Scope:** Optional future upgrades lane (`XO`) · **Mode:** Bounded execution-path abstraction

## 1) Objective
Introduce an execution backend adapter contract that standardizes provider and transport selection plus bounded retry behavior for execution tasks, while avoiding adoption of Chandra’s broad synchronous batch-system posture.

## 2) Scope and Non-goals
- **In-scope:** provider taxonomy, capability matrix, precedence-aware resolver, retry policy envelope, and evidence artifacts.
- **Out-of-scope:** any direct changes to production queue routing logic, transport workers, credential stores, or vendor API clients.

## 3) Contract boundaries (schema/runtime boundary)
- Control boundary stays within **lane `XO` optional hardening artifacts**.
- Runtime boundary must remain in **runtime pack + validation** artifacts under `state/continuity/latest/`.
- Runtime outputs must remain explicit about precedence and failure reasons to preserve auditability.

## 4) Runtime anchor defaults
- `default_transport`: `direct`
- `max_fallback_attempts`: `3`
- `default_timeout_seconds`: `30`
- `default_max_retries`: `2`
- `disallowed_transports`: `["batch", "queue_batch"]` (prevents implicit batch-posture drift)

## 5) Provider/transport schema
Each provider entry declares:
- `provider_id`
- `task_classes` it can execute
- `transport_modes` (mode + timeout + retry policy)

### Providers (initial catalog)
- `provider_chandra_hf`
- `provider_onyx_worker`
- `provider_chandra_vllm`

### Transport support rules
- Selection tests only over declared modes for that provider.
- `disallowed_transports` are rejected before task-class checks.
- Retry policy is provider/transport specific.

## 6) Resolution precedence
For each request, resolver order is:
1. **Request override** (provider and/or transport)
2. **Profile overrides** (`task_profile` defaults)
3. **Lane defaults** (`selection_defaults`)
4. **fallback** (if all attempts rejected)

### Bounded behavior
- Attempts are bounded by `max_fallback_attempts`.
- If all attempts are exhausted without a valid provider/transport pair, status is `FAIL` and failure reasons are preserved.

## 7) Retry/backoff posture
- Retry policy for each transport is bounded (`max_retries`, explicit `backoff_seconds` vector).
- Retry is explicit and deterministic by policy shape, never unbounded.
- Retry simulation evidence reports computed delay sequences and boundedness checks.

## 8) Why this is Chandra-lite
This slice captures Chandra’s strengths (explicit backend abstraction and config precedence) without inheriting Chandra’s full batch-system assumptions:
- no global batch queue required,
- no cross-item synchronous bulk fanout contract,
- no forced batch transport default.

## 9) Evidence artifacts produced
- Contract pack (`state/continuity/latest/xo_909_backend_adapter_contract_pack_<stamp>.json`)
- Selection matrix simulation (`state/continuity/latest/xo_909_backend_selection_matrix_simulation_<stamp>.json`)
- Retry/backoff regression (`state/continuity/latest/xo_909_backend_retry_backoff_regression_<stamp>.json`)
- Runtime manifest + validation packet

## 10) Closeout criteria
- Contract is bounded, precedence-documented, and test-covered.
- No generated selection selects `batch`.
- Retry backoff evidence demonstrates finite delay vectors and bounded retries.
- Optional queue entry `XO-909` transitions to `DONE` with closeout report path recorded.
