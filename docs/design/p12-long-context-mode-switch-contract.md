# P12 Long-Context Route — Mode-Switch Contract

**Status:** Proposed (spec for implementation)
**Date:** 2026-08-09
**Owner:** Kensei (spec), implementation split across
  `t_fb7af9a5` (Turbohaul mode manager), `t_18b0ffeb` (route wiring), `t_74d0713a` (E2E).
**Consumes:** `docs/adr/0012-p12-256k-memory-policy.md` (policy),
  `scripts/kv_cache_policy.py` (tier decision), Turbohaul-Manager v0.2 FSM/manifest model,
  `~/brain/conventions/performance-rules.md` (canonical P12 performance rule).
**Consumed by:** route wiring, Turbohaul mode manager, E2E tests.

---

## 1. Purpose

This document defines the **explicit long-context route** contract: the exact
trigger that lets a client request a long-context (RAM-KV) generation, the
automatic GPU-KV → RAM-KV mode switch, the guarantee that normal GPU-KV mode is
restored immediately after the request completes, and the defined behaviour for
every failure/timeout/concurrency edge case.

It is written so the mode manager, the route wiring, and the E2E suite can be
implemented against it **without ambiguous mode ownership**. The single-owner
rule (§6) is the load-bearing invariant; everything else follows from it.

---

## 2. Terminology

| Term | Meaning |
|---|---|
| **GPU-KV mode** (normal / fast path) | KV cache resident in VRAM (`llama-server` default; no `--no-kv-offload`). Context ≤ 128K. This is the default for all daily traffic. |
| **RAM-KV mode** (long-context) | KV cache in host system RAM via `no_kv_offload: true` (manifest flag → `--no-kv-offload`). Context up to 256K. Weights stay GPU0-only. Per ADR 0012, offload is opt-in and targets host RAM, never GPU1. |
| **Explicit long-context route** | The client-visible trigger defined in §3. Requesting it is the *only* way to enter RAM-KV mode. |
| **Mode switch** | A Turbohaul-managed **cold respawn** of the main-lane sidecar with a different manifest argv (RAM-KV manifest in, GPU-KV manifest out). Because Turbohaul is single-slot-per-GPU and llama.cpp flags only take effect on cold spawn (KV_CACHE_OFFLOADING.md §2), a switch is **not** a live toggle — it is a supervised restart with policy-gated argv. |
| **Mode session** | The duration from a successful switch-in to the guaranteed switch-back. Scoped to one request/route invocation. |

---

## 3. The explicit long-context route

### 3.1 Endpoint

The explicit long-context route is the Turbohaul OpenAI-compatible endpoint
with a **reserved model tag** that maps to the long-context manifest:

```
POST /v1/chat/completions
  { "model": "darwin-28b-256k", ... }
```

and the Ollama-compatible equivalent:

```
POST /api/chat
  { "model": "darwin-28b-256k", ... }
```

**The model tag is the trigger.** `darwin-28b-256k` is the reserved, explicit
long-context route marker. A request that names `darwin-28b-256k` is a
long-context request; a request that names `darwin-28b` (no suffix) is a
normal request and must never enter RAM-KV mode.

> Tag-safety note: Turbohaul's manifest `model_tag` is validated against
> `^[a-z0-9][a-z0-9._-]{0,63}$` (`TAG_RE` in `src/turbohaul/manifest.py`), so
> colons are **not** allowed. The hyphenated tag above is the valid form and
> follows the existing store convention (`darwin-28b-reason`,
> `darwin-28b-coder`). Do not use an Ollama-style `tag:256k` — it will be
> rejected by `validate_tag`.

### 3.2 Manifest

The route is backed by a dedicated manifest that exists in the manifests store
alongside the normal main-lane manifest:

```yaml
# state/manifests/darwin-28b-256k.yaml  (illustrative; exact flags per policy gate)
model_tag: darwin-28b-256k
llama_server_flags:
  ctx_size: 262144            # 256K
  no_kv_offload: true         # RAM-KV (the canonical affirmative form; never kv_offload: false)
  split_mode: none            # GPU0-only weights (ADR 0012 invariant)
  main_gpu: 0                 # GPU0-only weights (ADR 0012 invariant)
  # optional: parallel / kv_unified / cont_batching per capacity plan
```

Policy gate: the manifest must pass `scripts/p12_offload_gate.py`
(`P12_ALLOW_HUGE_CONTEXT_OFFLOAD=1` opt-in; see ADR 0012) before it is
eligible for spawn. If the opt-in is not set, the mode manager must refuse the
switch with a clear error (§6.5).

### 3.3 Automatic switch, not manual toggle

The user never toggles modes. The route handler (t_18b0ffeb) calls the mode
manager (t_fb7af9a5) automatically:

1. **On route entry:** request RAM-KV mode for this request.
2. **On route exit (success, error, or timeout):** request restore to GPU-KV.
3. **No manual action** by the user at any point.

---

## 4. Mode-switch lifecycle

### 4.1 Normal flow (happy path)

```
client → POST /v1/chat/completions {model: "darwin-28b-256k"}
        │
        ▼
route handler (t_18b0ffeb)
        │  1. mode_manager.enter_long_context(request_id)
        ▼
mode manager (t_fb7af9a5)
        │  2. policy check (opt-in gate + manifest allowlist + VRAM/RAM fit)
        │  3. COLD SPAWN RAM-KV sidecar (manifest darwin-28b-256k)
        │  4. mark mode session active for request_id
        ▼
        │  (sidecar now serving in RAM-KV mode)
        ▼
inference on RAM-KV sidecar  ──►  completion stream / response
        │
        ▼
route handler
        │  5. finally: mode_manager.restore(request_id)
        ▼
mode manager
        │  6. COLD SPAWN GPU-KV sidecar (manifest darwin-28b)  [swap back]
        │  7. mark mode session closed; log transition with request_id
        ▼
normal GPU-KV mode restored
```

### 4.2 Timing and ownership

- **Switch-in latency** is a cold-spawn (weight load) — the first token of a
  long-context request pays it. This is accepted and documented; it is not a
  hang. The route must surface it as "loading long-context engine" progress,
  not dead-air.
- **Ownership is exclusive.** At most **one mode session** exists at a time
  across the whole Turbohaul manager. The mode session owns the sidecar for
  its lifetime. §6 defines exactly who may enter/exit and who may not.

### 4.3 Restore is immediate

"Immediately after the long-context request/use completes" means: the restore
is issued **in a `finally` block** (or equivalent guaranteed path) the moment
the request finishes — success, failure, client disconnect, or timeout. There
is no grace window and no deferred cleanup. The normal GPU-KV sidecar is
respawned and made active before the route returns its final response.

---

## 5. Edge cases (defined behaviour)

### 5.1 Concurrent requests

| Scenario | Behaviour |
|---|---|
| Normal (GPU-KV) request arrives while a long-context session is active | The long-context session is exclusive. The normal request is **queued** behind the mode session (Turbohaul queue semantics) and served by the restored GPU-KV sidecar after switch-back. It must never be served by the RAM-KV sidecar and must never trigger a mode switch. |
| Two long-context requests arrive concurrently | **Serialized.** One acquires the mode session; the second waits on the same mode lock (or is rejected with `429` + `Retry-After` if the wait budget is exceeded). §6.3 defines the tie-break. |
| Long-context request arrives while a normal request is in flight | The normal request finishes on the GPU-KV sidecar first; only then does the mode session start. The switch never preempts an in-flight request. |

**Invariant:** a mode switch never happens while a request is in flight on the
sidecar being replaced. Turbohaul's FSM (`ACTIVE`/`GRACE`/`GRACE_BUSY`) is the
arbiter: switch only from a quiescent state.

### 5.2 Request failure

| Failure | Behaviour |
|---|---|
| Inference fails mid-stream (sidecar error, upstream 5xx) | Route returns the error to the client; `finally` still runs; restore to GPU-KV is issued and completes. The mode session is closed with outcome `failed`. |
| Client disconnects mid-stream | `watch_disconnect` path (already in chat_completion.py) aborts; restore is issued. The mode session is closed with outcome `disconnected`. |
| Model/tag not found | Route returns 404; no mode switch occurs at all (the route never entered the mode session). |
| Policy gate refuses offload (opt-in unset) | Route returns 503 with an actionable message (which env var to set); **no** mode switch, no sidecar respawn. GPU-KV mode untouched. |

### 5.3 Timeout

| Timeout | Behaviour |
|---|---|
| Switch-in times out (sidecar fails to boot in `MAX_SWITCH_IN_S`, default 120s) | Abort the mode session. Do **not** fall back to serving the request on the GPU-KV sidecar at 256K (it cannot fit). Return 503 `Retry-After`. GPU-KV mode restored (or never left). |
| Generation times out (Turbohaul sidecar timeout, `SidecarTimeoutError`) | Return 504/503 per existing route semantics; `finally` restores GPU-KV. |
| Switch-back (restore) times out | **Escalation path** (§6.5): mark the mode session `restore_failed`, surface a health/ops alert, and retry restore with backoff until GPU-KV is back. The sidecar is **not** left running in RAM-KV mode indefinitely; the supervisor retries. |

### 5.4 Already-in-long-context state

| State | Behaviour |
|---|---|
| Route called while a mode session is already active for the same request_id | No-op re-entry: the session is reused (idempotent `enter`). |
| Route called with a *different* request_id while a session is active | The new request waits on the mode lock (or 429 per §5.1). It never nests another session. |
| Session ends twice (double restore) | `restore` is idempotent: the second call is a no-op that logs `already_restored`. |
| Process/manager restart while a session is active | On boot reconcile, Turbohaul's `boot_reconcile` sees no active mode session (it was in-memory); the sidecar resident state is reconciled to the normal manifest. RAM-KV mode never survives a restart. |

### 5.5 Repeated routing

| Pattern | Behaviour |
|---|---|
| N consecutive long-context requests | Each is its own mode session: enter → serve → restore → enter → serve → restore. Restore before the next enter is guaranteed, so the GPU-KV sidecar is the steady-state between sessions. |
| Long-context request immediately after a normal request | Normal request served on GPU-KV; then mode session starts. No interaction. |
| Hammering the route (rapid enter/restore) | Serialized by the mode lock; each transition is a cold spawn. The mode manager must not queue more than `MAX_QUEUED_MODE_SESSIONS` (default 4); beyond that, 429. This protects the rig from spawn storms. |

### 5.6 Sidecar/manifest anomalies

| Anomaly | Behaviour |
|---|---|
| RAM-KV manifest missing from store | Route returns 503 `manifest_not_found`; no mode switch. |
| Manifest fails flag allowlist / policy gate | Route returns 503 `policy_refused` with the gate reason; no mode switch. |
| GPU-KV sidecar fails to restore | Escalation path (§6.5): ops alert; retry; do not return success to the client while the stack is degraded. The long-context *response* may already be delivered, but restore must still converge. |

---

## 6. Mode ownership

### 6.1 Single owner

The **mode manager** (t_fb7af9a5, `TurbohaulModeManager`) is the **sole owner**
of mode state. It owns:

- the mode lock,
- the mode session registry (request_id → session),
- all spawn/restore calls,
- all mode-transition logs.

The **route handler** (t_18b0ffeb) is the **sole trigger**. It calls
`enter_long_context(request_id)` and `restore(request_id)` and nothing else
mode-related. It never inspects or mutates mode state directly.

The **policy gate** (`scripts/p12_offload_gate.py` + `scripts/kv_cache_policy.py`)
is the **sole decision** authority for *whether* a switch is permitted and
*which* tier applies. Neither the route nor the manager re-implements the
decision.

### 6.2 API (consumed by t_fb7af9a5)

```python
class TurbohaulModeManager:
    async def enter_long_context(self, request_id: str) -> ModeSession:
        """Acquire the mode lock, policy-check, cold-spawn RAM-KV, register session."""

    async def restore(self, request_id: str) -> None:
        """Idempotent. Cold-spawn GPU-KV, close session, release lock."""

    def mode_snapshot(self) -> dict:
        """Current mode, active session, sidecar state — for /status and tests."""

    # internal: _acquire_lock(timeout), _spawn_ram_kv(), _spawn_gpu_kv(),
    #           _escalate_restore_failure(session), _log_transition(...)
```

### 6.3 Locking and tie-break

- A single **asyncio lock** (`mode_lock`) guards enter/restore.
- `enter_long_context` has a **wait budget** (`MODE_ACQUIRE_TIMEOUT_S`, default
  30s). If the lock is not acquired within the budget → `429` + `Retry-After`.
- Tie-break for concurrent long-context requests: **FIFO** by route arrival
  (matching Turbohaul's queue). No starvation; no priority inversion.

### 6.4 Logging / observability

Every transition logs a structured line with the `request_id`:

```
mode_transition event=enter request_id=... mode=ram_kv spawn_seq=... ok=true
mode_transition event=restore request_id=... mode=gpu_kv spawn_seq=... ok=true
mode_transition event=restore_failed request_id=... error=... retry=1
```

`/status` includes `mode` (`gpu_kv` | `ram_kv` | `switching` | `restore_failed`)
and the active `mode_session` (or null). The E2E suite (t_74d0713a) asserts on
both the state and the logs.

### 6.5 Failure escalation

- Restore failure is the only state that may outlive a request. The manager
  retries restore with exponential backoff (2s, 4s, 8s, … cap 60s, max 5
  attempts) and, on exhaustion, marks `mode=restore_failed` and raises an ops
  alert (health/cron surface). It must never silently leave the stack in
  RAM-KV mode.
- Policy-refused switches and manifest failures are client-facing (503) and do
  not enter the mode session at all.

---

## 7. What is explicitly out of scope

- **Automatic (implicit) long-context selection** for normal requests. The
  `kv_cache_policy.py` `auto` mode decides tiers for *policy* purposes, but the
  *route* only enters RAM-KV mode on the explicit `darwin-28b-256k` tag. There
  is no threshold-triggered auto-switch of the running sidecar.
- **Weight offload** (never; ADR 0012).
- **GPU1 usage** for the main lane (never; ADR 0012).
- **Live (in-process) KV relocation** — llama.cpp has no such API; switches are
  cold respawns.
- **Multiple simultaneous long-context sessions** (exclusive by design).

---

## 8. Acceptance mapping

| Acceptance criterion | Where it is satisfied |
|---|---|
| Exact explicit long-context route/trigger named and documented | §3 — `POST /v1/chat/completions` + `/api/chat` with `model: darwin-28b-256k`; the model tag is the trigger. |
| Automatic GPU-KV → RAM-KV switch and restore behaviour specified, incl. timing and ownership | §4 (lifecycle, timing) + §6 (single owner: mode manager; sole trigger: route; sole decision: policy gate). |
| Failure/timeout/concurrency/edge cases covered with defined behaviour | §5 (5.1–5.6) — every scenario has a defined, testable behaviour. |
| Review with engineering and UX stakeholders | This spec is the review artifact; walk through §5 with the implementation + UX owner. |
| Implementable without ambiguous mode ownership | §6.1 — ownership is partitioned into trigger / owner / decision; no overlap. |

---

## 9. References

- `~/brain/conventions/performance-rules.md` — canonical **P12 performance rule**: ordinary daily work must feel fast; 256K is a required capability; occasional near-256K jobs may run slower. This contract implements the slow-path (RAM-KV) part of that rule.
- `docs/adr/0012-p12-256k-memory-policy.md` — memory policy (weights GPU0-only,
  KV offload opt-in).
- `scripts/kv_cache_policy.py` — tier decision source of truth (128K threshold,
  256K max).
- `scripts/p12_offload_gate.py` — argv sanitization + policy gate for the
  offload manifest.
- Turbohaul-Manager `docs/KV_CACHE_OFFLOADING.md` — `no_kv_offload` mechanism,
  cold-spawn semantics, VRAM/RAM math.
- Turbohaul-Manager `src/turbohaul/fsm.py` — 10-state slot FSM (quiescence
  arbiter for switches).
- Turbohaul-Manager `src/turbohaul/api/chat_completion.py` — existing
  `/v1/chat/completions` + `/api/chat` routes, `watch_disconnect`,
  `SidecarTimeoutError` semantics.
