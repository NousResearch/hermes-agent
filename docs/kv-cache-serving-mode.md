# KV-Cache Serving-Mode Policy and Configuration

**Status:** Implemented
**Date:** 2026-08-09
**Applies to:** P12 local inference stack (`KenseiAgent`, dual-RTX-3090 rig through Turbohaul)
**Policy authority:** `docs/adr/0012-p12-256k-memory-policy.md`
**Performance rule:** `~/brain/conventions/performance-rules.md` — **P12**: *"The main model must feel fast for ordinary daily work. 256K context is a required capability, but occasional near-256K jobs may run slower."* (promoted 2026-08-09). This serving-mode policy is the operational expression of that rule.
**Implementation:** `scripts/kv_cache_policy.py` (single source of truth), `scripts/p12_offload_gate.py` (launch-path gate), `scripts/p12_offload_live_check.py` (live integration check)

---

## 1. The decision in one paragraph

The P12 serving stack runs **two KV-cache tiers**:

| Tier | Where the KV cache lives | Used for |
|---|---|---|
| **GPU-resident (fast path)** | VRAM, next to the weights | All ordinary contexts — **≤ 128K tokens by default** |
| **Host-RAM offload** | CPU/system RAM (`--no-kv-offload`) | Rare long-context work — **> 128K up to 256K tokens** |

The default is the **GPU-resident fast path**. Host-RAM offload is used only for
rare 128K–256K context work, is **strictly opt-in**, and never moves model
weights — weights stay GPU0-only at all times (ADR 0012).

The tier for a request is decided by a **single placement policy**
(`scripts/kv_cache_policy.py`, `decide_tier()`), not by ad-hoc checks in each
launch wrapper. The GPU fast path and the offload gate both read this one
policy, so the boundary can never drift between them.

---

## 2. How the placement policy chooses a tier

`decide_tier(context_len, override=None)` evaluates one request and returns a
`PlacementDecision` with a `tier` of `gpu` or `offload`.

**Default rule (`mode=auto`):**

```
context_len <= KV_CACHE_OFFLOAD_THRESHOLD   -> GPU-resident (fast path)
context_len >  KV_CACHE_OFFLOAD_THRESHOLD
    and <= KV_CACHE_MAX_CONTEXT              -> host-RAM offload
context_len >  KV_CACHE_MAX_CONTEXT          -> rejected (ContextOutOfRangeError)
```

With defaults that is:

- `<= 131072` (128K) → GPU-resident
- `131073..262144` (128K+1 .. 256K) → host-RAM offload
- `> 262144` (256K) → rejected up front — the supported maximum is 256K

Boundary specifics (all covered by `tests/scripts/test_kv_cache_policy.py`):

- Exactly 128K is **GPU-resident** (the threshold is inclusive).
- 128K+1 is **host-RAM offload**.
- A 0-length context (no prior context) is a valid request and stays on the
  GPU fast path.
- A negative context is a configuration error (`PolicyConfigError`).
- A context above the max raises `ContextOutOfRangeError` — the request is
  rejected before any KV-cache allocation is attempted.

**Mode overrides:** `KV_CACHE_MODE` can force a tier regardless of context
length (`gpu` / `offload`), and a per-request `override` argument wins over
everything — see §3.4 and §7.

**The offload gate uses the same policy.** `scripts/p12_offload_gate.py`
(`decide_offload(argv, ctx_size)`) consults `decide_tier()` for offload
eligibility and then applies the opt-in flag and argv sanitization on top. It
never re-implements the threshold.

---

## 3. Configuration reference

All policy configuration is read from environment variables, with an optional
JSON config file layered underneath. **Precedence: built-in defaults < config
file < environment variables.** A per-request override beats all of them (§3.4).

### 3.1 Environment variables

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `KV_CACHE_MODE` | string: `auto` \| `gpu` \| `offload` | `auto` | `auto` decides by context length; `gpu` forces GPU-resident; `offload` forces host-RAM offload. Forced modes are marked `forced=true` in the decision. |
| `KV_CACHE_OFFLOAD_THRESHOLD` | int (tokens) | `131072` (128K) | Contexts at or below this stay GPU-resident; above it (up to `KV_CACHE_MAX_CONTEXT`) route to host-RAM offload. Must be ≥ 1. |
| `KV_CACHE_MAX_CONTEXT` | int (tokens) | `262144` (256K) | Supported maximum context. Requests above this are rejected. Must be ≥ `KV_CACHE_OFFLOAD_THRESHOLD`. |
| `KV_CACHE_CONFIG` | path | *(unset)* | Path to a JSON config file (§3.2). Missing/unreadable/invalid file is a hard config error — a typo'd path never silently falls back to defaults. |
| `P12_ALLOW_HUGE_CONTEXT_OFFLOAD` | string: `1` \| `true` \| `yes` \| `on` | *(unset = OFF)* | **Opt-in switch for the launch path.** The policy may *route* a long context to offload, but the offload gate will only keep `--no-kv-offload` in the server argv when this flag is set. Anything else (unset, `0`, `false`, `no`, `off`, typo) means OFF and the fast path is preserved. |
| `HUGE_CONTEXT_MIN_TOKENS` | int (tokens) | `131072` (alias) | **Backward-compatible alias** for the old gate constant. It now resolves to `DEFAULT_OFFLOAD_THRESHOLD` from the policy module. New code should use `KV_CACHE_OFFLOAD_THRESHOLD`; existing importers keep working. |

### 3.2 JSON config file (`KV_CACHE_CONFIG`)

Keys mirror the env vars:

```json
{
  "mode": "auto",
  "offload_threshold": 131072,
  "max_context": 262144
}
```

Any key may be omitted; omitted keys fall back to the next precedence layer
(defaults, or env if the env var is set). The file must contain a JSON object.

### 3.3 Config precedence example

```bash
# KV_CACHE_CONFIG points at a file with offload_threshold=8192
KV_CACHE_CONFIG=/etc/kensei/kv-cache.json python scripts/kv_cache_policy.py --ctx-size 10000
# -> file alone would route 10000 to offload (threshold 8192)

# Env overrides the file:
KV_CACHE_CONFIG=/etc/kensei/kv-cache.json KV_CACHE_OFFLOAD_THRESHOLD=131072 \
    python scripts/kv_cache_policy.py --ctx-size 10000
# -> env threshold wins: 10000 <= 131072, GPU fast path
```

The decision carries a `source` field (`defaults` | `file` | `env`) so you can
see which layer set the effective configuration.

### 3.4 Per-request override (tests / edge cases)

The policy API accepts a per-request override that takes precedence over every
configuration source:

```python
from scripts.kv_cache_policy import decide_tier, KVCacheTier

# Force GPU-resident even for a 200K context:
decision = decide_tier(200000, override="gpu")
# Force host-RAM offload even for a tiny context:
decision = decide_tier(1000, override=KVCacheTier.HOST_RAM_OFFLOAD)
```

Overrides are intended for **tests and edge cases**, not routine serving. See
§7 for how to force a mode in production launches.

### 3.5 CLI

```bash
# Ask the policy for a tier:
python scripts/kv_cache_policy.py --ctx-size 131073
# -> {"tier": "offload", "context_len": 131073, "threshold": 131072, ...}

# Force a tier for one invocation (testing/edge cases):
python scripts/kv_cache_policy.py --ctx-size 200000 --override gpu

# Launch-path gate: decide offload for a server argv (exit 0 = allowed):
P12_ALLOW_HUGE_CONTEXT_OFFLOAD=1 \
    python scripts/p12_offload_gate.py --ctx-size 262144 -- \
        -m /path/to/model.gguf -ngl 99 --no-kv-offload --ctx-size 262144
```

The gate CLI exits `0` when the request is **allowed** (policy-compliant) and
`1` when it is a policy violation. "Allowed but not enabled" — e.g. a
huge-context request with the opt-in flag unset — exits `0` with
`"enabled": false` in the JSON payload; the argv is neutralized to the fast
path either way. Read the JSON, not just the exit code.

Invalid configuration exits non-zero with a JSON `{"error": ...}` payload.

### 3.6 Example: lower threshold on a smaller GPU

```bash
# On a card where the GPU-resident KV cache only fits ~64K comfortably:
export KV_CACHE_OFFLOAD_THRESHOLD=65536   # 64K
export KV_CACHE_MODE=auto
# Contexts <= 64K: GPU fast path. Above 64K up to 256K: offload.
```

---

## 4. Operational guidance — memory budgeting and tuning the threshold

### 4.1 What actually fits where (sizing model)

The KV cache grows **linearly with context length** (layers × attention
dimension), independent of the weights. The Turbohaul sizing model
(`docs/KV_CACHE_OFFLOADING.md`) is:

```
kv_cache_mib ≈ (9 KB/token per GiB of body, at f16) × quant_factor × ctx_size
```

**KV resident in VRAM (default):** everything must fit one budget —
`weights + kv_cache + overhead` (overhead floor ≈ 1024 MiB) ≤ free VRAM.

**KV offloaded to host RAM:** the KV term drops out of the VRAM budget, but a
context-linear scratch term stays (on-GPU attention scratch still grows with
context), and the KV cache is re-checked against free host RAM:

```
vram_need  = weights + (overhead + ctx_size / 128)     # KV removed
refuse if vram_need > free_vram
refuse if kv_cache > free_host_ram
```

This is why a long-context config that is ~24 GiB "needed" with KV in VRAM
drops to ~17 GiB of VRAM with the KV offloaded — the multi-GiB KV cache moves
to host RAM, and the card goes from "won't fit" to "fits with headroom".

### 4.2 When to lower the threshold

Lower `KV_CACHE_OFFLOAD_THRESHOLD` when you are **VRAM-bound, RAM-rich**:

- Your card cannot hold `weights + KV + overhead` at the context lengths you
  actually serve (OOM during KV allocation, or `nvidia-smi` shows near-zero
  free VRAM).
- You need **concurrency**: N parallel slots need roughly N× the KV cache, and
  the multiplied cache is what blows the VRAM budget. Moving KV to RAM keeps
  the GPU holding only weights + a modest scratch, which does not multiply
  with slot count.
- Throughput/capacity matters more than minimum single-request latency.

### 4.3 When to raise the threshold

Raise it (toward the 256K max) when you are **latency-critical**:

- Every millisecond of per-token decode counts and your KV fits VRAM
  comfortably at the contexts you serve.
- You run one slot with no concurrency pressure.
- You prefer not to pay the PCIe-read cost on every decode step.

**Never set `KV_CACHE_MAX_CONTEXT` below `KV_CACHE_OFFLOAD_THRESHOLD`** — the
policy rejects that combination at load time (`PolicyConfigError`).

### 4.4 Reference numbers from the live validation (TinyLlama 1.1B Q4, 256K ctx, 4 slots)

From `scripts/p12_offload_live_check.py` and the t_14bd82aa evidence
summary — a real turboquant `llama-server` spawn at 262144 context:

| Metric | Offload OFF (fast path) | Offload ON (host-RAM KV) |
|---|---|---|
| Server RSS | **0.58 GB** | **6.35 GB** (≈5.7 GB KV cache in host RAM; model is 669 MB) |
| KV cache location | VRAM | host RAM (4 slots × 262144 ctx; prompt cache 8192 MiB) |
| Weights | GPU0 | GPU0 (never moved) |
| GPU1 | untouched | untouched (isolation preserved) |
| Chat round-trip | OK (`P12-OK`) | OK (`P12-OK`) |

Boot log confirmed CUDA0 visible (GPU0, 21022 MiB free), CUDA1 present but
never used by the main lane, and 128714 MiB free host RAM.

### 4.5 Latency trade-off (from Turbohaul `docs/KV_CACHE_OFFLOADING.md`)

Measured on a 35B-class sparse-MoE at `parallel: 2` on a 24 GB-class GPU:

| Metric | KV in VRAM | KV in host RAM |
|---|---|---|
| Decode throughput | **~112 tok/s** | **~44 tok/s** (~2.5× slower) |
| VRAM used | would not fit | ~21,903 MiB (weights + 2-slot scratch) |

Offloading buys **capacity, not speed**: the KV cache is read on every decode
step, so moving it to RAM puts PCIe bandwidth in the hot path. Decode is the
most affected (KV-read-bound); prefill is comparatively less sensitive; the
gap widens at longer contexts and higher parallelism. Sparse-MoE models
tolerate it better than dense models of equal total size. The right mental
model: **a VRAM-for-bandwidth trade** — reach for it when the alternative is
"won't run at all" or "can't serve in parallel".

---

## 5. Troubleshooting

### 5.1 OOM fallback behavior

The policy rejects out-of-range contexts **before allocation**, and the launch
path never silently offloads. Concretely:

- **Context > `KV_CACHE_MAX_CONTEXT` (256K default):** `ContextOutOfRangeError`
  / CLI exit 1 with `{"error": "context ... exceeds supported maximum ..."}`.
  There is no fallback — the request is refused. Lower the max or the request
  size; nothing at >256K will be served by this policy.
- **A huge-context request arrives with `P12_ALLOW_HUGE_CONTEXT_OFFLOAD`
  unset:** the policy may route it to offload, but the offload gate **refuses**
  and neutralizes the argv back to the fast path (`--no-kv-offload` stripped,
  weights pinned GPU0). The server boots in GPU-KV mode. If the GPU genuinely
  cannot fit the KV at that context, the server will OOM during KV allocation
  — that is a real capacity limit, and the gate's job was to prevent a
  *silent* offload, not to resize. Fix: set the opt-in flag (and check §4.1
  host-RAM fit), or lower the threshold/context.
- **Weights pinned to a non-GPU0 device (main-gpu ≠ 0, `--device` list with
  GPU1+, split-mode ≠ none):** hard policy violation — offload is refused,
  argv neutralized to the fast path. Weights are never offloaded (ADR 0012).
- **GPU fast path OOMs during allocation at an ordinary context:** the card
  is VRAM-bound below the threshold. Lower the threshold so more contexts
  route to offload, reduce the context, or reduce concurrency (`parallel`).

### 5.2 How to force a mode per request

- **Policy API (in-process):** pass `override="gpu"` or `override="offload"`
  to `decide_tier()` — wins over every config source.
- **CLI:** `python scripts/kv_cache_policy.py --ctx-size N --override gpu|offload`.
- **Server launch (offload):** set `P12_ALLOW_HUGE_CONTEXT_OFFLOAD=1` **and**
  route the request through the offload gate so `--no-kv-offload` is kept in
  the argv (or use the reserved `darwin-28b-256k` model tag — see
  `docs/design/p12-long-context-mode-switch-contract.md`). The gate still
  forces weights to GPU0 in every mode.
- **Server launch (force GPU-resident):** set `KV_CACHE_MODE=gpu` (or the
  per-request override) — the policy returns GPU-resident regardless of
  context length. This is for tests and edge cases; normal serving uses
  `auto`.

### 5.3 Config errors that fail fast (not silently)

| Symptom | Cause | Fix |
|---|---|---|
| `PolicyConfigError: invalid KV_CACHE_MODE ...` | mode is not `auto`/`gpu`/`offload` | correct `KV_CACHE_MODE` |
| `PolicyConfigError: KV_CACHE_OFFLOAD_THRESHOLD must be an integer` | non-integer value | use an int token count |
| `PolicyConfigError: KV_CACHE_OFFLOAD_THRESHOLD must be >= 1` | 0 or negative | use ≥ 1 |
| `PolicyConfigError: KV_CACHE_MAX_CONTEXT ... must be >= KV_CACHE_OFFLOAD_THRESHOLD` | max < threshold | order the two correctly |
| `PolicyConfigError: cannot read KV_CACHE_CONFIG file ...` | bad path / invalid JSON | fix the path or file |
| `PolicyConfigError: context_len must be >= 0` | negative context passed | validate before calling |

All of these are hard errors by design — a misconfiguration never silently
falls back to a different tier than the operator intended.

---

## 6. Architecture notes

- **Single source of truth:** `scripts/kv_cache_policy.py` owns the threshold,
  the max, and the tier decision. The offload gate
  (`scripts/p12_offload_gate.py`) consults `decide_tier()` and layers the
  opt-in flag + argv sanitization on top; it never duplicates the boundary.
- **Weights are never offloaded:** the gate forces `--split-mode none
  --main-gpu 0` in every mode, and a non-GPU0 weight placement is a hard
  policy violation (allowed=False, argv neutralized).
- **GPU1 isolation is preserved at all times:** offload targets CPU/system
  RAM, never GPU1. GPU1 belongs to the aux lane + media/speech.
- **The module is stdlib-only** so it runs as a pure decision function in
  tests, CI, and launch wrappers without importing torch or the heavy stack.

---

## 7. Related documents

- `~/brain/conventions/performance-rules.md` — the canonical **P12 performance rule** (main model fast for ordinary daily work; 256K required capability; near-256K jobs may run slower). This policy operationalises it.
- `docs/adr/0012-p12-256k-memory-policy.md` — the memory policy: weights
  GPU0-only, context/KV offload opt-in, GPU1 isolation.
- `docs/design/p12-long-context-mode-switch-contract.md` — the explicit
  long-context route (`darwin-28b-256k` tag), mode-switch lifecycle, edge
  cases, ownership.
- `scripts/kv_cache_policy.py` — tier decision source of truth.
- `scripts/p12_offload_gate.py` — launch-path gate + argv sanitization.
- `scripts/p12_offload_live_check.py` — live integration check.
- `scripts/p12_fast_path_verify.py` — live verification that **normal daily
  work stays on the fast path** (weights GPU0-only, no offload, GPU1
  isolated) and a performance baseline that catches regressions from the
  P12 changes. CI-safe assertions in
  `tests/scripts/test_p12_fast_path.py`; the live suite runs on the rig or
  a self-hosted GPU runner via `.github/workflows/p12-gpu-verify.yml`
  (manual dispatch).
- `tests/scripts/test_kv_cache_policy.py` (30 tests) and
  `tests/scripts/test_p12_offload_gate.py` (30 tests) — boundary and
  coordination coverage.
- `/srv/kensei-ai/turbohaul-manager/docs/KV_CACHE_OFFLOADING.md` — the
  `no_kv_offload` mechanism, sizing math, parallel serving, perf cost.
