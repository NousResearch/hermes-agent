# Audit: Registry Main+4 Pool-Then-Next (3 tries) — 2026-09-03
**Scope:** `route-registry/registry/surfaces.yaml` + `route-registry/registry/route-slots.yaml` (Sept 2 rebuild). Hard read-only; no edits.  
**Prior-audit note:** Earlier audits used stale `agents/*/config.yaml`. This audit uses the canonical registry files only; prior profile tables are discarded. Preserved ID resolutions from prior context: lead `openai-codex/gpt-5.6-sol`, worker `openai-codex/gpt-5.6-luna`, local final `custom:turbohaul-local/qwen3.8-27b` via `127.0.0.1:11410` relay → manager on `11401`.  
**User locks:** 3 tries per slot before moving to next slot.

## 1. Source Summary
- **Slots:** 32 live slot entries in `route-slots.yaml`.
- **Surfaces:** 64 surfaces in `surfaces.yaml`.
- **Canonical final:** `local_final` block defines mandatory terminal slot: `custom:turbohaul-local / qwen3.8-27b / http://127.0.0.1:11410/v1`.
- **Codex fallback block:** `codex_fallback` defines `openai-codex / gpt-5.6-sol` and `openai-codex / gpt-5.6-luna`.

## 2. Slot Health Flags
| Flag type | Count | Slots |
|-----------|-------|-------|
| Retired | 3 | `slot-mini-m3free-1`, `slot-mini-m3-1`, `slot-qwen27b-1` |
| Disabled | 13 | `slot-mini-m3free-1/2/3`, `slot-mini-m3-2/3`, `slot-step37free-1/3`, `slot-qwen27b-2`, `slot-dspro-3`, `slot-kimik3-2`, `slot-nous-emergency-credit` |
| Candidate but not approved | 7 | `slot-mini-m3free-2/3`, `slot-mini-m3-2/3`, `slot-step37free-1/3`, `slot-qwen27b-2`, `slot-dspro-3`, `slot-kimik3-2` |
| Pool-then-next only | Multiple `ollama-cloud` and `custom:bai` groups defined under `pool_requirements` |

**Important:** `slot-mini-m3free-1`, `slot-mini-m3-1`, and `slot-qwen27b-1` are explicitly retired with `disabled: true` and `retire_reason` set.

## 3. Codex Constraint Check
Rule in `model_routing_constraints.codex_isolation`: `gpt-5.6-*` models may **only** be served by `openai-codex`.  
- Registry contains **no** `gpt-5.6-*` slots in the 32 canonical slots. Good — no codex constraint violation inside `route-slots.yaml`.
- Codex fallback block (`codex_fallback`) is **not** attached to any surface in `surfaces.yaml`. So codex is currently unreachable through registry surfaces; only resolvable through legacy/config paths.

## 4. Missing Pieces
1. **No registry surface currently appends codex or local final as explicit slots.** Every surface uses only the slots listed in `surfaces.yaml`. The `codex_fallback` and `local_final` blocks exist in `route-slots.yaml` but are not wired into surface slot lists.
2. **`default`, `deizy`, `gojo`, `kensei`, `work` surfaces** declare `routes_policy: keep_current` and have empty slot lists, meaning they do not participate in this Main+4 pool.

## 5. Main+4 Semantics with 3 Tries Per Slot
For each surface with a non-empty slot list, order is **Main-first slot** (the first listed slot after `main_model` target) then subsequent slots in declared order. Each slot gets up to 3 attempts before advancing to the next slot. After all registry slots are exhausted, the chain must terminate at `local_final` (`custom:turbohaul-local/qwen3.8-27b` via `127.0.0.1:11410`).

Expanded try syntax example for a 2-slot surface:
```
1. Slot A try 1
2. Slot A try 2
3. Slot A try 3
4. Slot B try 1
5. Slot B try 2
6. Slot B try 3
7. local_final try 1  # mandatory terminal, 1 try unless registry says otherwise
```

For surfaces with more than 2 slots, same pattern repeats per slot.

## 6. Per-Surface Chains (Main + slots in order; 3 tries each)

Legend:
- `A` = approved/active, `C` = candidate/disabled, `R` = retired/disabled
- Provider abbreviations: `xkiro-free` = `custom:xkiro-free`, `bai` = `custom:bai`, `nous` = `nous`, `ollama` = `ollama-cloud`, `cc` = `custom:commandcode`, `local` = `custom:turbohaul-local`
- ✅ = fully active, ⚠️ = candidate/disabled, ❌ = retired/disabled

### Tier SOL
**ceecee** — main `minimax/minimax-m3-free` (surface main_model, no explicit slot)
- Slot chain: `slot-mini-m3free-1 [R❌ xkiro-free]` → `slot-mini-m3free-2 [C⚠️ nous disabled]` → `slot-mini-m3free-3 [C⚠️ cc disabled]` → `local_final`
- Flags: All candidate slots disabled; no approved slot for `minimax-m3-free`. Codex not on this surface.
- Verdict: **NO** — `minimax-m3-free` is effectively unservable from registry; recommend remap to `minimax-m3` via `slot-mini-m3-4`.

**default** — `routes_policy: keep_current`, no slots. Chain: keep current; **N/A** from registry.

**deizy** — `routes_policy: keep_current`, no slots. Chain: keep current; **N/A** from registry.

**gojo** — `routes_policy: keep_current`, no slots. Chain: keep current; **N/A** from registry.

**kensei** — `routes_policy: keep_current`, `main_model: null`. **N/A**.

**kensei-review** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**light** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Flags: Both slots disabled; no approved slot for `step-3.7-flash:free`.
- Verdict: **NO** — no active slot; remap needed or enable approved candidate.

**moss** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [olloma]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**mrhermagi** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**octacon** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**orchestrator** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Flags: Same as `light`; no active slot.
- Verdict: **NO** — remap or enable candidate.

**quan** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**remii** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**sirvir** — `qwen3.8-27b`
- Slot chain: `slot-qwen27b-1 [R❌ ollama retired]` → `slot-qwen27b-2 [C⚠️ cc disabled]` → `slot-qwen27b-3 [bai]` → `local_final`
- Flags: Slot 1 retired; slot 2 disabled but approved local final keeps chain viable.
- Verdict: **YES**, with caveat — first slot dead.

**wesker** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

### Tier LUNA
**ceecee-brand** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**ceecee-reviewer** — `deepseek-v4-flash`
- Same chain as above. Verdict: **YES**

**ceecee-seo** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**ceecee-social** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**ceecee-writer** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**content-strategist** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**deizy-brand** — `minimax/minimax-m3-free`
- Slot chain: `slot-mini-m3free-1 [R❌ xkiro-free]` → `slot-mini-m3free-2 [C⚠️ nous disabled]` → `slot-mini-m3free-3 [C⚠️ cc disabled]` → `local_final`
- Verdict: **NO** — same issue as `ceecee`.

**deizy-component-lib** — `minimax-m3`
- Slot chain: `slot-mini-m3-1 [R❌ xkiro-free]` → `slot-mini-m3-2 [C⚠️ nous disabled]` → `slot-mini-m3-3 [C⚠️ cc disabled]` → `slot-mini-m3-4 [bai]` → `local_final`
- Verdict: **YES** — slot 4 saves it.

**deizy-design-system** — `minimax-m3`
- Same chain. Verdict: **YES**

**deizy-image-prompt** — `minimax-m3`
- Same chain. Verdict: **YES**

**deizy-ux-architect** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**deizy-ux-prototype** — `minimax-m3`
- Same chain as `deizy-component-lib`. Verdict: **YES**

**denji-ledger** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Verdict: **NO** — no active slot.

**denji-monitor** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**denji-reviewer** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**denji-skill** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**gojo-admin** — `gemma4:31b`
- No slots; `routes_policy: keep_current`. **N/A**.

**gojo-calendar** — `gemma4:31b`
- No slots; **N/A**.

**gojo-mailbox** — `gemma4:31b`
- No slots; **N/A**.

**light-archivist** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**light-indexer** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**light-wiki** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**market-scanner** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Verdict: **NO** — same as `light`.

**misa-misa** — `minimax-m3`
- Slot chain: `slot-mini-m3-1 [R❌ xkiro-free]` → `slot-mini-m3-2 [C⚠️ nous disabled]` → `slot-mini-m3-3 [C⚠️ cc disabled]` → `slot-mini-m3-4 [bai]` → `local_final`
- Verdict: **YES**

**octacon-architect** — `glm-5.3-flash`
- Slot chain: `slot-glm53flash-1 [ollama]` → `slot-glm53flash-2 [ollama]` → `slot-glm53flash-3 [bai]` → `slot-glm53flash-4 [bai]` → `local_final`
- Verdict: **YES**

**octacon-backend** — `glm-5.3-flash`
- Same chain. Verdict: **YES**

**octacon-frontend** — `kimi-k3`
- Slot chain: `slot-kimik3-1 [bai]` → `slot-kimik3-2 [C⚠️ cc disabled]` → `slot-kimik3-3 [ollama]` → `local_final`
- Verdict: **YES**

**octacon-infra** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**octacon-mobile** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**octacon-techwriter** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**octacon-testrunner** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**quan-arch** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**quan-code** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**quan-e2e** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**quan-perf** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**quan-security** — `deepseek-v4-flash`
- Same chain. Verdict: **YES**

**quan-ux** — `minimax-m3`
- Slot chain: `slot-mini-m3-1 [R❌ xkiro-free]` → `slot-mini-m3-2 [C⚠️ nous disabled]` → `slot-mini-m3-3 [C⚠️ cc disabled]` → `slot-mini-m3-4 [bai]` → `local_final`
- Verdict: **YES**

**remii-deep** — `deepseek-v4-pro`
- Slot chain: `slot-dspro-1 [xkiro-pro]` → `slot-dspro-2 [bai]` → `slot-dspro-3 [C⚠️ cc disabled]` → `slot-dspro-4 [ollama]` → `slot-dspro-5 [ollama]` → `local_final`
- Verdict: **YES**

**remii-digest** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**remii-gitradar** — `deepseek-v4-pro`
- Slot chain: `slot-dspro-1 [xkiro-pro]` → `slot-dspro-2 [bai]` → `slot-dspro-3 [C⚠️ cc disabled]` → `slot-dspro-4 [ollama]` → `slot-dspro-5 [ollama]` → `local_final`
- Verdict: **YES**

**remii-market** — `deepseek-v4-pro`
- Same chain as `remii-deep`. Verdict: **YES**

**skill-broker** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Verdict: **NO** — same as `light`.

**skill-research** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**triage-router** — `stepfun/step-3.7-flash:free`
- Slot chain: `slot-step37free-1 [C⚠️ nous disabled]` → `slot-step37free-3 [C⚠️ cc disabled]` → `local_final`
- Verdict: **NO**

**wesker-backup** — `gemma4:31b`
- No slots; **N/A**.

**wesker-ops** — `gemma4:31b`
- No slots; **N/A**.

**wesker-scanner** — `deepseek-v4-flash`
- Slot chain: `slot-dsflash-1 [xkiro-free]` → `slot-dsflash-2 [bai]` → `slot-dsflash-3 [bai]` → `slot-dsflash-4 [ollama]` → `slot-dsflash-5 [ollama]` → `local_final`
- Verdict: **YES**

**work** — `main_model: null`, `routes_policy: keep_current`. **N/A**.

## 7. Sahil Yes/No Summary
| Surface | Sahil Decision | Reason |
|---------|---------------|--------|
| ceecee | **NO** | `minimax-m3-free` has no active slot |
| ceecee-brand | **YES** | healthy `deepseek-v4-flash` chain |
| ceecee-reviewer | **YES** | healthy |
| ceecee-seo | **YES** | healthy |
| ceecee-social | **YES** | healthy |
| ceecee-writer | **YES** | healthy |
| content-strategist | **YES** | healthy |
| default | **N/A** | keep_current |
| deizy | **N/A** | keep_current |
| deizy-brand | **NO** | `minimax-m3-free` chain dead |
| deizy-component-lib | **YES** | slot 4 active |
| deizy-design-system | **YES** | slot 4 active |
| deizy-image-prompt | **YES** | slot 4 active |
| deizy-ux-architect | **YES** | healthy `glm-5.3-flash` |
| deizy-ux-prototype | **YES** | slot 4 active |
| gojo | **N/A** | keep_current |
| gojo-admin | **N/A** | keep_current |
| gojo-calendar | **N/A** | keep_current |
| gojo-mailbox | **N/A** | keep_current |
| kensei | **N/A** | keep_current |
| kensei-review | **YES** | healthy `glm-5.3-flash` |
| light | **NO** | `step-3.7-flash:free` slots all disabled |
| light-archivist | **YES** | healthy |
| light-indexer | **YES** | healthy |
| light-wiki | **YES** | healthy |
| market-scanner | **NO** | `step-3.7-flash:free` slots all disabled |
| misa-misa | **YES** | slot 4 active |
| moss | **YES** | healthy `glm-5.3-flash` |
| mrhermagi | **YES** | healthy |
| octacon | **YES** | healthy |
| octacon-architect | **YES** | healthy |
| octacon-backend | **YES** | healthy |
| octacon-frontend | **YES** | healthy |
| octacon-infra | **YES** | healthy |
| octacon-mobile | **YES** | healthy |
| octacon-techwriter | **YES** | healthy |
| octacon-testrunner | **YES** | healthy |
| orchestrator | **NO** | `step-3.7-flash:free` slots all disabled |
| quan | **YES** | healthy |
| quan-arch | **YES** | healthy |
| quan-code | **YES** | healthy |
| quan-e2e | **YES** | healthy |
| quan-perf | **YES** | healthy |
| quan-security | **YES** | healthy |
| quan-ux | **YES** | slot 4 active |
| remii | **YES** | healthy |
| remii-deep | **YES** | healthy |
| remii-digest | **YES** | healthy |
| remii-gitradar | **YES** | healthy |
| remii-market | **YES** | healthy |
| sirvir | **YES** | slot 3 active saves it |
| skill-broker | **NO** | `step-3.7-flash:free` slots all disabled |
| skill-research | **YES** | healthy |
| triage-router | **NO** | `step-3.7-flash:free` slots all disabled |
| wesker | **YES** | healthy |
| wesker-backup | **N/A** | keep_current |
| wesker-ops | **N/A** | keep_current |
| wesker-scanner | **YES** | healthy |
| work | **N/A** | keep_current |

## 8. Pool-Then-Next Ramifications & Proposed Improvement
**Current situation:** For models with multiple active slots (e.g., `deepseek-v4-flash`), all candidates live in the same `fill_first` pool group. With pool-then-next, the resolver:
1. Exhausts 3 tries on slot 1.
2. Moves to slot 2 for 3 tries.
3. Etc.

This is **sequential slot exhaustion**, not true pool parallelism. In practice:
- A transient error on slot 1 burns 3 attempts before any diversity.
- If slots share provider families (`xkiro-free` and `b.ai` both carry `deepseek-v4-flash`), they are treated as separate independent slots rather than load-balanced sub-accounts.
- Under high retry rate, latency multiplies because bad slots are tried to exhaustion before better slots are probed.

**Better approach:** Instead of pure 3x sequential per slot, use a **weighted pool sweep**:
- Treat all active slots for the same `model_id` as a single logical pool with 3 shared attempts across the group.
- Order slots by `reasoning_effort`/latency preference, but on failure rotate to next slot in the same pool without exhausting all 3 tries on the failing slot first.
- Reserve strict `3x per slot` only when the failure mode is slot-specific auth/402/403 rather than transient 5xx/timeout.

For 3 slots `[A, B, C]` with strict user lock, this expands to:
```
A1, B1, C1, A2, B2, C2, A3, B3, C3, local_final
```
instead of
```
A1, A2, A3, B1, B2, B3, C1, C2, C3, local_final
```
**Recommendation:** If Sahil wants the strict "3 tries per slot" contract, keep current ordering but enable parallel slot probing on first failure to reduce latency. If flexibility is acceptable, propose the interleaved pool sweep above.

---

*Generated 2026-09-03. Hard read-only from `route-registry/registry/surfaces.yaml` and `route-registry/registry/route-slots.yaml`.*
