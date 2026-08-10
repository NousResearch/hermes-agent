# Phase 8 — Final Activation Packet
**Date:** 2026-08-09
**Branch converged:** `audit/fix-full-suite-collection-20260809` → canonical `main` at `a2b5b03b6f`
**Status:** PENDING APPROVAL — no service restart, cron change, DB action, or live inference performed

---

## 0. TL;DR

The merged code fixes ~30 defects (CI was red → full suite green: 28,136 tests, 0 failed) and adds the P12 launch seam. **Nothing activates by itself.** The running services still execute the OLD code from memory. Activation = restarting services so they load the merged code. That's the only real action, and it's your call.

---

## 1. What is already DONE (no action needed)

| Item | State |
|---|---|
| Canonical main updated | `a2b5b03b6f` (81 commits: fixes + P12 seam) |
| Full CI green | 28,136 passed, 0 failed, exit 0 (post-merge) |
| Content DB authority | Verified coherent; no reconciliation needed |
| P12 launch seam | Wired into `turbohaul-main.service` unit file (in-repo), proven by tests, NOT loaded |

## 2. What activation WOULD change (the only real items)

### A. Gateway services (14 running)
All gateways run from `/home/kensei/repos/KenseiAgent/.venv/bin/python`. They currently hold the OLD code in memory. A restart picks up the merged fixes:

- `hermes-gateway.service` (main, running)
- `hermes-gateway-sirvir.service` (running)
- 12 specialist gateways (ceecee, denji, dezzy, gojo, kensei-review, light, misa-misa, mrhermagi, octacon, quan, remii, wesker) — all running

**Impact of restart:** picks up fixes to voice handling, TTS, STT empty-transcript guard, cron model pins, shutdown notifications, fallback auth exclusion, config accessors. Brief downtime per gateway (seconds). Risk: LOW (all covered by tests).

### B. `sirvir-gpu-watchdog.service` (running)
Runs `scripts/sirvir_turbohaul_observer.py` from the repo — restart picks up any observer changes. Low risk.

### C. `turbohaul-main.service` / `turbohaul-aux.service` — **DISABLED, not loaded**
- The P12 seam changes `turbohaul-main.service` ExecStart to route through `scripts/p12_launch_main.py`.
- **Neither service is currently enabled.** Enabling/starting turbohaul-main would START a live llama-server (Darwin 28B on GPU0) — this is the "live inference" item that needs explicit sign-off.
- Turbohaul-manager itself runs from `/opt/venv` (separate install) and is NOT changed by this merge.

### D. Hermes cron jobs
No cron definitions were changed by the merge. Crons will use the new code on their next natural run (no action needed). If you want crons to pick up new code immediately, they'd need a gateway restart (item A) since they run in the gateway process.

### E. Content DB files (2 zero-byte strays)
- `content_engine/db/content.db` (0B)
- `content_engine/content_engine.db` (0B, wrong path)
- **Not deleted.** Optional cleanup, only with your approval. Harmless either way.

---

## 3. What I will NOT do without explicit approval

- [ ] Restart any of the 14 gateway services (item A)
- [ ] Restart `sirvir-gpu-watchdog` (item B)
- [ ] Enable/start `turbohaul-main.service` (item C — live inference)
- [ ] Delete the 2 zero-byte content DB strays (item E)
- [ ] Any cron modification

## 4. Recommended activation sequence (if you approve)

1. Restart the 14 gateways (one at a time or in small batches, watch logs).
2. Restart `sirvir-gpu-watchdog`.
3. Verify each comes up clean (`systemctl status`, logs).
4. Leave turbohaul-main disabled unless you want the P12 fast path live — that's a separate decision (item C).
5. Optionally delete the 2 stray DB files.

## 5. What I need from you

A simple: **"activate A+B"**, **"activate A+B+C"**, or **"activate none — keep it as-is"**. That's the whole decision. Everything else is prepared and waiting.

---

## 6. EXECUTION RECORD — 2026-08-09 (Option 2 approved by Sahil)

**A+B: COMPLETE**
- Restarted all 14 gateway services (main, sirvir, 12 specialists) — all active, running merged code at `a2b5b03b6f`.
- Restarted `sirvir-gpu-watchdog` — active.
- Verified each via `systemctl is-active` + `ActiveEnterTimestamp` (all show today's restart).

**C: REVISED — NOT the dead systemd unit**
- Sahil corrected the model picture: the live stack uses **GRM 2.6 (main lane, port 11500)** and **Carwin MoE Nano (aux, GPU1, 256K)** via Turbohaul-Manager (`/opt/venv`, state in `/srv/kensei-ai/turbohaul-manager/`).
- `turbohaul-main.service` (the unit my P12 seam wired) is LEGACY — references Darwin-28B which no longer exists on disk. Left disabled. Correct.
- Turbohaul-Manager is the real inference control plane: healthy (PID 17038, uptime 2d9h), owns manifests (grm-2.6, carwin-moe, prism-27b-dq, etc.), enforces KV-cache fit via its own safety module.
- Carwin manifest (readable) confirms P12-consistent isolation: `main_gpu: 1`, `split_mode: none`, 256K ctx. GRM manifest is root-owned (manager config) — verified via engine logs it serves on 11500.
- No live inference was started by me; the manager was already the controller and remains so. The P12 repo seam (scripts/p12_launch_main.py) is committed as the reference launch wrapper for future manifest wiring.

**E: NOT deleted** — the 2 zero-byte stray DBs remain (per evidence-first rule, no approval given).

**Outstanding (optional, needs separate decision):**
- Whether Turbohaul-Manager's manifest-launch path should consume the repo's `p12_launch_main.py` gate directly (design task, not activation).
- Deleting the 2 stray 0-byte DB files.

## 7. FOLLOW-UP 1 — P12 gate ↔ Turbohaul-Manager (ASSESSED: no code change)

Wiring the repo's `p12_launch_main.py` into Turbohaul-Manager's spawn path was
evaluated and REJECTED as duplicate enforcement:

- The manager's `manifest.py` has a closed `SAFE_LLAMA_FLAGS` allowlist
  covering every P12-relevant flag: `main_gpu`, `split_mode`, `n_gpu_layers`,
  `ctx_size`, `kv_offload`, `no_kv_offload` — with bounds + enum validation.
- `flags_to_argv()` re-enforces the allowlist at argv-build (defense-in-depth).
- The isolation contract is expressed in manifests the manager owns:
  `carwin-moe.yaml` pins `main_gpu: 1` + `split_mode: none` (aux/GPU1);
  GRM 2.6 (main lane, port 11500) is root-managed.
- The manager has its own KV-cache fit safety (`check_kv_cache_fit`,
  `estimate_kv_cache_mib`) + nvidia-smi VRAM verify at spawn.
- Bolting the repo wrapper in would fight the manager's flag construction and
  risk a healthy serving stack. Extend-don't-duplicate applies.

Repo-side `p12_launch_main.py` remains valuable as: (a) the CI/reference
launch contract, (b) a pre-flight checker for future manifest authoring,
(c) documentation of the fast/long-context argv shape.

Recorded gap (not a defect): the manager allowlist PERMITS `split_mode:
layer/row/tensor` and `main_gpu: N>0` in any manifest; P12's main-lane
GPU0-only rule is enforced by manifest content (root-managed), not by a
fail-fast in manager code. If Sahil wants code-level enforcement of "main
lane must be GPU0", that's a Turbohaul-Manager feature request (its repo,
its PR) — flagged, not acted on.

## 8. FOLLOW-UP 2 — stray DB cleanup (DONE)

Deleted (both 0 bytes, unreferenced by production/cron, confirmed before
removal):
- `content_engine/content_engine.db`
- `content_engine/db/content.db`

Live authority DB `content_engine/db/content_engine.db` (1.4MB, 1098 drafts)
verified intact; `content_engine.pre_purge_20260512_130551.db` backup kept.
