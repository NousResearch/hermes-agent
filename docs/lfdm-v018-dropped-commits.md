# LFDM v0.18 Dropped Commits Ledger

**Date:** 2026-07-07
**Context:** Merge 0.17.0 → v0.18.0 ("The Judgment Release", tag `v2026.7.1` = `7c1a029553`); this branch `lfdm-v018` = v0.18.0 + 16 kept/reconciled LFDM commits.
**Recoverability:** All 10 dropped commits remain permanently recoverable on branch `lfdm-main`, in the `pre-v018` git bundle, and on the fork `LFDMcore/hermes-agent`. This ledger makes the *decisions* explicit and revisitable.

Audit basis: per-commit keep/drop/reconcile vetted by a three-model triad (claude + deepseek-v4-pro + codex). Guiding principle: where v0.18 developers built a native superset, drop the LFDM workaround (developer-maintained, survives future updates); keep ours only for real gaps.

---

## Group 1: Discord Compaction-Spam Suppression (2 commits — SUPERSEDED)

| | |
|---|---|
| **`4690928dc5`** | `fix: suppress Discord compaction status spam` — `gateway/run.py`, `tests/gateway/test_telegram_noise_filter.py` |
| **`61bcb1209a`** | `fix: suppress context compaction artifacts in gateway chats` — same files |
| **Why dropped** | Superseded by 0.18 native commit `d2ea948bc0` ("fix(gateway): suppress compression status noise on Discord and other chats (#39293)") — handles compaction artifacts gateway-wide for all platforms, with tests. LFDM's Discord-specific logic is redundant. |
| **0.18 equivalent** | `d2ea948bc0` (gateway-level noise filter, already shipped in v0.18). |
| **Recover** | `git cherry-pick 4690928dc5 61bcb1209a` (from `lfdm-main`) — will conflict with 0.18's native suppression; resolve favoring 0.18 unless a gap is proven. |
| **Revisit if** | 0.18's native suppression proves insufficient in production (specific Discord workflows still leak compaction artifacts) — then layer LFDM tuning on top of 0.18's base. |

## Group 2: Discord Threading (2 commits — SUPERSEDED)

| | |
|---|---|
| **`c803117a97`** | `Re-enable Discord mention auto-threading in free channels` — `plugins/platforms/discord/adapter.py`, `tests/e2e/test_discord_adapter.py` |
| **`de0a5c25e3`** | `Honor Discord no-thread channels from config` — same files |
| **Why dropped** | 0.18's Discord adapter natively supports per-channel threading policy via `auto_thread` + `no_thread_channels` config flags. LFDM's custom logic is built into 0.18's base. |
| **0.18 equivalent** | `plugins/platforms/discord/adapter.py` in v0.18 (`no_thread_channels` / `auto_thread` config). |
| **Recover** | `git cherry-pick c803117a97 de0a5c25e3` (from `lfdm-main`) — will conflict; favor 0.18's native config. |
| **Revisit if** | 0.18's native threading flags don't cover a required threading semantic — then layer LFDM policy on top. |

## Group 3: Native Project Search Tools (2 commits — MERGE DEBT / plugin candidate)

| | |
|---|---|
| **`194e6b3d5f`** | `feat: native local git/project search tools (project_search, project_discover, project_read, project_map)` — `tools/project_search.py` (843+), `tests/tools/test_project_search.py` (493+), `hermes_cli/config.py`, `toolsets.py` (~926 lines) |
| **`ad7e02e622`** | `fix: harden project search safety edges` — `tools/project_search.py` (+117), tests (+46). Depends on `194e6b3d5f`. |
| **Why dropped** | ~1,350-line carrying cost for a feature not load-bearing for v0.18 and not yet wired into agent workflows. It is **complementary** to 0.18's new `tools/project_tools.py` (GUI workspace create/switch/list — a different tier), not superseded. Dropped from *core* to avoid per-update merge debt; re-home as a plugin if usage justifies. |
| **0.18 equivalent** | `tools/project_tools.py` (GUI workspace lifecycle only — does NOT provide agent-side content search). Orthogonal, not a replacement. |
| **Recover** | `git cherry-pick 194e6b3d5f ad7e02e622` (from `lfdm-main`) — cherry-picks clean (no v0.18 conflict). Then re-validate config-schema integration + wire toolset registration into v0.18's resolver. |
| **Revisit if** | Agent code-search usage justifies it (cross-repo symbol search, "find recent API changes", etc.). **Preferred re-adoption: as a standalone `plugins/project_search/` plugin**, not carried in core. |

## Group 4: Project-Continuity CI Gate (4 commits — rides with project_search)

| Commit | Subject |
|---|---|
| **`0ef8dfec90`** | `docs: add native project search continuity gate` (+ `scripts/verify_project_continuity.py`) |
| **`3b5d072c03`** | `fix: enforce Hermes continuity changed-path gate` |
| **`ffaf24bfbd`** | `fix: wire project continuity gate into CI` (`.github/workflows/lint.yml`) |
| **`6c5ba5815c`** | `fix: harden project continuity CI gate` |

| | |
|---|---|
| **Why dropped** | These exist **solely to gate `project_search`** (ensure it doesn't regress). With `project_search` dropped, the gate is orphaned. Also, `lint.yml` was rewritten in 0.18's CI overhaul, so re-applying would conflict anyway. |
| **0.18 equivalent** | None (0.18 doesn't ship `project_search`, needs no such gate). |
| **Recover** | Cherry-pick **all four in order** — `git cherry-pick 0ef8dfec90 3b5d072c03 ffaf24bfbd 6c5ba5815c` — **only** when re-adopting `project_search`; the CI wiring will need reconciling against 0.18's rewritten `.github/workflows/`. |
| **Revisit if** | `project_search` is re-adopted. No independent value outside that context. |

---

## Recovery quick-reference

```bash
# From branch lfdm-main (all carried commits live there); or restore the bundle first:
#   git clone <pre-v018 bundle> ; or: git fetch lfdm  (fork LFDMcore/hermes-agent)

# Path A — Discord fixes (only if 0.18 natives prove insufficient; will conflict):
git cherry-pick 4690928dc5 61bcb1209a          # compaction spam
git cherry-pick c803117a97 de0a5c25e3          # threading

# Path B — Project search (clean cherry-pick; prefer re-homing as a plugin):
git cherry-pick 194e6b3d5f ad7e02e622          # tools + hardening (in order)
git cherry-pick 0ef8dfec90 3b5d072c03 ffaf24bfbd 6c5ba5815c   # continuity gate (only with project_search, in order)
```

**Rules:** never cherry-pick the continuity gate without `project_search`; always pick `ad7e02e622` right after `194e6b3d5f`; Discord fixes are independent (both-or-neither per platform).

## Dropped for correctness (not lost)
Commit hashes and file lists above are immutable. Branches/merges may shift, but `lfdm-main`, the `pre-v018` git bundle, and the fork preserve all history indefinitely. This is the canonical reference for what was dropped in the v0.18 merge and how to bring any of it back.
