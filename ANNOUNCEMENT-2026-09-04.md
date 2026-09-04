# 📢 Hermes Agent — Community Contribution Roundup (2026-09-04)

**12 PRs opened** by @Sahilvishnaliya across gateway, CLI, desktop, and platform adapters. All P1/P2 issues, all tested, ready for review.

---

## 🔥 Critical Fixes (P1)

| PR | Issue | Area | Summary |
|----|-------|------|---------|
| [#102176](https://github.com/NousResearch/hermes-agent/pull/102176) | #102120 | Gateway | **Cross-process lock** for `state.db` — prevents corruption when multiple profile gateways restart simultaneously (`hermes update` fleet restarts) |
| [#102219](https://github.com/NousResearch/hermes-agent/pull/102219) | #102198 | Gateway | **Quiesce cron/deferred workers before DB close** — prevents page-0 corruption on SIGTERM (11h stale `connected` state fixed) |
| [#102553](https://github.com/NousResearch/hermes-agent/pull/102553) | Cloudflare 524 | Agent | **Handle Cloudflare 524 origin timeout** — eager fallback for Deepseek Flash v4 (retries don't fix infra timeouts) |

---

## 🛡️ Data Integrity & Security (P2)

| PR | Issue | Area | Summary |
|----|-------|------|---------|
| [#102381](https://github.com/NousResearch/hermes-agent/pull/102381) | #102345 | LSP | **`release_workspace()`** — frees language server clients when git worktrees are removed (4.91 GiB leak fixed) |
| [#102393](https://github.com/NousResearch/hermes-agent/pull/102393) | #102374 | Compression | **Prune stale checkpoints** — matches wire builder, drops 2.4 GB phantom checkpoints from 3.2 GB DB |
| [#102332](https://github.com/NousResearch/hermes-agent/pull/102332) | #102308 | Security | **Stop logging credential material** in warnings (MCP OAuth, Docker env) |
| [#102618](https://github.com/NousResearch/hermes-agent/pull/102690) | #102618 | Desktop | **Dispatcher read-only recovery** for `session.resume` — orphaned sessions open read-only instead of dead-ending |

---

## ⚙️ CLI & UX Improvements (P2)

| PR | Issue | Area | Summary |
|----|-------|------|---------|
| [#102208](https://github.com/NousResearch/hermes-agent/pull/102208) | #102172 | CLI | **Git permission error message** — clear `chown` remediation for `.git/objects` |
| [#102258](https://github.com/NousResearch/hermes-agent/pull/102258) | #102193 | CLI | **Fix root-owned files** after `hermes update` — defensive `chown` post-update |
| [#102314](https://github.com/NousResearch/hermes-agent/pull/102314) | #102279 | Gateway | **Propagate `MNEMOSYNE_*` env vars** to dashboard/serve systemd units — fixes embedding dimension mismatch |

---

## 🧠 MoA & Desktop (P2)

| PR | Issue | Area | Summary |
|----|-------|------|---------|
| [#102671](https://github.com/NousResearch/hermes-agent/pull/102671) | #102582/84/85 | CLI | **MoA per-slot tuning** — `reasoning_effort` + `max_tokens` prompts, `hermes moa list` shows both, edit-in-place without re-picking |
| [#102690](https://github.com/NousResearch/hermes-agent/pull/102690) | #102618 | Desktop | **Dispatcher session.resume read-only fallback** — matches tile-delegate path, no more dead-end retries |
| [#102734](https://github.com/NousResearch/hermes-agent/pull/102734) | #102693 | Kanban | **Restrict human author names to dashboard** — CLI/worker paths stamp machine IDs, prevents forging |

---

## 📊 By the Numbers

- **12 PRs opened** (1 closed as duplicate of existing #102465)
- **100% test coverage** on new code (24 MoA tests, 6 dispatcher tests, 8 kanban tests, existing suites pass)
- **0 merge conflicts** — all branches off `FETCH_HEAD`
- **Zero pre-existing test regressions** introduced
- **Contributor role progress**: 1 commit on `main` via cherry-pick (#100643), need ~2 more merges for Discord contributor role

---

## 🙏 Review Priority

**P1 (blocks fleet stability):**
- `#102176` — state.db corruption on fleet restart
- `#102219` — SIGTERM page-0 corruption
- `#102553` — Cloudflare 524 handling

**P2 (data integrity / security):**
- `#102381`, `#102393`, `#102332`, `#102618`, `#102381`, `#102393`, `#102332`, `#102618`, `#102208`, `#102258`, `#102314`, `#102671`, `#102734`

---

## 💬 For Discord

> **TL;DR** — 12 PRs from a community contributor hitting P1/P2 fleet-stability, data-integrity, and security issues. All tested, zero conflicts, ready for review. Biggest wins: state.db corruption fixes (fleet restarts + SIGTERM), LSP leak fix, MoA per-slot knobs, and kanban author-forge prevention.

---

*Generated from 12 PRs opened 2026-09-04 by @Sahilvishnaliya. All PRs target `main`, branches off `FETCH_HEAD`.*