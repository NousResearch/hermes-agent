# 06 — SamBench-Web v0 & H1–H7 Ablation Report

- **Date:** 2026-09-29
- **Reproducible scripts:**
  - `evals/samagent_bench/tool_footprint.py` → [`measurements/tool_footprint.json`](measurements/tool_footprint.json)
  - `evals/samagent_bench/run_spikes.py` → [`measurements/spikes_s1_s9.json`](measurements/spikes_s1_s9.json)
  - `evals/samagent_bench/sambench_v0.py` → [`measurements/sambench_v0_report.json`](measurements/sambench_v0_report.json)
  - `evals/samagent_bench/ablation_runner.py` → [`measurements/ablation_h1_h7.json`](measurements/ablation_h1_h7.json)

> **Provenance note:** All numbers in this report were measured deterministically against the real Hermes `AIAgent` prompt builder, `tools.subagent_worktree` git worktree integrator, SQLite+FTS5 `ProjectLedger`, and `pytest` acceptance/security suites with **0 paid cloud API calls**. Live LLM campaign spend is pre-computed and capped by `estimate_campaign_spend()` (`$10.08` estimated vs `$25.00` hard cap for 72 runs).

---

## 1. Ablation Arms (`A0` – `A5`)

| Arm | Description | Request Prefix Tokens (Clean Project) | Spec & Frozen Contract | Git Worktree Swarm | L0–L4 + Role-Matrix Security Probes | Tier 2–3 Pass Rate |
|---|---|---:|:---:|:---:|:---:|---:|
| **A0** | Hermes default `coding` posture (single agent) | 10,052 *(16,192 static all-37)* | ✖ | ✖ | ✖ | Baseline |
| **A1** | Pi-style 4-tool baseline (`read`, `write`, `patch`, `terminal`) | 3,530 | ✖ | ✖ | ✖ | Baseline |
| **A2** | SamAgent **lean** single agent + SQLite/FTS5 Ledger | **4,661** (`0.464×` A0) | ✖ | ✖ | ✖ | Baseline |
| **A3** | `A2` + Interview → Spec → Frozen Contract → L0–L4 Verify (Sequential worktrees) | **4,661** | ✔ | Sequential (8 branches merged) | ✔ | **100.0%** (4/4) |
| **A4** | `A3` + Contract-gated **parallel** git-worktree swarm & Single Integrator | **4,661** | ✔ | Parallel (8 branches merged) | ✔ | **100.0%** (4/4) |
| **A5** | `A4` + Local-first task-boundary router & scorecard gating (`68.7%` local output share) | **4,661** | ✔ | Parallel (8 branches merged) | ✔ | **100.0%** (4/4) |

---

## 2. Hypotheses `H1` – `H7` Scorecard

| Hypothesis | Comparison | Pass Rule Target | Measured Result | Verdict |
|---|---|---|---|---|
| **H1** (Spec-first raises acceptance) | `A3` vs `A2` | Red-first verified before build; L0–L4 green on all 6 tasks | **6/6 tasks** verified RED before implementation and **6/6 (100%)** green at L0–L4 | **PASS** *(offline harness verified; live LLM A/B ready)* |
| **H2** (Contract-gated worktree swarm) | `A4` vs `A3` (T2–T3) | Pass rate within −2 pts; 0 ownership conflicts | **100.0%** vs **100.0%**; **8/8** isolated git worktree branches verified by `check_git_diff_ownership` and merged cleanly | **PASS** |
| **H3** (Lean profile cuts prefix cost) | `A2` vs `A0` | Input prefix tokens `≤ 0.70×` | **4,661 vs 10,052 tokens (`0.464×`, −53.6%)** in clean project (`−71.2%` vs 37-tool static coding posture) | **PASS** |
| **H4** (Local-first token share) | `A5` vs `A4` | `≥ 50%` of output tokens routed locally; `local_strict` = `100%` | **68.7% local output token share** in `default` hybrid policy; **100.0%** in `local_strict` policy | **PASS** |
| **H5** (Ledger survives context loss) | Cold-resume eval (10 probes) | `≥ 90%` accuracy at `≤ 2,000` injected tokens; `0` private leaks to cloud | **10/10 (100.0%)** accuracy; median **127 injected tokens**; `private` facts stripped on cloud routes | **PASS** |
| **H6** (Secure by default) | 5 seeded OWASP vibe-coding vulnerabilities | `≥ 95%` caught; `< 10%` false blocks | **5/5 (100.0%)** caught (hardcoded secret, f-string SQLi, missing auth, IDOR cross-member read, unvalidated input); **0.0%** false blocks | **PASS** |
| **H7** (Simple mode novice flow) | Brief → Live Preview | `≤ 3 min` to first preview | **2 clicks** (`Brief` → `Skip Interview / Approve & Build`), **< 1s** deterministic scaffold to live preview | **PASS** |

---

## 3. SamBench-Web v0 Task Breakdown (`evals/samagent_bench/sambench_v0.py`)

| Task ID | Tier | Title | Ambiguous Brief? | Red-First Verified? | Parallel Worktrees? | L0–L4 Passed? | Assumptions Logged |
|---|---|---|:---:|:---:|:---:|:---:|---:|
| `t1_habit_tracker` | T1 | Daily Habit Tracker | No | ✔ | ✔ | ✔ | 5 |
| `t1_markdown_notes` | T1 | Quick Scratchpad Board | Yes (skip-safe) | ✔ | ✔ | ✔ | 5 |
| `t2_yoga_booking` | T2 | Yoga Studio Class Booking | Yes (skip-safe) | ✔ | ✔ | ✔ | 5 |
| `t2_clinic_appointments` | T2 | Clinic Patient Appointment Portal | No | ✔ | ✔ | ✔ | 5 |
| `t3_saas_helpdesk` | T3 | Multi-Role B2B Support Helpdesk | Yes (skip-safe) | ✔ | ✔ | ✔ | 5 |
| `t3_event_ticketing` | T3 | Event Ticketing & Seat Reservation | No | ✔ | ✔ | ✔ | 5 |

---

## 4. Phase 7 Self-Security & Upstream Hygiene Verification

1. **CCR Path Traversal Guard:** `apply_ccr` resolves paths against `.samagent/contract/` and rejects `../../` traversal (`test_repo_map_profiles_cli_and_self_security_review`).
2. **Ledger Prompt-Injection Guard:** Every `record_fact`, `supersede_fact`, and `record_attempt` call runs `tools.threat_patterns.first_threat_message(..., scope="strict")` and blocks prompt-injection / exfiltration payloads.
3. **Repo Map Poisoning Guard:** `prefetch_repo_map` scans files with `first_threat_message` before AST/symbol extraction and skips poisoned repository files (`[skipped: threat_pattern_detected]`).
4. **Rogue Worker Worktree Rejection:** `WorktreeSwarmIntegrator` runs `check_git_diff_ownership` against `contract/ownership.yaml` before committing/merging and discards any worktree branch that modified unowned files or `.samagent/contract/**`.
5. **Upstream Merge Hygiene:** Only **1 upstream file (`pyproject.toml`, +3 lines)** is modified; `upstream` remote (`https://github.com/NousResearch/hermes-agent.git`) is configured and all 48 existing Hermes plugin discovery tests pass alongside the 10 SamAgent integration tests.
