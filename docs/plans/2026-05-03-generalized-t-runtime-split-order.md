# Generalized T-Trading Runtime — Optimal Subsequent Split Order

> **For Hermes:** Use subagent-driven-development to implement this plan phase-by-phase.

**Goal:** Evolve `hermes_t` from a 2-file module into a generalized, reusable T-trading runtime that can eventually be migrated to any symbol, with independent PRs at each phase.

**Current repo baseline (2026-05-03):**
- `hermes_t/` exists in repo: `__init__.py` + `post_market_review.py`
- `hermes_olin/` does NOT exist in repo (it's OpenClaw-side only)
- `pyproject.toml` already includes `hermes_t` / `hermes_t.*` in package discovery
- PR #19148 is OPEN (standalone post-market review module)
- 7 tests in `tests/test_post_market_review.py`, all passing

---

## Phase 1: Add TradingStateStore + shared CLI builder (next PR)

**Objective:** Grow `hermes_t` into a usable facade with a real store and shared CLI helper, without disturbing the post-market review module.

### Task P1a: Create `hermes_t/store.py` — TradingStateStore

Create: `hermes_t/store.py`

A directory-based state store that reads/writes:
- `execution_state.json`
- `pending_signal.json`
- `position.json`
- `dispatch_ledger.jsonl`
- `signal_send_history.jsonl`
- `push_state.json`
- `active_signal.json`

**Key design:** Profile-scoped paths (`base_dir / profile_id`). Not Olin-specific.

Steps:
1. Write failing test: `test_store_creates_base_dir_on_first_write`
2. Write failing test: `test_store_load_or_default_uses_empty_state`
3. Write minimal `TradingStateStore` implementation
4. Write failing test: `test_store_profile_scoped_path_isolation`
5. Write failing test: `test_store_save_and_load_pending_signal`
6. Write failing test: `test_store_append_jsonl`
7. Implement GREEN for all
8. Run: `uv run python3 -m pytest tests/test_trading_runtime.py -k 'store' -q`
9. Commit: `git commit -m "feat: add TradingStateStore with profile-scoped paths"`

### Task P1b: Add shared CLI builder — `hermes_t/cli_shared.py`

Create: `hermes_t/cli_shared.py`

Extract:
- `build_runtime_parser()` — shared argparse builder
- `build_runtime_profile_from_args(args)` — validated single-profile builder
- `build_runtime_store(base_dir, profile, prefer_legacy_olin_store=False)`

Steps:
1. Write failing test: `test_shared_parser_has_default_base_dir_at_home`
2. Write failing test: `test_shared_parser_env_var_fallback`
3. Write failing test: `test_build_runtime_profile_from_args_rejects_blank_required_strings`
4. Write failing test: `test_build_runtime_profile_from_args_rejects_non_positive_ints`
5. Write failing test: `test_build_runtime_store_creates_TradingStateStore_by_default`
6. Implement GREEN for all
7. Commit

**Do NOT touch `hermes_olin.__main__` in this phase** — that's legacy compat.

### Task P1c: Update `hermes_t/__main__.py` for single-profile CLI

Modify `hermes_t/__main__.py` to use shared CLI builder and implement `python -m hermes_t --signal sell --score 20` single-profile path.

Steps:
1. Write failing test: `test_hermes_t_cli_single_profile_routes_through_shared_builder`
2. Implement: `hermes_t/__main__.py` wired to `cli_shared.build_runtime_parser()`
3. GREEN + full regression
4. Commit

**Test regression command:** `uv run python3 -m pytest tests/test_trading_runtime.py tests/test_post_market_review.py -q`

---

## Phase 2: Runtime cycle skeleton (independent PR)

**Objective:** Add the core `run_runtime_cycle()` skeleton to `hermes_t`, generic + profile-aware.

### Task P2a: Create `hermes_t/runtime.py`

Create: `hermes_t/runtime.py`

Minimal generic runtime that:
- Reads state via `TradingStateStore`
- Takes `tech_data` as input dict (not quote/realtime)
- Produces `{pending, suggestion, summary}` output

**Key constraint:** `signal_policy` is pluggable — use `DEFAULT_SIGNAL_POLICY` dataclass from `hermes_t/signal_policy.py`.

### Task P2b: Create `hermes_t/signal_policy.py`

Create: `hermes_t/signal_policy.py`

Extract:
- `SignalPolicy` dataclass
- `DEFAULT_SIGNAL_POLICY`
- `render_signal_text(action, policy)`
- All thresholds and Chinese text templates

**This is needed so runtime code never hardcodes strings like "第N次买入".**

### Task P2c: Wire `hermes_t/__main__.py` to call `run_runtime_cycle()`

The single-profile CLI should call the runtime cycle and output its result as JSON.

**No quote providers yet** — `--signal` and `--score` CLI args become the `tech_data`.

---

## Phase 3: Quote provider → TechDataAdapter split (independent PR)

**Objective:** Add the full `QuoteProvider` / `TechDataProvider` protocol split so `hermes_t` can consume real quote snapshots without changing runtime code.

### Files:
- Modify `hermes_t/tech_data.py` (create if doesn't exist)
- Wire to `hermes_t/__main__.py` as optional `--quote-data-config` / `--quote-snapshot-config`

### Key additions:
- `JsonQuoteDataProvider` (from file)
- `QuoteTechDataAdapter` (wraps provider, extracts `tech_data`)
- `TdxQuoteSnapshotSource` (minimal real TDX TCP, injectable API)
- `build_tech_data_provider(...)` that assembles the chain

**Test regression:** `uv run python3 -m pytest tests/test_trading_runtime.py -k 'quote' -q`

---

## Phase 4: Multi-profile orchestrator (independent PR)

**Objective:** Add multi-profile execution via JSON config so `hermes_t` can manage multiple symbols in sequence.

### Files:
- Create `hermes_t/orchestrator.py`
  - `load_runtime_profiles_from_json(path)` → list of `RuntimeProfile`
  - `run_profiles_from_config(profiles_config, tech_data_config, ...)` → summary dict
- Wire to `hermes_t/__main__.py` as `--profiles-config` / `--tech-data-config`

### Landed minimum scope:
- `python -m hermes_t --profiles-config profiles.json` is now wired
- Current mode is **sequential only**
- Multi-profile output is `{"profile_count": N, "results": [...]}`
- `--quote-data-config` / `--quote-snapshot-config` reuse the same provider assembly path

### Key constraint:
- Sequential execution only (no concurrency in this phase)
- Symbol-level tech_data fallback: missing → default

---

## Phase 5: Post-landing hardening (PR on any phase above)

**Objective:** Close the safety gaps discovered in earlier phases.

Priority list (from experience):
1. `load_runtime_profiles_from_json()` numeric field validation (reject bool, reject 0/negative)
2. Single-profile → multi-profile validation parity
3. `_validated_positive_int()` + `_missing_required_fields()` helpers
4. `_runtime_delivery_kwargs(args)` shared assembly helper
5. Readme synchronization for each phase

---

## Phase 6: Post-market review integration + monitoring (independent PR)

**Objective:** Hook the existing `build_post_market_review()` into the runtime cycle output and add a CLI flag for standalone review.

### Minimal scope:
- `run_runtime_cycle()` optionally calls `build_post_market_review()` when `post_market=True`
- `hermes_t/__main__.py` gets `--post-market` flag
- `python -m hermes_t --post-market --state-dir ...` works standalone
- Test: cross-module integration (runtime + review)

---

## Phase 7: Hermes agent tool integration

**Objective:** Create a Hermes skill (or cron job) that calls `uv run python3 -m hermes_t --post-market ...` and delivers the result.

### Scope:
- Create skill: `hermes-t-post-market-report`
- Cron job setup: `cronjob(action='create', schedule='30 9 * * 1-5', ...)`
- Deliver to Feishu home channel

---

## Execution principles

- **Each phase = one independent PR** — can be reviewed and merged separately
- **Each phase has TDD** — RED first, GREEN second, commit per task
- **Test regression command for each phase:**
  ```bash
  uv run python3 -m pytest tests/test_trading_runtime.py tests/test_post_market_review.py -q
  ```
- **Optimize for reviewability** — small PRs with clear boundaries
- **No hermes_olin coupling** — generic `TradingStateStore` throughout
- **Don't touch OpenClaw** — all work is in `hermes-agent` repo only

---

## Current progress

| Phase | Status | PR |
|-------|--------|----|
| Phase 0: Post-market review module | ✅ DONE + reviewed | #19148 (open, 2 commits) |
| Phase 1: Store + shared CLI | ✅ DONE | (PR already open, atop Phase 0) |
| Phase 2: Runtime cycle skeleton | ✅ DONE | (committed, atop Phase 1) |
| Phase 3: Quote provider split | ✅ DONE (minimal provider path) | atop current branch |
| Phase 4: Multi-profile orchestrator | ✅ DONE (minimal sequential CLI) | atop current branch / PR #19268 |
| Phase 5: Hardening | 🟡 IN PROGRESS | atop current branch |
| Phase 6: Review integration | — | — |
| Phase 7: Agent tool integration | — | — |
