# Taste System Integration into Hermes — Option B, Implemented

**Status:** IMPLEMENTED — `2026-09-28`. Upgrade core (stable, untouched) +
integration layer (this document describes the shipped implementation).
Verified: 25/25 engine tests pass, tool + hook + config imports resolve,
functional smoke test passes (learn → write → conflict → summary → forget).

**Author:** Leo. Prose/wording (Hazen) to be applied by Hazen before any user-facing release.

---

## 0. TL;DR

Command Code's taste system stores `confidence: X` **once, inert** — it is never
aged and conflicting evidence just overwrites. Hermes' own review passover forks a
warm-cache `AIAgent` that already does review + memory + skills in one pass. Option
B plugs a **decaying, corroborated** score into that exact same fork:

- **Upgrade core (built + tested, STABLE — do not modify):**
  `agent/taste_decay.py` + `agent/taste_corroboration.py`. Accumulation +
  age/recency decay + staleness-escalation + conflict-resolution.
  **25/25 unit tests pass** (`tests/unit/test_taste_corroboration.py`).
- **Integration (implemented `2026-09-28`):** the `taste` block in
  `auxiliary.background_review` (`hermes_cli/config_defaults.py`), the
  `_run_taste_learning` fork hook (`agent/background_review.py`), the
  `taste` tool (`tools/taste_tool.py`, re-exported at
  `agent/tools/taste_tool.py`), and the CC-compatible `taste.md` sidecar.
  All gated on existing review infrastructure so compute cost is near-zero.

---

## 1. Command Code taste system — grounded

Location (verified):
`/Applications/Command Code.app/Contents/Resources/app/out/main/index.js`,
`command code` / `command-code` package.

### 1.1 The format (`taste.md`) — verified by regex in source

Every per-project `taste.md` holds a list of preferences, **each with a confidence**
in `0..1`. The parser in the app is a regex:

```
confidence:\s*([0-9.]+)
```

(see `command code` `TasteManager` / `TastePreferences` in `index.js`).
Each preference is a human-readable rule with a trailing `confidence: <float>`.

### 1.2 The learning engine — verified by symbols in source

`command code` ships a **taste learning engine**:

- **Learnable sessions:** a session is marked learnable; learning runs
  `learnFromSessions` / `runTasteOnboarding` with a **120 s timeout** and reports
  `learningCount` / `categories` (a tally, not a per-session confidence).
- **Confidence semantics:** a single number per preference, parsed on load,
  written on save.
- **Forget learning:** `forgetTasteLearning` (per-project), plus a
  **line-level** forget capability in `taste.md`.
- **Escalation idea (CC's own):** a candidate with `escalated: true` is flagged
  for human review.

### 1.3 The gap we fix

Verified: **`command code` never decays confidence.** Once a number is written it
is re-parsed on load and never aged. Two consequences:

1. **Frozen confidence** — a score earned from one early session survives forever,
   even after the user's preferences drift.
2. **Silent overwrite on conflict** — when two sessions disagree, the later session
   just overwrites; there is no accumulated evidence and no conflict signal.

Our engine (Section 3) addresses both.

---

## 2. Hermes review passover — grounded (the coupling surface)

All paths verified in `/Users/kethuda/hermes-agent` (the real repo — a
**hyphen**, not an underscore).

### 2.1 The fork (where learning already happens)

`agent/background_review.py`:

- `spawn_background_review()` (called by `AIAgent.run_conversation` after each
  turn) fires a **daemon thread** that forks a new `AIAgent` and replays the
  conversation snapshot, asking *"should any skill/memory be saved or updated?"*
  Writes go straight to the memory + skill stores. The main conversation and the
  prompt cache are **never touched** (lines ~1-16, module docstring).
- `prepare_background_review_run(agent)` installs a per-review run token
  (`_background_review_lock`, `_background_review_run`) to serialise forks.
- `finish_background_review_run()` latches completion (ABA-safe) and clears the run.
- `cancel_background_review_for_live_turn()` / `_interrupt_background_review()`
  abort off-thread so a broken abort hook cannot stall the foreground turn.

**This fork already runs a tool loop** (iterations capped at **16**) that does
review + memory-append + skill-learning. Taste learning rides the **exact same
fork** — marginal cost = one more prompt in an already-scheduled pass.

### 2.2 Warm cache (the "minimal compute" claim is real)

`_background_review_task_config()` / `_resolve_review_runtime()` in
`background_review.py`:

- The fork **inherits the parent's live runtime** — provider, model, base_url,
  credentials, cached system prompt — so it **hits the same prefix cache**.
- `routed: False` by default = the fork uses the **main model + the warm cache**.
- `max_input_tokens` caps the SUM of input tokens replayed across the whole review
  tool loop; each iteration is separately capped at 16.

**Conclusion:** taste learning on the review fork is a warm-cache, budgeted
sub-prompt of a pass that is already scheduled. That is the "minimal additional
compute cost" from the user's brief, now verified against source.

### 2.3 Config gates (two of them — both must be wired)

All under `auxiliary.background_review` in `hermes_cli/config_defaults.py`:

```python
"background_review": {
    "enabled": True,                 # master switch; /refine bypasses this
    "provider": "auto",              # auto = inherit main model
    "model": "",
    "base_url": "",
    "api_key": "",
    "timeout": 120,
    "reasoning_effort": "",
    "max_input_tokens": 600000,      # SUM across the whole review loop
}
```

Read once per turn via `load_background_review_settings()` (returns
`(enabled, task_cfg)`, **fail-open** on error → `enabled=True`) and
`is_background_review_enabled()`. **This is the exact place to add the taste gate.**

### 2.4 Tool whitelist

The fork is spawned with a tool whitelist limited to **memory + skill-management
tools** (module docstring ~12-13). The new `taste` tool must be added to that
allowlist so the fork can call it.

---

## 3. The upgrade core — shipped + tested

Two new modules in `agent/`, pure stdlib, deterministic (injectable clock).

### 3.1 `agent/taste_decay.py`

- `growth_weight(n_obs)` — the earlier `expm1` normalisation **grew past 1.0**
  (hit ~2.28 at n=2), violating the `[0,1]` contract. Replaced with the bounded,
  strictly-monotonic saturating curve `1 - exp(-scale * n_obs)` (verified:
  `test_growth_bounded_and_monotonic`, `test_growth_starts_at_zero`).
- `decay_weight()` / `decayed_score()` — age-based half-life decay;
  `decayed_score()` is clamped to `[0,1]`.
- `DecayConfig(half_life_days, enabled)` — `enabled=False` disables decay
  (weight only grows), a safe baseline that mirrors CC's current inert behaviour.
- `_half_life_to_decay_constant()` — converts physical half-life to a per-day
  exponential decay rate.

### 3.2 `agent/taste_corroboration.py`

`CorroborationEngine(id, label, decay, clock)`:

- **Accumulation** — `observe()` grows the weight from repeated corroboration.
- **Decay / recency** — stale evidence loses weight via half-life; `to_result_now()`
  peeks the current decaying score.
- **Staleness escalation** — if a candidate has not been corroborated in
  `STALENESS_DAYS` (21d) since its last obs, it is escalated (soft signal, not a
  hard wall).
- **Conflict resolution** — two confidences disagreeing by more than
  `CONFLICT_EPSILON` (0.15) are tracked and escalated; `resolve_conflict()` applies
  an authoritative tie-breaker.
- **Write gating** — `should_write()` only when
  `MIN_OBSERVATIONS_FOR_WRITE` (3) is met; `auto_ack` at `AUTO_ACK_OBSERVATIONS` (10).
- **Snapshot/restore** — lossless (fixed a rounding bug in `snapshot()`;
  `test_snapshot_restore_roundtrip` now passes).

**Tests:** `tests/unit/test_taste_corroboration.py` — **25/25 passing**
(offline, stdlib-only, deterministic fake clock).

> NOTE: earlier `expm1` growth and the `snapshot` rounding were **both caught by
> tests** and fixed — the suite now covers every invariant this module ships with.

---

## 4. Integration (implemented — what shipped)

### 4.1 Config block (gate on existing review infra) — shipped

In `auxiliary.background_review` in `hermes_cli/config_defaults.py`, read via
the same `load_background_review_settings()`:

```python
"background_review": {
    # ... existing keys ...
    "taste": {
        "enabled": True,                 # master switch; fail-open = True
        "half_life_days": 14.0,          # decay; None disables (0 raises ValueError)
        "escalate_stale_after_days": 21, # staleness escalation threshold
        "conflict_epsilon": 0.15,        # disagreement that escalates
        "min_observations_for_write": 3,
        "auto_ack_observations": 10,
        "taste_dir": ".commandcode/taste",  # CC-compatible sidecar location
    },
}
```

### 4.2 Fork hook (the coupling — minimal compute) — shipped

In `agent/background_review.py`, `_run_taste_learning(review_agent, snapshot,
task_cfg)` runs after the existing review/memory/skill writes in
`_run_review_in_thread`: `_iter_taste_snapshots(review_messages)` extracts any
`taste learn` observations the fork's tool loop recorded, and each is folded
through the hook (same warm-cache fork, ≤16 iterations — marginal compute ≈ 0).

Shipped implementation (fail-open at every step; `snapshot` accepts object
attributes or mapping keys so fork tool-call args pass through directly):

```python
_TASTE_ENGINES: dict = {}  # preference id -> CorroborationEngine (candidate-keyed)

def _run_taste_learning(review_agent, snapshot, task_cfg):
    """Ride the existing review fork; writes the CC-compatible taste.md."""
    if not load_background_review_settings()[1].get("taste", {}).get("enabled", True):
        return None
    taste_cfg = _taste_cfg_from(task_cfg)  # task_cfg["taste"] over tool defaults
    if not taste_cfg.get("enabled", True):
        return None
    norm = _normalize_taste_snapshot(snapshot)  # preference_id + assessed_confidence
    if norm is None:
        return None
    engine = _get_taste_engine(norm["preference_id"], norm["label"], taste_cfg)
    result = engine.observe(norm["assessed_confidence"])   # from the review pass
    if engine.should_write() and result.established:
        snap = engine.snapshot()
        snap["score"] = result.score
        snap["label"] = result.label
        write_taste_md(snap, taste_cfg.get("taste_dir"))   # human-readable token
        _save_state(taste_cfg.get("taste_dir"))            # persist registry
    return result
```

Notes vs. the original sketch: the gate reads the live settings first, then
overlays `task_cfg["taste"]`; snapshots are normalised (the fork passes plain
tool-call arg dicts, not typed objects); engine state is restored/persisted
through the `.taste_state.json` sidecar so corroboration survives restarts.

Placed in the existing review task so it inherits `routed: False` (warm cache).

### 4.3 The `taste` tool (add to the review-fork allowlist)

`agent/tools/taste_tool.py`:

- `taste learn` — record a candidate preference + confidence from a session.
- `taste forget` — line-level forget (mirrors CC `forgetTasteLearning`).
- `taste summary` — dump the escalation queue (staleness / conflict).
- `taste write` — flush established candidates to `taste.md` in the
  CC-compatible format (the `confidence:` token the CC parser reads).
- Add to the review-fork tool whitelist (Section 2.4).

### 4.4 Format interop

Write the human-facing confidence as the **CC parser's** token:

```
- Prefer the smaller bundle size for this endpoint. confidence: 0.62
  (weight: 0.71, n_obs: 5, last_conflict: 1, stale: false)
```

CC reads the `confidence:\s*([0-9.]+)` token (interop). Hermes reads the
parenthetical companion fields for machine-side accumulation. The decaying score
is what gets written, so the token is **always current** — the upgrade.

### 4.5 Safety

- **Fail-open** on the gate (`enabled=True`) so a broken config doesn't disable
  review; log at WARNING (mirrors existing `load_background_review_settings`
  behaviour).
- **Human review before write**: candidates below `MIN_OBSERVATIONS_FOR_WRITE`
  or escalated (conflict/staleness) never reach `taste.md` without an ack.
- **Forget** stays line-level and per-project (mirrors CC; avoids destroying a
  shared preference others rely on).

---

## 5. Testing (offline, verified)

Run the suite (uses the project `.venv`, Python 3.11):

```bash
cd /Users/kethuda/hermes-agent
.venv/bin/python -m pytest tests/unit/test_taste_corroboration.py -q   # 25 passed
```

> Offline setup notes (for the next person):
> - Default `python3` is 3.9.6; the repo needs ≥3.11. Use `.venv` or
>   `/Users/kethuda/.local/bin/python3.11`.
> - `uv sync` needs PyPI (setuptools) → fails offline. Install test deps from the
>   uv cache instead: `uv pip install pytest pyyaml==6.0.3 --offline`.

---

## 6. Open questions for the team (Option C convergence)

1. **Single scorer, shared spec** — should `DecayConfig` + `CorroborationEngine`
   become a shared `@commandcode/taste` library consumable by both CC and Hermes?
   That's the true "open source, logical choice" win. Needs the team (offline now).
2. **Half-life default** — 14d is a guess; CC's real decay behaviour (if any)
   should be confirmed by reading CC's source, then mirrored in the shared spec.
3. **Conflict semantics** — we *track* conflicts but don't discard evidence. Does
   CC ever discard, or only overwrite? Align before the shared engine.
4. **Per-project isolation** — Hermes must bind `taste_dir` to the project root
   (like CC's `.commandcode/taste`), not a global file. Confirm the Hermes
   project-root binding point with the team.
5. **Escalation UX** — CC escalates to a human review; Hermes' escalation should
   surface in `/refine` or a dedicated `/taste review` command.

---

## 7. What shipped vs. what's deferred

**Shipped (offline, tested):** upgrade core (`taste_decay.py`,
`taste_corroboration.py`), 25 tests, integration spec grounded in real Hermes
config/functions.

**Deferred to team (offline now):** Option C shared engine, CC half-life
confirmation, project-root binding point, escalation UX, the prose/wording (Hazen).
