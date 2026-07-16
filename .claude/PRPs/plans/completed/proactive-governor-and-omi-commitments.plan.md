# Plan: Notification/Attention Budget Governor + Omi Commitment → Kanban

## Summary
Add two coupled proactivity features to Hermes/Mercury. **Feature 1** is a *notification/attention budget governor*: every agent-initiated proactive message (cron delivery, webhook push, goal-status notice, Omi nudge) is scored and gated against a hard daily attention budget with per-category thresholds that self-tune from user dismiss/act feedback. **Feature 2** is an *Omi commitment extractor*: a scheduled pass reads the Omi wearable transcript via existing MCP tools, extracts commitments the user personally made, and files them as kanban cards — with every resulting notification routed through Feature 1 so it never nags.

## User Story
As a power user who lives in Slack and wears an Omi device,
I want Mercury to proactively surface commitments and reminders *without* flooding me,
so that I trust it as a chief-of-staff instead of muting it.

## Problem → Solution
**Current state:** Mercury has rich proactive *mechanisms* (cron, webhooks, goals loop) but no *judgment layer* — nothing decides whether an interruption is worth the user's attention, and ambient Omi capture never becomes action. → **Desired state:** A single governor caps and prioritizes all proactive sends and learns per-category from feedback; Omi commitments flow into the kanban board and are announced only when they clear the budget bar.

## Metadata
- **Complexity**: Large (2 new subsystems, 3 new state tables, ~4 integration points, config + tests)
- **Source PRD**: N/A (free-form, from research synthesis in this session)
- **PRD Phase**: N/A
- **Estimated Files**: ~10 new/changed (2 new core modules, `hermes_state.py`, `hermes_cli/config.py`, `cli-config.yaml.example`, `gateway/delivery.py`, `cron/scheduler.py`, 1 CLI command module, 2+ test files)

---

## UX Design

### Before
```
┌───────────────────────────────────────────────┐
│ cron/webhook/goal fire → adapter.send() → USER │
│  (no cap, no priority, no learning)            │
│ Omi transcript → (nothing)                     │
└───────────────────────────────────────────────┘
```

### After
```
┌──────────────────────────────────────────────────────────────┐
│ proactive producer                                             │
│   → notification_budget.should_deliver(category, value_hint)   │
│       ├─ ALLOW    → adapter.send() → USER   (ledger: allowed)  │
│       └─ SUPPRESS → dropped/deferred        (ledger: deferred) │
│                                                                │
│ user dismisses (❌ react / /notify mute) → threshold[cat] ↑    │
│ user engages   (reply / /notify keep)     → threshold[cat] ↓   │
│                                                                │
│ Omi scan (cron) → extract commitments (aux LLM, user-only)     │
│   → kanban create_task(idempotency_key)  → notify via governor │
└──────────────────────────────────────────────────────────────┘
```

### Interaction Changes
| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Proactive cron/webhook/goal message | Always delivered | Delivered only if under budget AND score ≥ threshold | Live replies to the user are **never** gated |
| Feedback | none | `/notify mute <cat>`, `/notify keep <cat>`, `hermes notify status`, ❌ reaction | Reaction wiring is a stretch item; CLI/slash is v1 |
| Omi transcript | ignored | scheduled scan → kanban cards | Opt-in; `omi_commitments.enabled: false` by default |
| Daily digest | none | `hermes notify status` shows budget used + suppressed/deferred items | Deferred items are retained, not lost |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `hermes_state.py` | 745-798, 843-911, 1249-1301, 1408-1474, 6407-6425 | Table DDL block, `_execute_write`, reconciler, `get/set_meta` template |
| P0 | `gateway/delivery.py` | 376-386, 388-527 | The router send path + existing pre-send silence gate to mirror |
| P0 | `hermes_cli/config.py` | 976-990, 2210-2233, 6302-6327, 6800-6814 | `DEFAULT_CONFIG` shape, section example, `_deep_merge`, `load_config` |
| P0 | `hermes_cli/kanban_db.py` | 1096-1180, 2387-2411, 2545-2553, 2725-2740 | `tasks` schema (NO due-date col), `create_task`, idempotency dedup, `list_tasks` |
| P0 | `tools/mcp_tool.py` | 4023-4192, 4683-4690 | `_make_tool_handler`, prefixed-name mapping (JSON-string return) |
| P0 | `agent/auxiliary_client.py` | 5177-5185, 6486-6510 | `get_text_auxiliary_client`, `call_llm` for extraction |
| P1 | `cron/scheduler.py` | 1405-1416, 1694-1713, 3521-3640 | Cron delivery path + standalone fallback (bypasses router) |
| P1 | `hermes_cli/goals.py` | 886-948 | Aux-LLM judge pattern (verbatim extractor analogue) |
| P1 | `gateway/run.py` | 12858-12870 | Goal-status notice send (a proactive producer to gate) |
| P1 | `gateway/platforms/webhook.py` | 1120-1266 | Webhook direct-deliver + cross-platform send |
| P2 | `tests/tools/test_kanban_redaction.py` | 19-38 | `worker_env` fixture: tmp kanban DB pattern |
| P2 | `tests/tools/test_mcp_structured_content.py` | 21-110 | Mocking an MCP server/handler in tests |
| P2 | `tests/tools/test_smart_approval_injection.py` | 124-141 | Mocking an aux-LLM call |
| P2 | `tests/conftest.py` | 30, 206, 328 | Hermetic env: `HERMES_HOME`→tmp, `HERMES_KANBAN_*` blanked, sys.path |

## External Documentation
| Topic | Source | Key Takeaway |
|---|---|---|
| Attention/notification budget | tianpan.co/blog/2026-05-13-background-agents-notification-budget-attention-economy | Optimize "notifications acted on", not "sent"; a dismissed-unopened ping is worse than none |
| Dismiss-driven threshold learning | wikimolt.ai/page/Interrupt%20Budget | Per-category threshold auto-raises on dismissal |
| Omi API/data shape | github.com/BasedHardware/omi | Conversations carry transcript segments with speaker/is_user; MIT-licensed |

No further external research needed — all integration surfaces are established internal patterns captured below.

---

## Patterns to Mirror

Follow these exactly. All snippets are verbatim from the live tree.

### NAMING_CONVENTION / MODULE_HEADER
```python
// SOURCE: agent/title_generator.py:1-20
"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. ...
FailureCallback = Callable[[str, BaseException], None]
```
Rule: module docstring → stdlib imports → `hermes_*` top-level imports → `agent.`/`tools.` intra-package → `logger = logging.getLogger(__name__)` → UPPER_SNAKE constants. Classes `PascalCase`, functions `snake_case`, module-private `_snake_case`.

### STATE_TABLE_DDL (add to the single SCHEMA_SQL string)
```python
// SOURCE: hermes_state.py:843-894 (state_meta + index style to mirror)
CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
...
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
```

### STATE_WRITE (every write routes through _execute_write; nested _do(conn), no commit)
```python
// SOURCE: hermes_state.py:6417-6425
    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the state_meta key/value store."""
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._execute_write(_do)
```

### STATE_READ (hold self._lock, return dict(row))
```python
// SOURCE: hermes_state.py:2932-2939
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None
```

### CONFIG_SECTION (new top-level key in DEFAULT_CONFIG dict)
```python
// SOURCE: hermes_cli/config.py:2210-2233 (memory section — booleans, ints, string selector)
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "write_approval": False,
        "memory_char_limit": 2200,
        "provider": "",
    },
```

### CONFIG_RUNTIME_ACCESS
```python
// SOURCE: cron/scheduler.py:2003
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
```

### PRE_SEND_GATE (mirror this drop-before-adapter gate exactly)
```python
// SOURCE: gateway/delivery.py:461-472
        if self._filter_silence_narration_enabled() and _is_silence_narration(content):
            logger.warning(
                "Dropped silence-narration outbound to %s (chat=%s): %r",
                target.platform.value, target.chat_id, content[:40],
            )
            return {
                "success": True,
                "filtered": "silence_narration",
                "delivered": False,
            }
```

### FLAG_RESOLUTION (env override → config default)
```python
// SOURCE: gateway/delivery.py:376-386
    def _filter_silence_narration_enabled(self) -> bool:
        env = os.getenv("HERMES_FILTER_SILENCE_NARRATION")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(getattr(self.config, "filter_silence_narration", True))
```

### KANBAN_CREATE (programmatic; idempotency_key dedups)
```python
// SOURCE: tests/tools/test_kanban_redaction.py:30-38 + hermes_cli/kanban_db.py:2387
from hermes_cli import kanban_db as kb
conn = kb.connect(board=board or None)
try:
    tid = kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee or None,
        idempotency_key=idem_key,   # dedup: returns existing id, no dup
        initial_status="triage",     # sit for review, don't auto-dispatch
    )
finally:
    conn.close()
```

### MCP_CALL_PROGRAMMATIC (returns a JSON *string*; json.loads + check "error")
```python
// SOURCE: tools/mcp_tool.py:4023 + registry dispatch tools/registry.py:614
from tools.registry import registry
raw = registry.dispatch("mcp__omi__get_conversations", {"limit": 50})
import json
data = json.loads(raw)
if "error" in data:
    ...  # failed fetch looks like a normal string result — must branch
convs = data.get("result")
```

### AUX_LLM_CLASSIFY (verbatim extractor analogue — the goal judge)
```python
// SOURCE: hermes_cli/goals.py:886-948
    client, model = get_text_auxiliary_client("goal_judge")
    if client is None or not model:
        return "continue", "no auxiliary client configured", False, None
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=_goal_judge_max_tokens(),
        timeout=timeout,
        extra_body=get_auxiliary_extra_body() or None,
    )
    raw = resp.choices[0].message.content or ""
```

### TEST_STRUCTURE (tmp kanban DB fixture)
```python
// SOURCE: tests/tools/test_kanban_redaction.py:19-38
@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()   # reset per-process init cache
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
    finally:
        conn.close()
    return tid
```

### TEST_MOCK_MCP + TEST_MOCK_AUX
```python
// SOURCE: tests/tools/test_mcp_structured_content.py:52-62
@pytest.fixture
def _patch_mcp_server():
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"omi": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session
```
```python
// SOURCE: tests/tools/test_smart_approval_injection.py:136-141
@patch("agent.auxiliary_client.call_llm")
def test_x(self, mock_call_llm):
    mock_call_llm.return_value = self._make_response("...json...")
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `agent/notification_budget.py` | CREATE | Governor: scoring, budget accounting, threshold learning. Pure logic + state access. |
| `agent/omi_commitments.py` | CREATE | Omi scan → extract (aux LLM) → dedup → kanban → notify. |
| `hermes_cli/notify_cmd.py` | CREATE | `hermes notify status/mute/keep`, `hermes omi scan/enable` CLI + `/notify` slash. |
| `hermes_state.py` | UPDATE | Add 3 tables to `SCHEMA_SQL`; add CRUD methods (ledger insert, category-stats upsert/read, omi-processed upsert/read). |
| `hermes_cli/config.py` | UPDATE | Add `notifications` + `omi_commitments` sections to `DEFAULT_CONFIG`. |
| `cli-config.yaml.example` | UPDATE | Document both new sections. |
| `gateway/delivery.py` | UPDATE | Insert governor gate in `_deliver_to_platform` when metadata flags proactive. |
| `cron/scheduler.py` | UPDATE | Tag cron deliveries with proactive metadata; gate the standalone (out-of-gateway) delivery path that bypasses the router. |
| `tests/agent/test_notification_budget.py` | CREATE | Unit tests for scoring/budget/learning. |
| `tests/agent/test_omi_commitments.py` | CREATE | Unit tests for extraction/dedup/consent, mocked MCP + aux LLM. |

## NOT Building
- ❌ No new kanban `due_date` **column** (schema migration to `tasks`). Due dates are embedded in card body as `Due: <ISO>` + provenance comment. (Adding a column to the kanban schema is deferred — out of scope.)
- ❌ No Slack Block Kit interactive buttons / reaction listeners in v1. Feedback is via `hermes notify`/`/notify`; ❌-reaction wiring is a documented stretch item.
- ❌ No gating of **live user-facing replies** — only messages explicitly tagged proactive are gated.
- ❌ No new external memory provider, no bi-temporal fact store (that was a separate research idea, not this plan).
- ❌ No ML model for `P(user acts)` — v1 uses an EWMA act-rate prior per category.
- ❌ No gating of the GitHub-comment webhook egress (`_deliver_github_comment`, subprocess) — documented uncovered path for v1.

---

## Data Model

Three new tables added to `SCHEMA_SQL` in `hermes_state.py` (bare `CREATE TABLE IF NOT EXISTS` — no `SCHEMA_VERSION` bump needed since there is no row backfill; `executescript(SCHEMA_SQL)` creates them every startup). Values shown are synthetic.

```sql
CREATE TABLE IF NOT EXISTS notification_ledger (
    id            TEXT PRIMARY KEY,          -- uuid4 hex, e.g. '9f1c...'
    candidate_id  TEXT,                       -- producer-supplied idempotency id (nullable)
    category      TEXT NOT NULL,              -- e.g. 'omi_commitment', 'cron:daily-brief', 'goal_status'
    score         REAL NOT NULL,              -- e.g. 0.42
    p_act         REAL NOT NULL,              -- e.g. 0.6
    value_hint    REAL NOT NULL,              -- e.g. 0.7
    attention_cost REAL NOT NULL,             -- e.g. 0.4
    threshold_used REAL NOT NULL,             -- e.g. 0.5
    decision      TEXT NOT NULL,              -- 'allowed' | 'suppressed' | 'deferred'
    platform      TEXT,                       -- e.g. 'slack'
    chat_id       TEXT,                       -- e.g. 'C0BHM1F7W10'
    day_key       TEXT NOT NULL,              -- 'YYYY-MM-DD' in local TZ, e.g. '2026-07-16'
    created_at    REAL NOT NULL,              -- epoch seconds float, e.g. 1784175252.0
    feedback      TEXT,                       -- 'act' | 'dismiss' | NULL
    feedback_at   REAL
);

CREATE TABLE IF NOT EXISTS notification_category_stats (
    category      TEXT PRIMARY KEY,           -- e.g. 'omi_commitment'
    threshold     REAL NOT NULL,              -- learned per-category bar, e.g. 0.55
    p_act_ewma    REAL NOT NULL,              -- exponential moving act-rate, e.g. 0.48
    sent_count    INTEGER NOT NULL DEFAULT 0,
    act_count     INTEGER NOT NULL DEFAULT 0,
    dismiss_count INTEGER NOT NULL DEFAULT 0,
    updated_at    REAL NOT NULL               -- epoch seconds float
);

CREATE TABLE IF NOT EXISTS omi_processed_conversations (
    conversation_id   TEXT PRIMARY KEY,       -- Omi conversation id string
    processed_at      REAL NOT NULL,          -- epoch seconds float
    commitments_found INTEGER NOT NULL DEFAULT 0
);
```
Indexes (in `SCHEMA_SQL`, all reference base columns so no `DEFERRED_INDEX_SQL` needed):
```sql
CREATE INDEX IF NOT EXISTS idx_notif_ledger_day ON notification_ledger(day_key);
CREATE INDEX IF NOT EXISTS idx_notif_ledger_cat_day ON notification_ledger(category, day_key);
CREATE INDEX IF NOT EXISTS idx_notif_ledger_candidate ON notification_ledger(candidate_id);
```

### Config schema
```python
"notifications": {
    "enabled": True,
    "daily_cap": 3,            # soft cap: below this, threshold gate applies
    "daily_ceiling": 5,        # hard cap: never exceed, regardless of score
    "base_threshold": 0.5,     # starting per-category bar
    "escalation_threshold": 0.8,  # bar to spend budget between cap and ceiling
    "dismiss_step": 0.1,       # threshold += on dismiss
    "act_step": 0.05,          # threshold -= on act
    "threshold_min": 0.1,
    "threshold_max": 0.95,
    "p_act_ewma_alpha": 0.3,   # learning rate for act-rate prior
    "default_value_hint": 0.5, # producer value when unspecified
    "categories": {},          # per-category overrides, e.g. {"omi_commitment": {"daily_cap": 2}}
},
"omi_commitments": {
    "enabled": False,          # OPT-IN (consent). Off by default.
    "scan_interval_hours": 6,
    "lookback_hours": 24,
    "min_confidence": 0.6,
    "board": "",               # empty = default board
    "assignee": "",            # empty = unassigned (triage, human-reviewed)
    "create_notification": True,
    "max_conversations_per_scan": 25,
},
```

---

## Governor Semantics (Feature 1 core logic)

`should_deliver(category, *, value_hint=None, candidate_id=None, platform=None, chat_id=None) -> BudgetDecision`

1. If `not cfg.notifications.enabled` → **ALLOW** (governor disabled = passthrough).
2. Idempotency: if `candidate_id` already has a non-deferred ledger row today → return its prior decision (no double-count).
3. Load/seed `notification_category_stats[category]` (defaults from config). Apply per-category config overrides.
4. `p_act = stats.p_act_ewma` (seed = `base_threshold`); `value = value_hint or default_value_hint`; `attention_cost = min(1.0, sent_today / daily_ceiling)`.
5. `score = p_act * value - attention_cost` (clamp to [-1, 1]).
6. `sent_today = COUNT(ledger WHERE decision='allowed' AND day_key=today)` (all categories).
7. Decision:
   - `sent_today >= daily_ceiling` → **SUPPRESS** (`decision='deferred'`, retained for digest).
   - `sent_today >= daily_cap` and `score < escalation_threshold` → **SUPPRESS** (`deferred`).
   - `score < threshold[category]` → **SUPPRESS** (`deferred`).
   - else → **ALLOW** (`allowed`), increment `stats.sent_count`.
8. Insert ledger row. Return `BudgetDecision(allow: bool, reason: str, score, threshold, category, ledger_id)`.

`record_feedback(category, signal, *, ledger_id=None)` where `signal ∈ {"act","dismiss"}`:
- `act`: `threshold = max(min, threshold - act_step)`; `p_act_ewma = (1-α)*p_act_ewma + α*1.0`; `act_count += 1`.
- `dismiss`: `threshold = min(max, threshold + dismiss_step)`; `p_act_ewma = (1-α)*p_act_ewma + α*0.0`; `dismiss_count += 1`.
- Stamp `feedback`/`feedback_at` on the ledger row if `ledger_id` given.

`DAY_KEY`: `time.strftime("%Y-%m-%d")` (local TZ; tests pin `TZ=UTC` via conftest).

---

## Step-by-Step Tasks

### Task 1: State tables + CRUD in `hermes_state.py`
- **ACTION**: Add the 3 `CREATE TABLE` blocks + 3 indexes to `SCHEMA_SQL` (near `hermes_state.py:882`, before the closing `"""` at `:894`).
- **IMPLEMENT**: Add methods on `SessionDB`: `record_notification(ledger_row: dict) -> None`; `count_notifications_today(day_key, *, decision="allowed") -> int`; `get_category_stats(category) -> Optional[dict]`; `upsert_category_stats(category, **fields) -> None`; `find_notification_by_candidate(candidate_id, day_key) -> Optional[dict]`; `set_notification_feedback(ledger_id, signal) -> None`; `list_deferred_today(day_key) -> List[dict]`; `mark_omi_conversation(conversation_id, commitments_found) -> None`; `omi_conversation_seen(conversation_id) -> bool`.
- **MIRROR**: STATE_WRITE (`set_meta`, `:6417`) for every writer; STATE_READ (`get_session`, `:2932`) for every reader.
- **IMPORTS**: already present in `hermes_state.py` (`json`, `time`). Add `import uuid` at top if absent (verify).
- **GOTCHA**: Writers MUST use nested `def _do(conn): ...; self._execute_write(_do)` and must NOT call `commit()`. Reads use `with self._lock:`. Use `INSERT ... ON CONFLICT(pk) DO UPDATE SET x=excluded.x` for upserts. Do NOT bump `SCHEMA_VERSION` — bare new tables need no data migration.
- **VALIDATE**: `python -c "from hermes_state import SessionDB; import tempfile,pathlib; db=SessionDB(pathlib.Path(tempfile.mkdtemp())/'s.db'); db.upsert_category_stats('t', threshold=0.5, p_act_ewma=0.5, updated_at=0.0); print(db.get_category_stats('t'))"` prints a dict.

### Task 2: Config sections
- **ACTION**: Add `"notifications"` and `"omi_commitments"` keys to `DEFAULT_CONFIG` (`hermes_cli/config.py:976+`, alongside `memory`/`delegation`).
- **IMPLEMENT**: The two dicts from the Data Model section verbatim.
- **MIRROR**: CONFIG_SECTION (`memory`, `:2210`).
- **GOTCHA**: `_deep_merge` ignores a `None` override of a dict default, so a bare `notifications:` line won't wipe defaults — safe. New keys are auto-available after `load_config()`; no other registration required. Optionally add both keys to `_KNOWN_ROOT_KEYS` (`:5266`) for tidiness (not required to load).
- **VALIDATE**: `python -c "from hermes_cli.config import load_config as l; c=l(); print(c['notifications']['daily_cap'], c['omi_commitments']['enabled'])"` → `3 False`.

### Task 3: Governor module `agent/notification_budget.py`
- **ACTION**: Create the governor implementing the semantics above.
- **IMPLEMENT**: `@dataclass BudgetDecision`; `should_deliver(...)`; `record_feedback(...)`; `budget_status(day_key=None) -> dict` (used by CLI). Resolve the `SessionDB` via `SessionDB(DEFAULT_DB_PATH)` opened lazily inside each function (mirror how cron opens a `SessionDB`). Read config via `load_config().get("notifications", {})`. Env kill-switch `HERMES_NOTIFICATIONS_DISABLED`.
- **MIRROR**: MODULE_HEADER (`title_generator.py:1-20`); FLAG_RESOLUTION (`delivery.py:376`) for the env override; LOGGING (`logger.info("... %s", arg)`).
- **IMPORTS**: `import logging, time, uuid`; `from dataclasses import dataclass`; `from typing import Optional`; `from hermes_cli.config import load_config`; `from hermes_state import SessionDB, DEFAULT_DB_PATH`.
- **GOTCHA**: Keep this module import-light and side-effect-free at import time (no DB open at module scope) — it's imported by `gateway/delivery.py` on the hot path. Open the DB lazily inside functions. Fail **open**: any exception → log at debug and ALLOW (never let the governor block a message due to its own bug — mirror the fail-open posture in `goals.py:886`).
- **VALIDATE**: unit tests in Task 9.

### Task 4: Wire governor into the router (`gateway/delivery.py`)
- **ACTION**: In `_deliver_to_platform` (`:388`), immediately after the silence-narration gate (`:472`), add a proactive-budget gate.
- **IMPLEMENT**:
  ```python
  if send_metadata and send_metadata.get("proactive"):
      from agent.notification_budget import should_deliver
      decision = should_deliver(
          category=send_metadata.get("notification_category", "generic"),
          value_hint=send_metadata.get("value_hint"),
          candidate_id=send_metadata.get("candidate_id"),
          platform=target.platform.value,
          chat_id=target.chat_id,
      )
      if not decision.allow:
          logger.info("Notification budget suppressed proactive to %s (cat=%s, score=%.2f<thr=%.2f)",
                      target.platform.value, decision.category, decision.score, decision.threshold)
          return {"success": True, "filtered": "notification_budget", "delivered": False}
  ```
- **MIRROR**: PRE_SEND_GATE (`delivery.py:461-472`) — same early-return dict shape (`{"success": True, "filtered": ..., "delivered": False}`).
- **GOTCHA**: Import `should_deliver` **inside** the method (lazy) to avoid an import cycle (`delivery.py` ↔ `notification_budget` ↔ config/state). Only gate when `metadata["proactive"]` is truthy — this is what protects live replies. Governor never raises (fail-open).
- **VALIDATE**: with stub metadata `{"proactive": True, "notification_category":"test"}` and a mocked `should_deliver` returning `allow=False`, `_deliver_to_platform` returns the filtered dict without calling `adapter.send`.

### Task 5: Tag proactive producers + cover the standalone cron path
- **ACTION**: (a) In cron `route_metadata` construction (`cron/scheduler.py:1694-1713`), add `proactive: True`, `notification_category: "cron:<job_id or tag>"`, `candidate_id: <job_id+run_ts>`, `value_hint` to `route_metadata`. (b) The standalone `_send_to_platform` path (`cron/scheduler.py:1903`) bypasses the router, so call `should_deliver(...)` directly there before sending and skip on suppress. (c) In `gateway/run.py:_send_goal_status_notice` (`:12858`), tag its `metadata` with `proactive: True, notification_category: "goal_status"`. (d) Webhook `_deliver_cross_platform` (`webhook.py:1223`) — add the same metadata tag so router-path webhook deliveries are gated.
- **IMPLEMENT**: metadata dict additions + one direct `should_deliver` call at the standalone path.
- **MIRROR**: CONFIG_RUNTIME_ACCESS for reading category defaults.
- **GOTCHA**: Do NOT tag the agent's normal turn reply path or `send_message` tool calls unless they are genuinely proactive — scope v1 to cron, webhook cross-platform, goal-status. `candidate_id` must be stable per logical notification so retries don't double-spend budget.
- **VALIDATE**: manual — a cron job whose delivery is suppressed logs `notification_budget` and writes a `deferred` ledger row.

### Task 6: Omi extractor `agent/omi_commitments.py`
- **ACTION**: Create `run_omi_commitment_scan() -> dict` (returns summary: scanned, extracted, created, notified).
- **IMPLEMENT**:
  1. Guard: if `not cfg.omi_commitments.enabled` → return `{"skipped": "disabled"}` (consent gate).
  2. Fetch: `registry.dispatch("mcp__omi__get_conversations", {"limit": max_conversations_per_scan})`; `json.loads`; branch on `"error"`. Filter to `lookback_hours` and unseen (`omi_conversation_seen`).
  3. Extract per conversation via `get_text_auxiliary_client("omi_commitment")` → `chat.completions.create(..., temperature=0)` with a system prompt that: returns strict JSON `{"commitments":[{"text","due_iso"|null,"confidence","made_by_user":bool}]}`, and **only includes commitments the device owner personally made** (ignore bystanders, TV/radio, other speakers) using the transcript's speaker/is_user segments.
  4. Filter `confidence >= min_confidence` and `made_by_user`.
  5. Create card: `kb.create_task(conn, title=<≤80-char summary>, body="Due: <due_iso or 'none'>\n\nFrom Omi conversation <id> @ <ts>\n> <quote>", assignee=cfg.assignee or None, idempotency_key=sha256(conv_id + "|" + normalized_text)[:32], initial_status="triage")`. Add provenance comment via `kb.add_comment`.
  6. `mark_omi_conversation(conv_id, commitments_found)`.
  7. If `create_notification`: build a summary line and deliver it through the governor — route via the gateway `send_message`/router with `metadata={"proactive": True, "notification_category": "omi_commitment", "candidate_id": conv_id, "value_hint": max(confidence)}`. In a standalone (cron) context, call `should_deliver(...)` first, then the standalone sender.
- **MIRROR**: MCP_CALL_PROGRAMMATIC; AUX_LLM_CLASSIFY (`goals.py:886`); KANBAN_CREATE.
- **IMPORTS**: `import hashlib, json, logging, time`; `from hermes_cli.config import load_config`; `from hermes_cli import kanban_db as kb`; `from tools.registry import registry`; `from agent.auxiliary_client import get_text_auxiliary_client, get_auxiliary_extra_body`; `from hermes_state import SessionDB, DEFAULT_DB_PATH`.
- **GOTCHA**: MCP handler returns a **JSON string**, and a failed Omi fetch returns `{"error":...}` (never raises) — must branch. MCP requires `discover_mcp_tools()` to have run and the background loop alive; when run from cron this is already true (`cron/scheduler.py:3158`), but guard with a clear log if `registry.dispatch` returns an MCP-not-connected error. Kanban `tasks` has **NO due-date column** — embed due in body. `create_task` default `initial_status="running"` → pass `initial_status="triage"` so cards await human review and aren't auto-dispatched. Parse the aux-LLM JSON defensively (strip code fences, `try/except json.JSONDecodeError` → skip conversation, log).
- **VALIDATE**: unit tests in Task 10.

### Task 7: CLI/slash `hermes notify` + `hermes omi scan/enable`
- **ACTION**: Create `hermes_cli/notify_cmd.py` and register in the shared command registry (mirror how `hermes cron` is registered in `hermes_cli/commands.py`).
- **IMPLEMENT**: `hermes notify status` (prints today's budget: used/cap/ceiling, per-category thresholds, deferred items from `list_deferred_today`); `hermes notify mute <category>` → `record_feedback(cat, "dismiss")`; `hermes notify keep <category>` → `record_feedback(cat, "act")`; `hermes omi scan` → `run_omi_commitment_scan()` (manual trigger); `hermes omi enable/disable` → flip `omi_commitments.enabled` + (de)register the cron job (Task 8). Expose `/notify` as a gateway slash command mirroring an existing read-only slash.
- **MIRROR**: existing CLI command modules (e.g. `hermes_cli/webhook.py` CLI shape) and slash registration in `hermes_cli/commands.py`.
- **GOTCHA**: Keep it read-mostly; feedback + enable/disable are the only writers. No secrets in output.
- **VALIDATE**: `hermes notify status` runs against a fresh DB and prints `0/3` used.

### Task 8: Register the Omi scan as a cron job (opt-in)
- **ACTION**: `hermes omi enable` registers a cron job that runs `agent.omi_commitments.run_omi_commitment_scan` every `scan_interval_hours`; `hermes omi disable` removes/pauses it.
- **MIRROR**: cron job creation via `cron/jobs.py` store API (as `hermes cron` uses).
- **GOTCHA**: Do not auto-register on install — only on explicit opt-in (consent). Idempotent: check for an existing job by a stable name/tag before creating.
- **VALIDATE**: after `hermes omi enable`, `hermes cron list` shows the omi-scan job; `disable` removes it.

### Task 9: Tests — governor (`tests/agent/test_notification_budget.py`)
- **ACTION**: Unit-test scoring, caps, and learning with a tmp `SessionDB`.
- **IMPLEMENT**: fixtures per TEST_STRUCTURE (tmp `HERMES_HOME`, fresh state.db). Cases in Testing Strategy table below.
- **MIRROR**: `worker_env` tmp-DB fixture; conftest hermetic env (already autouse).
- **VALIDATE**: `scripts/run_tests.sh tests/agent/test_notification_budget.py` green.

### Task 10: Tests — Omi extractor (`tests/agent/test_omi_commitments.py`)
- **ACTION**: Unit-test extraction/dedup/consent with mocked MCP + aux LLM.
- **IMPLEMENT**: `_patch_mcp_server` fixture returning synthetic Omi conversations (inline dicts — no shared fixture dir); patch `get_text_auxiliary_client` (or `call_llm`) returning a canned JSON commitments payload. Assert: disabled→skip; bystander/TV commitment excluded; card created with `Due:` in body; second run on same conversation creates no duplicate (idempotency_key); low-confidence dropped.
- **MIRROR**: TEST_MOCK_MCP, TEST_MOCK_AUX, KANBAN_CREATE (tmp DB).
- **GOTCHA**: `kb._INITIALIZED_PATHS.clear()` before `kb.init_db()` in the fixture. MCP mock's `call_tool` must be an `AsyncMock` returning a fake result with `.content`/`.structuredContent` so `_make_tool_handler`/dispatch yields the JSON-string shape.
- **VALIDATE**: `scripts/run_tests.sh tests/agent/test_omi_commitments.py` green.

---

## Testing Strategy

### Unit Tests
| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| governor disabled → passthrough | `enabled=False` | `allow=True`, no ledger row | |
| first send under cap | 0 sent today, score≥threshold | `allow=True`, ledger `allowed` | |
| threshold not met | score < threshold | `allow=False`, ledger `deferred` | |
| soft cap + low score | sent=3 (cap), score<escalation | `allow=False` `deferred` | ✓ |
| soft cap + high score | sent=3, score≥escalation, sent<ceiling | `allow=True` | ✓ |
| hard ceiling | sent=5 (ceiling) | `allow=False` regardless of score | ✓ |
| idempotent candidate | same `candidate_id` twice | second returns first decision, no double-count | ✓ |
| dismiss raises threshold | `record_feedback(cat,"dismiss")` | `threshold += dismiss_step`, clamped ≤ max | |
| act lowers threshold | `record_feedback(cat,"act")` | `threshold -= act_step`, clamped ≥ min | |
| per-category override | `categories.omi_commitment.daily_cap=1` | omi capped at 1 independent of global | ✓ |
| governor internal error | DB open fails | `allow=True` (fail-open), logged | ✓ |
| omi disabled | `omi_commitments.enabled=False` | `{"skipped":"disabled"}`, no MCP call | ✓ |
| omi MCP error | dispatch returns `{"error":...}` | scan returns gracefully, 0 created, logged | ✓ |
| bystander excluded | conv w/ non-user speaker commitment | not created (`made_by_user=false`) | ✓ |
| dedup | same conversation scanned twice | 1 card total (idempotency_key) | ✓ |
| low confidence dropped | `confidence < min_confidence` | not created | ✓ |
| due embedded | commitment w/ `due_iso` | card body contains `Due: <iso>` | |
| malformed aux JSON | LLM returns non-JSON | conversation skipped, no crash | ✓ |

### Edge Cases Checklist
- [x] Empty input (no conversations / no commitments)
- [x] Maximum size input (`max_conversations_per_scan` cap honored)
- [x] Invalid types (malformed aux-LLM JSON, MCP error dict)
- [x] Concurrent access (writes via `_execute_write` BEGIN IMMEDIATE + retry)
- [x] Network failure (MCP/Omi unreachable → graceful)
- [x] Permission denied / disabled (consent gate, governor disabled)

---

## Validation Commands

### Static Analysis
```bash
cd /home/jtomek/.hermes/hermes-agent
python -m pyflakes agent/notification_budget.py agent/omi_commitments.py hermes_cli/notify_cmd.py
```
EXPECT: no undefined-name / unused-import errors. (Repo uses `logging.getLogger(__name__)`, `%`-style logging, type hints — match.)

### Unit Tests (affected area)
```bash
cd /home/jtomek/.hermes/hermes-agent
scripts/run_tests.sh tests/agent/test_notification_budget.py
scripts/run_tests.sh tests/agent/test_omi_commitments.py
```
EXPECT: all pass. (Runner isolates each file in a subprocess; conftest provides hermetic `HERMES_HOME`.)

### Config load smoke
```bash
python -c "from hermes_cli.config import load_config as l; c=l(); print(c['notifications'], c['omi_commitments'])"
```
EXPECT: both sections present with documented defaults.

### State schema smoke
```bash
python -c "import tempfile,pathlib; from hermes_state import SessionDB; d=SessionDB(pathlib.Path(tempfile.mkdtemp())/'s.db'); print(sorted(r[0] for r in d._conn.execute(\"select name from sqlite_master where type='table' and (name like 'notification%' or name like 'omi%')\")))"
```
EXPECT: `['notification_category_stats', 'notification_ledger', 'omi_processed_conversations']`.

### Full Test Suite (regression)
```bash
cd /home/jtomek/.hermes/hermes-agent
scripts/run_tests.sh
```
EXPECT: no new failures vs. baseline.

### Manual Validation
- [ ] `hermes notify status` on a fresh profile prints `0/3` used, default thresholds.
- [ ] Create a cron job that delivers a message; force it 6× in a day → 3 (or up to 5 for high score) delivered, rest logged `notification_budget` + `deferred` in ledger.
- [ ] `hermes notify mute cron:<tag>` then re-fire → threshold raised, more suppression.
- [ ] With `omi_commitments.enabled: true` + a live Omi server, `hermes omi scan` creates triage cards with `Due:` in body; re-run creates no duplicates.
- [ ] Confirm a normal @-mention reply to Mercury is **never** gated (no `proactive` metadata).

---

## Acceptance Criteria
- [ ] All 10 tasks completed.
- [ ] All validation commands pass; no regressions in full suite.
- [ ] Governor gates only messages tagged `proactive`; live replies untouched.
- [ ] Daily cap (3) and hard ceiling (5) enforced; per-category thresholds learn from dismiss/act.
- [ ] Omi scan is opt-in (consent), extracts only user-made commitments, dedups, embeds due-date in body, and routes its notification through the governor.
- [ ] No type/lint errors; no hardcoded secrets.

## Completion Checklist
- [ ] Code follows discovered patterns (STATE_WRITE/READ, CONFIG_SECTION, PRE_SEND_GATE, MCP_CALL, AUX_LLM).
- [ ] Error handling fail-open in the governor; graceful in the Omi scan.
- [ ] Logging via `logging.getLogger(__name__)`, `%`-style, `exc_info=True` on errors.
- [ ] Tests follow tmp-DB + mock-MCP + mock-aux patterns.
- [ ] No hardcoded values (all knobs in config).
- [ ] `cli-config.yaml.example` documents both sections.
- [ ] Self-contained — no codebase searching needed during implementation.

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Governor accidentally gates live replies | Low | High | Gate ONLY when `metadata["proactive"]` truthy; explicit manual test; producers opt-in |
| Import cycle `delivery.py`↔governor | Med | Med | Lazy import inside `_deliver_to_platform`; governor has no module-scope DB open |
| Cron standalone path bypasses router gate | High (known) | Med | Explicitly call `should_deliver` at `cron/scheduler.py:1903`; documented as a second insertion point |
| Omi speaker attribution wrong (files bystander/TV commitments) | Med | Med | Aux prompt requires `made_by_user`; `initial_status="triage"` keeps human in loop; confidence floor |
| MCP not connected in scan context | Med | Low | Branch on `{"error":...}`; clear log; scan returns gracefully |
| Double-spending budget on retries | Med | Med | Stable `candidate_id` + idempotency check in `should_deliver` |
| Kanban has no due-date column | Certain (known) | Low | Embed `Due:` in body (documented NOT-building item); column migration deferred |
| GitHub-comment webhook egress ungated | Low | Low | Documented as uncovered in v1 |

## Notes
- **Fail-open is a hard invariant** for the governor: a bug in scoring must never silence a message. Every governor entry point wraps its body in `try/except → log(debug) → return ALLOW`.
- **Coverage boundary**: v1 gates the router path (`_deliver_to_platform`) + the cron standalone path + goal-status + webhook cross-platform. It does NOT gate `_deliver_github_comment` (subprocess) — acceptable for v1.
- **Why source-tagging over blind chokepoint gating**: gating `adapter.send` universally would suppress live replies. Tagging proactive producers + one router gate is the minimal-surface, live-reply-safe design, and mirrors the existing silence-narration gate precedent.
- **Learning signal quality**: v1 relies on explicit `/notify mute|keep` + (stretch) ❌-reaction. Implicit act-detection (user replies within N min) is a documented future enhancement.
- Follow-on ideas from research not in this plan: bi-temporal fact memory, nightly consolidation, eval-gated instinct promotion, stalled-thread follow-up (the stalled-thread detector would be a natural third proactive producer feeding this same governor).
```
