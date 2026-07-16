# Plan: Stalled-Thread Follow-Up (proactive chief-of-staff nudges)

## Summary
A scheduled, opt-in detector that finds open loops — commitments that went past due or untouched, and conversations awaiting the user's reply — and resurfaces them as one batched digest ("2 things need follow-up: …"). It is a new *proactive producer* that feeds the already-shipped notification budget governor (category `stalled_thread`), so it never floods. It reuses the Omi scan structure end-to-end; the only new logic is open-loop detection over two data sources.

## User Story
As a busy operator who commits to things in chat and via my Omi wearable,
I want Mercury to remind me of commitments and replies I've let go quiet,
so that nothing important silently falls through the cracks — without being nagged.

## Problem → Solution
**Current state:** Mercury files commitments (Omi → kanban) but never follows up on them; threads awaiting my reply are invisible once they scroll past. → **Desired state:** a daily (configurable) pass detects stalled commitments + awaiting-reply threads, dedups against prior nudges, and delivers ONE governed digest to the home channel.

## Metadata
- **Complexity**: Large (new detector module + 1 state table + config + CLI + tests; heavy reuse of shipped infra)
- **Source PRD**: N/A (research idea #3 from this session's shortlist)
- **PRD Phase**: N/A
- **Estimated Files**: ~8 (1 new detector module, `hermes_state.py`, `hermes_cli/config.py`, `cli-config.yaml.example`, `hermes_cli/subcommands/notify.py`, `hermes_cli/notify_cmd.py`, `hermes_cli/main.py`, 1 test file)

---

## UX Design

### After
```
┌──────────────────────────────────────────────────────────────┐
│ scheduled scan (opt-in) OR `hermes threads scan`               │
│   SOURCE A (primary, reliable): open kanban cards              │
│     past `Due:` (parsed from body) or untouched > staleness_h  │
│   SOURCE B (secondary, best-effort): live gateway threads      │
│     whose last active message is from someone-not-the-bot      │
│     and quiet > staleness_h  (heuristic — see NOT Building)     │
│        ↓                                                        │
│   aux LLM classifies each candidate open-vs-resolved +         │
│     writes a one-line "what's owed & to whom"                  │
│        ↓                                                        │
│   dedup vs stalled_nudges table (cooldown window)              │
│        ↓                                                        │
│   ONE digest → should_deliver("stalled_thread") → home channel │
│     "2 open loops: (1) Send the deck to Sarah — due Mon;       │
│      (2) @josh has waited 3d for your reply in #proj"          │
└──────────────────────────────────────────────────────────────┘
```

### Interaction Changes
| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Open commitments | filed, never resurfaced | nudged once past due/stale | governed by budget |
| Awaiting-reply threads | invisible | best-effort nudge | secondary, heuristic |
| Control | none | `hermes threads {scan,enable,disable,list}`; `hermes notify mute stalled_thread` | reuses governor feedback |
| Cadence | none | opt-in cron (default off) | consent |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `agent/omi_commitments.py` | 1-340 | The scan module to mirror EXACTLY (structure, opt-in gate, aux-LLM, dedup, governor notify, connection lifecycle) |
| P0 | `agent/notification_budget.py` | `should_deliver` signature | Route the digest through this; category `stalled_thread` |
| P0 | `hermes_cli/kanban_db.py` | 839-857, 1096-1180, 2706-2774, 8709-8752 | `Task` dataclass, `tasks` schema (NO due col), `list_tasks`/`get_task`, `task_age`+`_to_epoch` helpers |
| P0 | `hermes_state.py` | 800-821, 750-798, 2099-2135, 4947-4999 | `messages`/`sessions` schema, `list_gateway_sessions`, `list_recent_user_messages` |
| P0 | `hermes_state.py` | 745-894 (SCHEMA_SQL), 1249-1301 (`_execute_write`), 6407-6425 (`set/get_meta`) | Where to add the new table + write/read patterns |
| P1 | `hermes_cli/subcommands/notify.py` | all | `build_notify_parser`/`build_omi_parser` to mirror for `build_threads_parser` |
| P1 | `hermes_cli/notify_cmd.py` | all | `omi_command`/`_omi_scan`/`_omi_set_enabled` to mirror for `threads_command` |
| P1 | `hermes_cli/config.py` | 976-990, `omi_commitments` section, `_KNOWN_ROOT_KEYS` | Add `stalled_threads` section |
| P1 | `gateway/run.py` | 867-889 | How `observed=1` group messages are treated (context, not addressed turns) |
| P2 | `tests/agent/test_omi_commitments.py` | all | Test structure to mirror (mock aux LLM, autouse HERMES_HOME, tmp kanban) |

## External Documentation
None — feature uses only established internal patterns. No external research needed.

---

## Patterns to Mirror

### SCAN_MODULE_SHAPE (mirror omi_commitments.py)
```python
// SOURCE: agent/omi_commitments.py:184-217
def run_omi_commitment_scan() -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg.get("enabled", False):
        return {"skipped": "disabled"}
    ...
    db = SessionDB(get_hermes_home() / "state.db")
    ...
    conn = kb.connect(board=board)
    try:
        ...
    finally:
        conn.close()
```

### STATE_TABLE_DDL (add to SCHEMA_SQL; bare CREATE, no SCHEMA_VERSION bump)
```python
// SOURCE: hermes_state.py:843-846 (state_meta shape to mirror)
CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

### STATE_WRITE / STATE_READ
```python
// SOURCE: hermes_state.py:6417-6425
def set_meta(self, key: str, value: str) -> None:
    def _do(conn):
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    self._execute_write(_do)
```
```python
// SOURCE: hermes_state.py:2932-2939
def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
    with self._lock:
        cursor = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
    return dict(row) if row else None
```

### KANBAN_OPEN_CARDS (fetch-all-then-filter; status filter is single-value only)
```python
// SOURCE: tests/agent/test_omi_commitments.py:49-56 + kanban_db.py:2725
from hermes_cli import kanban_db as kb
conn = kb.connect(board=board or None)
try:
    open_cards = [
        t for t in kb.list_tasks(conn, include_archived=True)
        if t.status in {"triage", "todo", "ready", "scheduled", "blocked"}
    ]  # exclude done/archived/running/review
finally:
    conn.close()
```

### KANBAN_AGE + DUE_PARSE (reuse _to_epoch; handle 'Due: none' sentinel)
```python
// SOURCE: hermes_cli/kanban_db.py:8709-8752 (task_age, _to_epoch — REUSE, do not reinvent)
from hermes_cli.kanban_db import task_age, _to_epoch
age = task_age(card)["created_age_seconds"]        # untouched-too-long, seconds
# past-due: parse body's first "Due: <iso|none>" line, then _to_epoch(value)
```

### LIVE_THREADS_QUERY (secondary source; last message via id DESC, active=1)
```sql
-- SOURCE: hermes_state.py:2112-2132 (live threads, most-recent activity first)
SELECT sessions.*,
       COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = sessions.id),
                sessions.started_at) AS last_active
FROM sessions
WHERE session_key IS NOT NULL AND ended_at IS NULL
  AND started_at = (SELECT MAX(s2.started_at) FROM sessions s2 WHERE s2.session_key = sessions.session_key)
ORDER BY last_active DESC
```
```sql
-- SOURCE: hermes_state.py:4964-4968 (the LAST message of a session — order by id, active only)
SELECT id, role, content, timestamp, observed FROM messages
WHERE session_id = ? AND active = 1
ORDER BY id DESC LIMIT 1
```

### AUX_LLM_CLASSIFY
```python
// SOURCE: agent/omi_commitments.py:82-113 (get_text_auxiliary_client + strict-JSON prompt + defensive parse)
client, model = get_text_auxiliary_client("stalled_thread")
if client is None or not model:
    return []
resp = client.chat.completions.create(model=model, messages=[...], temperature=0,
    max_tokens=1024, extra_body=get_auxiliary_extra_body() or None)
raw = resp.choices[0].message.content or ""
```

### GOVERNOR_NOTIFY (batched digest, one candidate_id for the whole digest)
```python
// SOURCE: agent/omi_commitments.py:282-330 (_notify → should_deliver → send_message_tool)
from agent.notification_budget import should_deliver
decision = should_deliver(category="stalled_thread", value_hint=<0..1>,
                          candidate_id=f"stalled:{day_key}")   # one digest/day
if decision.allow:
    from tools.send_message_tool import send_message_tool
    send_message_tool({"message": digest})
```

### CLI_PARSER + HANDLER (mirror omi)
```python
// SOURCE: hermes_cli/subcommands/notify.py (build_omi_parser) + hermes_cli/notify_cmd.py (omi_command)
def build_threads_parser(subparsers, *, cmd_threads):
    p = subparsers.add_parser("threads", help="...")
    sub = p.add_subparsers(dest="threads_action")
    sub.add_parser("scan"); sub.add_parser("enable"); sub.add_parser("disable"); sub.add_parser("list")
    p.set_defaults(func=cmd_threads)
```

### TEST_STRUCTURE
```python
// SOURCE: tests/agent/test_omi_commitments.py:14-46 (cfg_patch fixture, autouse HERMES_HOME via conftest)
@pytest.fixture
def cfg_patch():
    def _apply(**o): return patch.object(st, "_cfg", return_value=_cfg(**o))
    return _apply
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `agent/stalled_threads.py` | CREATE | Detector: `run_stalled_thread_scan()`, kanban + thread sources, aux-LLM classify, dedup, governed digest |
| `hermes_state.py` | UPDATE | Add `stalled_nudges` table to SCHEMA_SQL + CRUD (`record_stall_nudge`, `stall_nudged_recently`, `list_live_threads_for_stall`) |
| `hermes_cli/config.py` | UPDATE | Add `stalled_threads` section to DEFAULT_CONFIG + `_KNOWN_ROOT_KEYS` |
| `cli-config.yaml.example` | UPDATE | Document `stalled_threads` |
| `hermes_cli/subcommands/notify.py` | UPDATE | Add `build_threads_parser` |
| `hermes_cli/notify_cmd.py` | UPDATE | Add `threads_command` + `_threads_{scan,list,set_enabled}` |
| `hermes_cli/main.py` | UPDATE | `cmd_threads` wrapper + `build_threads_parser` call + `"threads"` in the 2 command allow-lists |
| `tests/agent/test_stalled_threads.py` | CREATE | Unit tests |

## NOT Building
- **Structured group-chat sender identity.** The `messages` table has none — third party and owner are both `role="user"`; only `observed=1` / in-content `[name|id]` prefixes distinguish them. Source B is therefore **best-effort heuristic**: a thread is a candidate only when its last active message is `role="user"` AND quiet > staleness. The aux LLM makes the final open-vs-resolved call. We do NOT attempt reliable "who owes whom" from columns.
- **Reading external platform APIs** (Slack unread counts, etc.). Only the local `messages`/`sessions` tables + kanban DB.
- **Per-item pings.** One batched digest per scan (cap via governor anyway).
- **New kanban due-date column.** Due date is parsed from the card body `Due:` line (owning the `Due: none` sentinel), consistent with how Omi writes it.
- **Auto-resolving/closing cards.** Detection + nudge only; the human acts.
- **Gating cron/webhook/goal** (already decided in the governor feature — only genuinely agent-initiated proactivity is gated).

---

## Data Model

One new table in `hermes_state.py` SCHEMA_SQL (bare CREATE — no `SCHEMA_VERSION` bump). Synthetic values shown.

```sql
CREATE TABLE IF NOT EXISTS stalled_nudges (
    candidate_id   TEXT PRIMARY KEY,   -- 'card:<task_id>' or 'thread:<session_key>'
    kind           TEXT NOT NULL,      -- 'commitment' | 'thread'
    last_nudged_at REAL NOT NULL,      -- epoch float
    nudge_count    INTEGER NOT NULL DEFAULT 1,
    summary        TEXT                -- last one-line "what's owed" (for digest/debug)
);
CREATE INDEX IF NOT EXISTS idx_stalled_nudges_time ON stalled_nudges(last_nudged_at);
```
`SessionDB` methods: `stall_nudged_recently(candidate_id, cooldown_seconds) -> bool`, `record_stall_nudge(candidate_id, kind, summary) -> None`, and `list_live_threads_for_stall(cutoff_epoch, exclude_sources) -> List[dict]` (wraps LIVE_THREADS_QUERY + last-message lookup).

### Config
```python
"stalled_threads": {
    "enabled": False,             # OPT-IN (consent)
    "scan_interval_hours": 12,
    "staleness_hours": 48,        # quiet this long → candidate
    "cooldown_hours": 72,         # don't re-nudge same item within this window
    "lookback_hours": 336,        # 14d — ignore threads/cards older than this
    "min_confidence": 0.6,        # aux-LLM open-vs-resolved threshold
    "scan_threads": True,         # source B (best-effort); False = commitments only
    "max_items_per_digest": 5,
    "board": "",
    "exclude_sources": ["tool", "tui"],   # not real human conversations
},
```

---

## Detection Semantics
1. Opt-in gate: `not cfg.enabled` → `{"skipped": "disabled"}`.
2. **Source A (commitments, primary):** open cards (`status in {triage,todo,ready,scheduled,blocked}`, exclude done/archived). For each: `due = _to_epoch(parse_due(body))` (skip `Due: none`); stale if `due and due < now` (past due) OR `task_age.created_age_seconds > staleness_hours*3600` (untouched). Skip if older than lookback.
3. **Source B (threads, best-effort, if `scan_threads`):** `list_live_threads_for_stall(now - lookback)`; keep threads whose LAST active message is `role="user"` and `last_active < now - staleness_hours*3600`. (Owner-vs-third-party left to the aux LLM.)
4. **Classify:** batch candidates to the aux LLM (`stalled_thread` task) → strict JSON `[{candidate_id, owed_summary, still_open: bool, confidence}]`. Keep `still_open and confidence >= min_confidence`.
5. **Dedup:** drop any candidate where `stall_nudged_recently(candidate_id, cooldown_hours*3600)`.
6. **Digest:** take top `max_items_per_digest`; build one message; `should_deliver("stalled_thread", candidate_id=f"stalled:{day_key}", value_hint=<max confidence>)`. If allowed, send via `send_message_tool`, then `record_stall_nudge(...)` for each included item.
7. Return summary dict `{scanned, candidates, nudged, delivered}`. Fail-soft throughout.

---

## Step-by-Step Tasks

### Task 1: State table + CRUD (`hermes_state.py`)
- **ACTION**: Add `stalled_nudges` table + index to SCHEMA_SQL; add the 3 methods.
- **IMPLEMENT**: `stall_nudged_recently` (READ: `SELECT last_nudged_at ... WHERE candidate_id=?`, compare to `time.time()-cooldown`); `record_stall_nudge` (WRITE upsert incrementing `nudge_count`); `list_live_threads_for_stall` (READ: LIVE_THREADS_QUERY + per-thread last-message via `ORDER BY id DESC LIMIT 1`, returning dicts with source/chat_id/thread_id/session_key/last_active/last_role/last_observed/last_content).
- **MIRROR**: STATE_WRITE (`set_meta`), STATE_READ (`get_session`), LIVE_THREADS_QUERY.
- **IMPORTS**: existing (`time`, `json`).
- **GOTCHA**: message/session timestamps are **epoch float** (REAL) — compare with `time.time()-N*3600`, never ISO. Order last-message by **`id` DESC**, not timestamp (clock non-monotonic). Filter `active=1`. Live thread = `ended_at IS NULL AND session_key IS NOT NULL`. Use `COALESCE(thread_id,'')`. Exclude `LOWER(source) IN ('tool','tui')`.
- **VALIDATE**: smoke: insert a nudge, assert `stall_nudged_recently` True within window / False after; `list_live_threads_for_stall(0)` returns rows with a `last_role` key.

### Task 2: Config section (`hermes_cli/config.py` + example)
- **ACTION**: Add `stalled_threads` to DEFAULT_CONFIG (after `omi_commitments`) + `_KNOWN_ROOT_KEYS`; document in `cli-config.yaml.example`.
- **MIRROR**: the `omi_commitments` section.
- **GOTCHA**: `_deep_merge` ignores `None`-over-dict; new keys auto-load.
- **VALIDATE**: `python -c "from hermes_cli.config import load_config as l; print(l()['stalled_threads']['enabled'])"` → `False`.

### Task 3: Detector module (`agent/stalled_threads.py`)
- **ACTION**: Create `run_stalled_thread_scan() -> dict` + helpers `_parse_due(body)`, `_gather_commitment_candidates(conn, cfg, now)`, `_gather_thread_candidates(db, cfg, now)`, `_classify(candidates)`, `_deliver_digest(items)`.
- **IMPLEMENT**: per Detection Semantics. `_parse_due`: first line matching `^Due:\s*(.+)$`; if value `== "none"` → None; else `_to_epoch(value)`.
- **MIRROR**: SCAN_MODULE_SHAPE, KANBAN_OPEN_CARDS, KANBAN_AGE + DUE_PARSE, AUX_LLM_CLASSIFY, GOVERNOR_NOTIFY.
- **IMPORTS**: `hashlib, json, logging, re, time`; `from datetime import datetime`; `from hermes_cli.config import load_config`; `from hermes_cli import kanban_db as kb`; `from hermes_cli.kanban_db import task_age, _to_epoch`; `from agent.auxiliary_client import get_text_auxiliary_client, get_auxiliary_extra_body`; `from hermes_state import SessionDB`; `from hermes_constants import get_hermes_home`.
- **GOTCHA**: NO MCP needed → do NOT copy `_ensure_mcp_ready` (Sources A/B are local DBs). Resolve state DB via `get_hermes_home()/"state.db"` at call time (profile/test-safe — the Omi lesson). Kanban timestamps are epoch **int**; message/session timestamps are epoch **float** — never subtract across them. Fail-soft: any source that throws contributes zero, scan still returns a summary. `_parse_due` must handle date-only / tz-less / `Z` forms (that's why reuse `_to_epoch`).
- **VALIDATE**: unit tests (Task 6).

### Task 4: CLI (`subcommands/notify.py` + `notify_cmd.py` + `main.py`)
- **ACTION**: `build_threads_parser` (scan/enable/disable/list); `threads_command` dispatch + `_threads_scan` (calls `run_stalled_thread_scan`, prints summary), `_threads_list` (prints open candidates without nudging — a dry-run), `_threads_set_enabled` (reuse the atomic `read_raw_config`+`atomic_config_write` pattern from `_omi_set_enabled`); `cmd_threads` wrapper + builder call + `"threads"` in both allow-lists.
- **MIRROR**: the `omi`/`build_omi_parser`/`_omi_set_enabled` trio.
- **GOTCHA**: `_omi_set_enabled` already uses `atomic_config_write` (the post-review fix) — copy that, not a raw dump.
- **VALIDATE**: `hermes threads list` runs on a fresh profile; `hermes threads enable` writes `stalled_threads.enabled: true`.

### Task 5: Register opt-in cron (guidance)
- **ACTION**: `threads enable` prints `hermes cron create 'every 12 hours' --name stalled-thread-scan --no-agent --script threads_scan.py` guidance (mirror `_register_omi_job`, corrected to real `hermes cron create` syntax).
- **MIRROR**: `notify_cmd._register_omi_job`.
- **GOTCHA**: user-scheduled, not auto-registered (visible in `hermes cron list`).
- **VALIDATE**: guidance text uses the real positional-schedule cron syntax.

### Task 6: Tests (`tests/agent/test_stalled_threads.py`)
- **ACTION**: Unit-test both sources, dedup, governor routing, consent, `_parse_due`.
- **IMPLEMENT**: mock `get_text_auxiliary_client`/classify; autouse HERMES_HOME (conftest); seed a tmp kanban card with `Due:` in body + an old `created_at`; stub `should_deliver`.
- **MIRROR**: TEST_STRUCTURE from `test_omi_commitments.py`.
- **VALIDATE**: `python -m pytest tests/agent/test_stalled_threads.py -q`.

---

## Testing Strategy

### Unit Tests
| Test | Input | Expected | Edge? |
|---|---|---|---|
| disabled → skip | `enabled=False` | `{"skipped":"disabled"}`, no DB read | ✓ |
| past-due card detected | card body `Due: <yesterday>`, open status | candidate | |
| `Due: none` ignored | body `Due: none`, fresh card | not past-due (age-only) | ✓ |
| untouched card | open card, `created_at` > staleness ago | candidate | |
| done/archived excluded | card status done | not a candidate | ✓ |
| thread awaiting reply | last active msg role=user, quiet > staleness | candidate (source B) | |
| thread bot-spoke-last | last msg role=assistant | not a candidate | ✓ |
| resolved by LLM | classify still_open=false | dropped | ✓ |
| low confidence dropped | confidence < min | dropped | ✓ |
| dedup within cooldown | same candidate_id nudged 1h ago, cooldown 72h | dropped | ✓ |
| re-nudge after cooldown | last nudge > cooldown ago | included | ✓ |
| digest batched | 3 candidates | ONE should_deliver call | ✓ |
| governor suppresses | should_deliver allow=False | delivered=0, no send, no record_stall_nudge | ✓ |
| `_parse_due` variants | date-only / tz / Z / `none` / missing | correct epoch or None | ✓ |
| source throws | kanban connect raises | scan returns summary, threads still scanned | ✓ |

### Edge Cases Checklist
- [x] Empty (no open cards, no live threads)
- [x] Max (`max_items_per_digest` honored)
- [x] Invalid types (malformed `Due:`, non-JSON LLM output)
- [x] Concurrent access (writes via `_execute_write`)
- [x] Permission/disabled (consent gate)
- [x] Timestamp format (epoch-int kanban vs epoch-float messages — not crossed)

---

## Validation Commands

### Static Analysis
```bash
cd /home/jtomek/.hermes/hermes-agent
python -m ruff check agent/stalled_threads.py hermes_state.py hermes_cli/config.py \
  hermes_cli/notify_cmd.py hermes_cli/subcommands/notify.py hermes_cli/main.py \
  tests/agent/test_stalled_threads.py
```
EXPECT: All checks passed.

### Unit Tests
```bash
python -m pytest tests/agent/test_stalled_threads.py -q -p no:cacheprovider
```
EXPECT: all pass. (Runner: use the active nix `python -m pytest`, NOT `scripts/run_tests.sh` which targets venv without pytest.)

### Config + schema smoke
```bash
python -c "from hermes_cli.config import load_config as l; print(l()['stalled_threads'])"
python -c "import tempfile,pathlib; from hermes_state import SessionDB; d=SessionDB(pathlib.Path(tempfile.mkdtemp())/'s.db'); print([r[0] for r in d._conn.execute(\"select name from sqlite_master where name='stalled_nudges'\")])"
```

### Regression
```bash
python -m pytest tests/agent/ tests/gateway/test_delivery.py -q -p no:cacheprovider
```
EXPECT: no regressions.

### LIVE DATA VERIFICATION (mandatory — the Omi lesson)
Before trusting the scan, verify the REAL data shapes and standalone execution context:
```bash
# 1. Confirm live-thread rows have readable last-message role + epoch-float last_active on THIS profile's real data:
python -c "
from hermes_state import SessionDB; from hermes_constants import get_hermes_home
db=SessionDB(get_hermes_home()/'state.db')
rows=db.list_live_threads_for_stall(0, exclude_sources=['tool','tui'])
print('live threads:', len(rows))
[print(r.get('source'), r.get('last_role'), r.get('last_observed'), str(r.get('last_active'))[:16]) for r in rows[:5]]
"
# 2. Confirm real kanban cards + the Due-line format on actual Omi-created cards:
python -c "
from hermes_cli import kanban_db as kb
conn=kb.connect()
[print(t.status, repr((t.body or '').splitlines()[0] if t.body else '')) for t in kb.list_tasks(conn, include_archived=True)[:5]]; conn.close()
"
# 3. Run the real scan standalone (no gateway) and confirm it does NOT silently return 0 due to a context/format bug:
hermes threads enable && hermes threads scan
hermes threads list        # dry-run view of candidates
```
EXPECT: real threads/cards enumerated with correct roles + parsed due dates; `scan` reports candidates that match what you know is actually open (or a clear, correct "0 candidates").

### Manual Validation
- [ ] `hermes threads list` shows real open commitments from your kanban board
- [ ] A past-due Omi card appears as a candidate
- [ ] Running scan twice within cooldown produces only ONE nudge
- [ ] `hermes notify mute stalled_thread` then re-scan → more suppression
- [ ] `hermes threads disable` removes it from future scans

---

## Acceptance Criteria
- [ ] All tasks complete; validation green; no regressions
- [ ] Opt-in (default disabled); consent honored
- [ ] Source A (commitments) reliable; Source B (threads) clearly bounded as best-effort
- [ ] Every nudge routed through the governor (category `stalled_thread`); one batched digest
- [ ] Dedup via cooldown; no re-nudge within window
- [ ] **Live-data verification step executed** (not just mocked tests)
- [ ] No type/lint errors; no hardcoded values

## Completion Checklist
- [ ] Mirrors omi_commitments structure (scan/dedup/governor-notify/CLI)
- [ ] Fail-soft everywhere; epoch-float vs epoch-int never crossed
- [ ] `_parse_due` handles `none`/date-only/tz/Z
- [ ] Tests + live verification both done
- [ ] `cli-config.yaml.example` documents the section
- [ ] Self-contained — no codebase searching needed

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Source B false positives (owner vs third-party unresolvable from columns) | High | Med | Best-effort + aux-LLM gate + `scan_threads` toggle to disable entirely; documented NOT-building |
| Body `Due:` parse fragility | Med | Low | Reuse `_to_epoch`; own the `Due: none` sentinel; unit tests for all forms |
| epoch-int (kanban) vs epoch-float (messages) confusion | Med | Med | Explicit gotcha; never subtract across sources; tests pin both |
| Nagging despite governor | Low | High | Single digest + governor cap/ceiling + cooldown dedup + mute feedback |
| Standalone scan silent-zero (Omi-style) | Med | Med | Mandatory live-verification step; no MCP dependency here reduces surface |
| Clock non-monotonic ordering | Low | Low | Order by `id` for last-message, per core convention |

## Notes
- **Reuse ratio is high:** governor, router gate, CLI trio, config pattern, and scan skeleton all exist. Net-new is one table + `agent/stalled_threads.py` detection logic. Large-but-fast.
- **Source A is the value.** Commitments (incl. every Omi card) are structured and reliable — that alone delivers the "you said you'd send the deck" behavior. Source B is a bonus that degrades gracefully (toggle off).
- **Omi live-test lessons baked in:** dynamic DB-path resolution, a mandatory live-data verification step, and an explicit standalone-execution check — the three bug classes mocked tests missed last time.
- Natural follow-on: a `morning brief` (research #7) could aggregate stalled-threads + omi + calendar into one daily card digest, reusing this same producer + governor.
```
