# MemPalace Sync Daemon — Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.
> **Related design doc:** `2026-04-19-mempalace-sync-daemon-design.md`

**Goal:** Build a standalone `mempalace-sync` daemon that incrementally mirrors Hermes `state.db` conversation history into MemPalace, enabling semantic/entity/temporal recall via MCP without touching Hermes itself.

**Architecture:** Independent repo `~/repo/jethro/mempalace-sync/`. Read-only SQLite poll of `~/.hermes/state.db` → export new messages as JSONL → invoke MemPalace `convo_miner` → advance cursor. Hermes changes = 0 LOC (only `~/.hermes/config.yaml` gets ~10 lines for MCP pass-through).

**Tech Stack:**
- Python 3.11, `sqlite3` (stdlib), `mempalace` (PyPI ≥ 3.3.1)
- `pytest` for TDD
- `uv` for env mgmt
- `fcntl.flock` for cross-process lock
- systemd unit for production; cron for simpler setups

---

## Phase 0: Pre-flight Checks (Must Pass Before Phase 1)

### Task 0.1: Verify zh-TW recall with default embedder ✅ DONE (2026-04-19)

**Result:** Ran against real `~/.hermes/state.db` (263 msgs / 17 sessions).
zh-TW: 8/8, English: 4/4, Overall: 12/12. Mine time: 5.2s.
Mean top-1 score 0.345 (low but hybrid BM25 + vector rescues recall).

**Decision:** Default Chroma embedder (`all-MiniLM-L6-v2`) is good enough for
dogfood — no need to patch `ChromaBackend` to swap embedders.
`MEMPALACE_EMBED_MODEL` env var does NOT exist — plan revised to remove it.

Artifact: `~/tmp/mempalace-spike/phase0_scale.py`

**Objective (original, now moot):** Verify `paraphrase-multilingual-MiniLM-L12-v2` recovers Chinese recall before any code is written.

**Files:**
- Create: `~/tmp/mempalace-spike/test_multilingual.py` (throwaway)

**Step 1: Create test script**

```python
# ~/tmp/mempalace-spike/test_multilingual.py
"""Test whether multilingual embedder recovers zh-TW recall."""
import subprocess, tempfile, json, shutil
from pathlib import Path

PALACE = Path.home() / ".mempalace" / "test-zh"
if PALACE.exists():
    shutil.rmtree(PALACE)

sample = [
    {"ts": "2025-01-01T00:00:00Z", "role": "user",
     "content": "我想把 postgres 從 13 升到 16，資料庫遷移要怎麼做？"},
    {"ts": "2025-01-01T00:01:00Z", "role": "assistant",
     "content": "可以用 pg_dumpall 然後 pg_restore..."},
]
tmp = Path(tempfile.mkdtemp())
(tmp / "session.jsonl").write_text("\n".join(json.dumps(m) for m in sample))

env = {"MEMPALACE_EMBED_MODEL": "paraphrase-multilingual-MiniLM-L12-v2"}
subprocess.run(
    ["mempalace", "convo-mine", str(tmp), "--palace", str(PALACE)],
    check=True, env={**__import__("os").environ, **env},
)
result = subprocess.run(
    ["mempalace", "search", "資料庫遷移", "--palace", str(PALACE), "--json"],
    capture_output=True, text=True, env={**__import__("os").environ, **env},
)
print(result.stdout)
hits = json.loads(result.stdout)
assert any("postgres" in h.get("content", "") for h in hits), "zh-TW recall failed!"
print("✓ Multilingual recall works")
```

**Step 2: Run it**

```bash
cd ~/tmp/mempalace-spike && source venv/bin/activate
python test_multilingual.py
```

Expected: `✓ Multilingual recall works`

**Step 3: Decision gate**

- If ✅ passes → proceed to Phase 1
- If ❌ fails → **STOP**. Options: (a) upstream PR to MemPalace adding `--embed-model`, (b) swap to a different vector lib, (c) cancel the whole plan and stick with FTS5-only

### Task 0.2: Confirm ingest CLI ✅ DONE (2026-04-19)

**Result:** The actual command is `mempalace --palace <path> mine <dir> --mode convos`
(global `--palace` flag BEFORE the subcommand, `convos` mode switch, no separate
`convo-mine` subcommand). All downstream tasks updated.

---

## Phase 1: Scaffold Repo

### Task 1.1: Create repo + pyproject.toml

**Objective:** Fresh repo with uv + pytest + ruff.

**Files:**
- Create: `~/repo/jethro/mempalace-sync/pyproject.toml`
- Create: `~/repo/jethro/mempalace-sync/.gitignore`
- Create: `~/repo/jethro/mempalace-sync/README.md`

**Step 1: Init**

```bash
mkdir -p ~/repo/jethro/mempalace-sync
cd ~/repo/jethro/mempalace-sync
git init
```

**Step 2: Write `pyproject.toml`**

```toml
[project]
name = "mempalace-sync"
version = "0.1.0"
description = "Sync Hermes state.db into MemPalace incrementally"
requires-python = ">=3.11"
dependencies = [
    "mempalace>=3.3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
]

[project.scripts]
mempalace-sync = "mempalace_sync.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

**Step 3: `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.coverage
```

**Step 4: Bootstrap venv**

```bash
uv venv
uv pip install -e ".[dev]"
```

Expected: installs mempalace + pytest without errors.

**Step 5: Commit**

```bash
git add .
git commit -m "chore: scaffold mempalace-sync project"
```

### Task 1.2: Package skeleton

**Files:**
- Create: `~/repo/jethro/mempalace-sync/src/mempalace_sync/__init__.py`
- Create: `~/repo/jethro/mempalace-sync/src/mempalace_sync/cli.py` (stub)
- Create: `~/repo/jethro/mempalace-sync/tests/__init__.py`
- Create: `~/repo/jethro/mempalace-sync/tests/conftest.py`

**Step 1: `__init__.py`**

```python
"""mempalace-sync — daemon that mirrors Hermes state.db into MemPalace."""
__version__ = "0.1.0"
```

**Step 2: `cli.py` stub**

```python
"""CLI dispatcher (filled in Phase 4)."""
import sys

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    print("mempalace-sync stub; implement in Phase 4")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: `tests/conftest.py`**

```python
import sqlite3, tempfile, shutil, os
from pathlib import Path
import pytest

# Copy the Hermes schema verbatim so tests mirror production exactly.
HERMES_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT, billing_base_url TEXT, billing_mode TEXT,
    estimated_cost_usd REAL, actual_cost_usd REAL,
    cost_status TEXT, cost_source TEXT, pricing_version TEXT,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT
);
"""

@pytest.fixture
def fake_state_db(tmp_path):
    """Returns path to a fresh SQLite file matching Hermes schema."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(HERMES_SCHEMA)
    conn.commit()
    conn.close()
    return db

@pytest.fixture
def seed_session(fake_state_db):
    """Inserts one session with N messages. Returns (session_id, message_ids)."""
    def _seed(n: int = 3, session_id: str = "test-001", source: str = "cli"):
        conn = sqlite3.connect(fake_state_db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            (session_id, source, 1745000000.0),
        )
        ids = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, f"msg {i} content", 1745000000.0 + i),
            )
            ids.append(cur.lastrowid)
        conn.commit()
        conn.close()
        return session_id, ids
    return _seed
```

**Step 4: Verify**

```bash
cd ~/repo/jethro/mempalace-sync
pytest
```

Expected: `no tests ran` (no tests yet, but collection works).

**Step 5: Commit**

```bash
git add .
git commit -m "chore: package skeleton + test fixtures"
```

---

## Phase 2: Core Reader (read-only, no writes)

### Task 2.1: `StateReader.query_new_messages` — TDD

**Objective:** Read-only function that returns all `messages` with `id > cursor`, joined with session metadata.

**Files:**
- Create: `tests/test_state_reader.py`
- Create: `src/mempalace_sync/state_reader.py`

**Step 1: Write failing test**

```python
# tests/test_state_reader.py
from mempalace_sync.state_reader import StateReader

def test_query_new_messages_returns_all_when_cursor_zero(seed_session, fake_state_db):
    sid, ids = seed_session(n=3)
    reader = StateReader(fake_state_db)
    msgs = reader.query_new_messages(since_id=0)
    assert len(msgs) == 3
    assert [m["id"] for m in msgs] == ids
    assert all(m["session_id"] == sid for m in msgs)
    assert msgs[0]["role"] == "user"
    assert msgs[0]["source"] == "cli"

def test_query_new_messages_respects_cursor(seed_session, fake_state_db):
    sid, ids = seed_session(n=5)
    reader = StateReader(fake_state_db)
    msgs = reader.query_new_messages(since_id=ids[2])  # skip first 3
    assert [m["id"] for m in msgs] == ids[3:]

def test_query_new_messages_opens_readonly(fake_state_db):
    reader = StateReader(fake_state_db)
    # Should not acquire write lock — can run alongside writer
    msgs = reader.query_new_messages(since_id=0)
    assert msgs == []
```

**Step 2: Verify failure**

```bash
pytest tests/test_state_reader.py -v
```

Expected: `ModuleNotFoundError: mempalace_sync.state_reader`

**Step 3: Minimal implementation**

```python
# src/mempalace_sync/state_reader.py
"""Read-only view over Hermes state.db."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any

class StateReader:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        # mode=ro ensures we never accidentally write
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def query_new_messages(self, since_id: int, limit: int = 10_000) -> list[dict[str, Any]]:
        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.tool_name,
                   m.timestamp, s.source, s.model
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.id > ?
            ORDER BY m.id ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (since_id, limit)).fetchall()
        return [dict(r) for r in rows]
```

**Step 4: Verify pass**

```bash
pytest tests/test_state_reader.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add .
git commit -m "feat: StateReader.query_new_messages (read-only)"
```

### Task 2.2: Handle SQLite busy / locked gracefully

**Step 1: Failing test**

```python
# tests/test_state_reader.py (add)
import pytest
from mempalace_sync.state_reader import StateReader, StateReadError

def test_raises_on_missing_db(tmp_path):
    reader = StateReader(tmp_path / "nope.db")
    with pytest.raises(StateReadError):
        reader.query_new_messages(0)
```

**Step 2: Run → fail, then add exception class + try/except in `_connect()`.**

```python
# state_reader.py (append)
class StateReadError(RuntimeError):
    pass

# Wrap _connect:
    def _connect(self):
        try:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise StateReadError(f"Cannot open {self.db_path}: {e}") from e
```

**Step 3: Run → pass, commit.**

```bash
git add . && git commit -m "feat: StateReader error handling"
```

---

## Phase 3: JSONL Exporter

### Task 3.1: `export_messages_to_jsonl` — TDD

**Objective:** Convert a list of message dicts (from StateReader) into MemPalace-compatible JSONL, one file per session.

**Files:**
- Create: `tests/test_exporter.py`
- Create: `src/mempalace_sync/exporter.py`

**Step 1: Failing test**

```python
# tests/test_exporter.py
import json
from mempalace_sync.exporter import export_messages_to_jsonl

def test_groups_by_session_one_file_each(tmp_path):
    msgs = [
        {"id": 1, "session_id": "A", "role": "user", "content": "hello",
         "timestamp": 1745000000.0, "source": "cli", "model": "opus", "tool_name": None},
        {"id": 2, "session_id": "A", "role": "assistant", "content": "hi",
         "timestamp": 1745000001.0, "source": "cli", "model": "opus", "tool_name": None},
        {"id": 3, "session_id": "B", "role": "user", "content": "other",
         "timestamp": 1745000002.0, "source": "telegram", "model": "opus", "tool_name": None},
    ]
    files = export_messages_to_jsonl(msgs, tmp_path)
    assert len(files) == 2
    assert (tmp_path / "A.jsonl").exists()
    assert (tmp_path / "B.jsonl").exists()
    lines_a = (tmp_path / "A.jsonl").read_text().strip().splitlines()
    assert len(lines_a) == 2
    first = json.loads(lines_a[0])
    assert first["role"] == "user"
    assert first["content"] == "hello"
    assert first["session_id"] == "A"
    assert "ts" in first  # ISO-8601 string

def test_appends_on_rerun(tmp_path):
    msgs_first = [{"id": 1, "session_id": "A", "role": "user", "content": "a",
                   "timestamp": 1745000000.0, "source": "cli", "model": "opus", "tool_name": None}]
    msgs_second = [{"id": 2, "session_id": "A", "role": "user", "content": "b",
                    "timestamp": 1745000001.0, "source": "cli", "model": "opus", "tool_name": None}]
    export_messages_to_jsonl(msgs_first, tmp_path)
    export_messages_to_jsonl(msgs_second, tmp_path)
    lines = (tmp_path / "A.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

def test_skips_empty_content(tmp_path):
    msgs = [{"id": 1, "session_id": "A", "role": "tool", "content": None,
             "timestamp": 1745000000.0, "source": "cli", "model": "opus", "tool_name": "terminal"}]
    files = export_messages_to_jsonl(msgs, tmp_path)
    assert files == []  # nothing worth ingesting
```

**Step 2: Run → fail.**

**Step 3: Implementation**

```python
# src/mempalace_sync/exporter.py
"""Convert Hermes messages to MemPalace-compatible JSONL."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def export_messages_to_jsonl(messages: Iterable[dict], out_dir: Path) -> list[Path]:
    """Append messages into per-session JSONL files.

    Returns list of files touched (may be empty if all messages had empty content).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_session: dict[str, list[dict]] = {}
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue  # skip tool-only / empty rows; palace doesn't need them
        by_session.setdefault(m["session_id"], []).append({
            "ts": _ts_to_iso(m["timestamp"]),
            "role": m["role"],
            "content": content,
            "session_id": m["session_id"],
            "source": m.get("source"),
            "model": m.get("model"),
        })

    touched = []
    for sid, entries in by_session.items():
        fp = out_dir / f"{sid}.jsonl"
        with fp.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        touched.append(fp)
    return touched
```

**Step 4: Run → pass.**

**Step 5: Commit.**

```bash
git add . && git commit -m "feat: JSONL exporter grouped by session"
```

---

## Phase 4: State File + Cursor

### Task 4.1: `SyncState` — load/save atomic — TDD

**Files:**
- Create: `tests/test_sync_state.py`
- Create: `src/mempalace_sync/sync_state.py`

**Step 1: Failing test**

```python
# tests/test_sync_state.py
from mempalace_sync.sync_state import SyncState

def test_load_missing_returns_defaults(tmp_path):
    s = SyncState.load(tmp_path / "state.json")
    assert s.last_synced_message_id == 0
    assert s.total_synced == 0

def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "state.json"
    s = SyncState.load(path)
    s.last_synced_message_id = 42
    s.total_synced = 100
    s.save(path)
    s2 = SyncState.load(path)
    assert s2.last_synced_message_id == 42
    assert s2.total_synced == 100

def test_corrupted_file_falls_back(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    s = SyncState.load(path)  # should not raise
    assert s.last_synced_message_id == 0

def test_save_is_atomic(tmp_path):
    """Save uses tmp+rename so crash-mid-write leaves old file intact."""
    path = tmp_path / "state.json"
    s = SyncState.load(path); s.last_synced_message_id = 1; s.save(path)
    # Ensure no stray .tmp files remain
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []
```

**Step 2: Implementation**

```python
# src/mempalace_sync/sync_state.py
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict, field
from pathlib import Path

SCHEMA_VERSION = 1

@dataclass
class SyncState:
    last_synced_message_id: int = 0
    last_run_at: float = 0.0
    total_synced: int = 0
    failed_session_ids: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)  # atomic on POSIX
```

**Step 3: Run tests → pass → commit.**

```bash
git add . && git commit -m "feat: SyncState atomic load/save"
```

---

## Phase 5: Orchestrator + Lock

### Task 5.1: `run_once()` happy path — TDD

**Objective:** End-to-end one-shot: read → export → invoke convo_miner (mocked) → update cursor.

**Files:**
- Create: `tests/test_sync_engine.py`
- Create: `src/mempalace_sync/sync_engine.py`

**Step 1: Failing test**

```python
# tests/test_sync_engine.py
from unittest.mock import MagicMock
from mempalace_sync.sync_engine import run_once, SyncConfig

def test_run_once_happy_path(seed_session, fake_state_db, tmp_path, monkeypatch):
    sid, ids = seed_session(n=4)
    cfg = SyncConfig(
        state_db=fake_state_db,
        palace_dir=tmp_path / "palace",
        state_path=tmp_path / "state.json",
        inbox_dir=tmp_path / "inbox",
    )
    miner = MagicMock(return_value=None)
    monkeypatch.setattr("mempalace_sync.sync_engine.run_convo_miner", miner)

    stats = run_once(cfg)
    assert stats.new == 4
    assert stats.success is True
    miner.assert_called_once()
    # cursor advanced
    from mempalace_sync.sync_state import SyncState
    assert SyncState.load(cfg.state_path).last_synced_message_id == ids[-1]

def test_run_once_idempotent_when_empty(fake_state_db, tmp_path, monkeypatch):
    cfg = SyncConfig(
        state_db=fake_state_db,
        palace_dir=tmp_path / "palace",
        state_path=tmp_path / "state.json",
        inbox_dir=tmp_path / "inbox",
    )
    miner = MagicMock()
    monkeypatch.setattr("mempalace_sync.sync_engine.run_convo_miner", miner)
    stats = run_once(cfg)
    assert stats.new == 0
    miner.assert_not_called()

def test_run_once_does_not_advance_cursor_on_miner_failure(seed_session, fake_state_db, tmp_path, monkeypatch):
    sid, ids = seed_session(n=2)
    cfg = SyncConfig(
        state_db=fake_state_db,
        palace_dir=tmp_path / "palace",
        state_path=tmp_path / "state.json",
        inbox_dir=tmp_path / "inbox",
    )
    def boom(*a, **kw): raise RuntimeError("miner died")
    monkeypatch.setattr("mempalace_sync.sync_engine.run_convo_miner", boom)
    stats = run_once(cfg)
    assert stats.success is False
    from mempalace_sync.sync_state import SyncState
    # cursor stays at 0 so next run retries
    assert SyncState.load(cfg.state_path).last_synced_message_id == 0
```

**Step 2: Implementation**

```python
# src/mempalace_sync/sync_engine.py
"""Single-shot sync orchestrator."""
from __future__ import annotations
import subprocess, time, logging
from dataclasses import dataclass
from pathlib import Path

from .state_reader import StateReader
from .exporter import export_messages_to_jsonl
from .sync_state import SyncState

log = logging.getLogger(__name__)

@dataclass
class SyncConfig:
    state_db: Path
    palace_dir: Path
    state_path: Path
    inbox_dir: Path
    batch_limit: int = 10_000

@dataclass
class SyncStats:
    new: int = 0
    success: bool = True

def run_convo_miner(inbox_dir: Path, palace_dir: Path) -> None:
    """Invoke `mempalace --palace <dir> mine <dir> --mode convos`.

    Raises CalledProcessError on non-zero exit. Note: `--palace` is a GLOBAL
    flag, it MUST come before the `mine` subcommand.
    """
    subprocess.run(
        [
            "mempalace", "--palace", str(palace_dir),
            "mine", str(inbox_dir), "--mode", "convos",
        ],
        check=True,
    )

def run_once(cfg: SyncConfig) -> SyncStats:
    cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
    cfg.palace_dir.mkdir(parents=True, exist_ok=True)

    state = SyncState.load(cfg.state_path)
    reader = StateReader(cfg.state_db)
    msgs = reader.query_new_messages(state.last_synced_message_id, limit=cfg.batch_limit)
    if not msgs:
        state.last_run_at = time.time(); state.save(cfg.state_path)
        return SyncStats(new=0, success=True)

    files = export_messages_to_jsonl(msgs, cfg.inbox_dir)
    if not files:
        # all messages had empty content
        state.last_synced_message_id = max(m["id"] for m in msgs)
        state.last_run_at = time.time(); state.save(cfg.state_path)
        return SyncStats(new=0, success=True)

    try:
        run_convo_miner(cfg.inbox_dir, cfg.palace_dir)
    except Exception as e:
        log.error("convo_miner failed: %s", e)
        state.failed_session_ids = list({m["session_id"] for m in msgs})
        state.last_run_at = time.time()
        state.save(cfg.state_path)
        return SyncStats(new=len(msgs), success=False)

    state.last_synced_message_id = max(m["id"] for m in msgs)
    state.total_synced += len(msgs)
    state.failed_session_ids = []
    state.last_run_at = time.time()
    state.save(cfg.state_path)
    return SyncStats(new=len(msgs), success=True)
```

**Step 3: Run → pass → commit.**

```bash
git add . && git commit -m "feat: sync engine orchestrator (run_once)"
```

### Task 5.2: File lock (prevent two daemons)

**Step 1: Failing test**

```python
# tests/test_sync_engine.py (append)
import threading, time
from mempalace_sync.sync_engine import acquire_lock, LockHeldError
import pytest

def test_lock_prevents_concurrent(tmp_path):
    lock = tmp_path / "sync.lock"
    with acquire_lock(lock):
        with pytest.raises(LockHeldError):
            with acquire_lock(lock):
                pass

def test_lock_releases_on_exit(tmp_path):
    lock = tmp_path / "sync.lock"
    with acquire_lock(lock):
        pass
    # second acquire should succeed
    with acquire_lock(lock):
        pass
```

**Step 2: Implementation (append to sync_engine.py)**

```python
import fcntl
from contextlib import contextmanager

class LockHeldError(RuntimeError): pass

@contextmanager
def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise LockHeldError(f"Another sync is running ({lock_path})")
    try:
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
```

**Step 3: Run → pass → commit.**

```bash
git add . && git commit -m "feat: flock-based daemon singleton"
```

---

## Phase 6: CLI

### Task 6.1: `mempalace-sync run` / `status` / `watch`

**Files:**
- Modify: `src/mempalace_sync/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Failing test (smoke-level)**

```python
# tests/test_cli.py
from mempalace_sync.cli import main

def test_status_prints_state(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("mempalace_sync.cli.default_config_dir", lambda: tmp_path / "mempalace-sync")
    rc = main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "last_synced_message_id" in out

def test_run_executes(tmp_path, monkeypatch, seed_session, fake_state_db):
    sid, ids = seed_session(n=2)
    monkeypatch.setattr("mempalace_sync.sync_engine.run_convo_miner", lambda *a, **k: None)
    argv = [
        "run",
        "--state-db", str(fake_state_db),
        "--palace-dir", str(tmp_path / "palace"),
        "--config-dir", str(tmp_path / "cfg"),
    ]
    rc = main(argv)
    assert rc == 0
```

**Step 2: Implementation**

```python
# src/mempalace_sync/cli.py
from __future__ import annotations
import argparse, json, logging, os, sys, time
from dataclasses import asdict
from pathlib import Path

from .sync_engine import SyncConfig, run_once, acquire_lock, LockHeldError
from .sync_state import SyncState

log = logging.getLogger("mempalace_sync")

def default_config_dir() -> Path:
    return Path(os.environ.get("HOME", ".")) / ".hermes" / "mempalace-sync"

def _build_config(args) -> SyncConfig:
    cfg_dir = Path(args.config_dir or default_config_dir())
    return SyncConfig(
        state_db=Path(args.state_db or (Path.home() / ".hermes" / "state.db")),
        palace_dir=Path(args.palace_dir or (Path.home() / ".mempalace" / "default")),
        state_path=cfg_dir / "state.json",
        inbox_dir=cfg_dir / "inbox",
    )

def cmd_run(args) -> int:
    cfg = _build_config(args)
    try:
        with acquire_lock(cfg.state_path.parent / "sync.lock"):
            stats = run_once(cfg)
    except LockHeldError:
        log.warning("another sync is running; skipping")
        return 0
    print(f"synced={stats.new} success={stats.success}")
    return 0 if stats.success else 1

def cmd_watch(args) -> int:
    cfg = _build_config(args)
    interval = args.interval
    log.info("watching every %ds", interval)
    while True:
        try:
            with acquire_lock(cfg.state_path.parent / "sync.lock"):
                stats = run_once(cfg)
            log.info("synced=%d success=%s", stats.new, stats.success)
        except LockHeldError:
            log.debug("lock held; skip")
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            log.exception("sync iteration failed: %s", e)
        time.sleep(interval)

def cmd_status(args) -> int:
    cfg = _build_config(args)
    state = SyncState.load(cfg.state_path)
    print(json.dumps(asdict(state), indent=2))
    return 0

def cmd_reset(args) -> int:
    cfg = _build_config(args)
    SyncState().save(cfg.state_path)
    print(f"reset {cfg.state_path}")
    return 0

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="mempalace-sync")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--state-db", default=None)
    p.add_argument("--palace-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    w = sub.add_parser("watch"); w.add_argument("--interval", type=int, default=300)
    sub.add_parser("status")
    sub.add_parser("reset")

    args = p.parse_args(argv)
    return {
        "run": cmd_run, "watch": cmd_watch, "status": cmd_status, "reset": cmd_reset,
    }[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Run tests → pass → commit.**

```bash
git add . && git commit -m "feat: CLI — run/watch/status/reset"
```

### Task 6.2: `backfill` subcommand

**Objective:** One-shot mode that ignores cursor and ingests from the beginning.

**Files:** Modify `cli.py`, add test.

**Step 1: Test**

```python
def test_backfill_ignores_cursor(seed_session, fake_state_db, tmp_path, monkeypatch):
    sid, ids = seed_session(n=3)
    # advance cursor past everything
    state_path = tmp_path / "cfg" / "state.json"
    state_path.parent.mkdir(parents=True)
    from mempalace_sync.sync_state import SyncState
    s = SyncState(); s.last_synced_message_id = 999; s.save(state_path)
    monkeypatch.setattr("mempalace_sync.sync_engine.run_convo_miner", lambda *a, **k: None)
    from mempalace_sync.cli import main
    rc = main([
        "backfill",
        "--state-db", str(fake_state_db),
        "--palace-dir", str(tmp_path / "palace"),
        "--config-dir", str(tmp_path / "cfg"),
    ])
    assert rc == 0
    assert SyncState.load(state_path).last_synced_message_id == ids[-1]
```

**Step 2: Add `cmd_backfill` that resets cursor to 0 then calls `run_once`. Wire into `main()` subparsers.**

**Step 3: Commit.**

---

## Phase 7: Deployment Artifacts

### Task 7.1: systemd unit

**Files:** Create `deploy/mempalace-sync.service` and `deploy/README.md`.

```ini
# deploy/mempalace-sync.service
[Unit]
Description=MemPalace sync daemon (Hermes → MemPalace)
After=default.target

[Service]
Type=simple
ExecStart=%h/.local/bin/mempalace-sync watch --interval 300
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
# never let this eat memory
MemoryMax=512M

[Install]
WantedBy=default.target
```

Install instructions in `deploy/README.md`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/mempalace-sync.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mempalace-sync
systemctl --user status mempalace-sync
journalctl --user -u mempalace-sync -f
```

**Commit:** `chore: systemd user unit`

### Task 7.2: Hermes MCP config snippet

**File:** `deploy/hermes-config-snippet.yaml`

```yaml
# Append to ~/.hermes/config.yaml under mcp_servers:
mcp_servers:
  mempalace:
    command: "mempalace"
    args: ["--palace", "/home/jethro/.mempalace/default", "mcp"]
    env: {}
    timeout: 120
    connect_timeout: 60
    tools:
      include:
        - "search_memories"
        - "recall_entity"
        - "list_wings"
        - "list_rooms_in_wing"
        - "read_drawer"
```

**Commit:** `docs: Hermes MCP config snippet`

---

## Phase 8: Dogfood & Validation

### Task 8.1: Run against real state.db (read-only) for 24h

**Steps:**
1. `mempalace-sync status` → expect zero-state
2. `mempalace-sync run` (single shot against real `~/.hermes/state.db`) — inspect inbox/, state.json
3. Enable systemd unit with `--interval 300`
4. After 24h: `mempalace-sync status` + `journalctl --user -u mempalace-sync` → no errors
5. Manually query via MCP: spin up Hermes, ask "找找上次討論 MemPalace 的對話" → confirm hit

### Task 8.2: Stop-the-world test

1. `systemctl --user stop mempalace-sync`
2. Continue using Hermes normally for 1 hour
3. Restart daemon → verify it catches up without duplicates

### Task 8.3: Retrieval A/B compare

Write a throwaway script that queries both `session_search` (FTS5) and `search_memories` (MemPalace) for 10 canned Chinese queries; inspect which wins. Save findings to `docs/eval-results.md` in this repo.

---

## Phase 9: Hermes Config Rollout

### Task 9.1: Edit `~/.hermes/config.yaml`

Append the snippet from Task 7.2. Restart Hermes CLI/gateway.

### Task 9.2: Verify MCP tool appears

```bash
hermes tools | grep mempalace
```

Expected: `mempalace.search_memories`, `mempalace.recall_entity`, etc.

### Task 9.3: Prompt nudge (optional)

If LLM ignores MemPalace, add to `~/.hermes/memories/memory.md`:

> When a user references a past conversation and `session_search` returns nothing, try `mempalace.search_memories` before asking the user to repeat themselves.

---

## Success Criteria

- [ ] All phases' tests green (`pytest` in mempalace-sync repo, coverage ≥ 85%)
- [ ] Hermes repo: **0** lines changed in `hermes_state.py`, `run_agent.py`, `cli.py`, `gateway/*`
- [ ] `~/.hermes/config.yaml`: only +10 lines under `mcp_servers.mempalace`
- [ ] After 24h dogfood: no errors in journalctl, cursor advancing, palace growing
- [ ] A/B test shows MemPalace recalls at least 3/10 Chinese queries FTS5 missed
- [ ] `systemctl --user stop mempalace-sync` cleanly reverts to FTS5-only operation

---

## Pitfalls & Notes

- **Don't rename `messages.id`** — Hermes may add columns in future migrations, but the AUTOINCREMENT cursor assumption only breaks if they ever delete or renumber rows, which they don't.
- **`mempalace convo-mine` CLI name** — verified in Task 0.2; if upstream renames, patch `run_convo_miner` in `sync_engine.py`.
- **Embed model env var** — `MEMPALACE_EMBED_MODEL` is assumed; if upstream uses a different mechanism, adapt in Task 0.1.
- **Chroma lock contention** — with ONE writer (this daemon) and ChromaDB's SQLite, we should never hit the lock issue the spike warned about. If a future feature adds a second writer (e.g., Hermes itself also writing), revisit.
- **Don't mirror `tool_calls` / `reasoning`** — exporter drops them intentionally; revisit only if users ask to recall tool outputs.
- **Time zones** — `_ts_to_iso` uses UTC; MemPalace KG stores as-is. If user expects local time, adjust at query layer, not ingest.

---

## Handoff

Plan complete. Ready to execute using `subagent-driven-development` — one subagent per task, two-stage review (spec compliance → code quality). Phase 0 is the critical gate; if it fails, stop and re-plan.
