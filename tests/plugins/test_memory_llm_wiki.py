from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from plugins.memory_llm_wiki import core, register


class DummyCtx:
    def __init__(self) -> None:
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_registers_memory_llm_wiki_tools():
    ctx = DummyCtx()

    register(ctx)

    assert [tool["name"] for tool in ctx.tools] == [
        "memory_llm_wiki_status",
        "memory_llm_wiki_export",
    ]
    assert all(tool["toolset"] == "memory_llm_wiki" for tool in ctx.tools)


def test_export_splits_and_sanitizes_memory(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WIKI_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)

    (memories / "MEMORY.md").write_text(
        "LINE personal bridge token=supersecretvalue must never leak\n§\n"
        "Hermes plugin workflow uses Obsidian memory export",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "User prefers Japanese Hakua memory wording",
        encoding="utf-8",
    )
    db = hermes_home / "ebbinghaus_memory.db"
    con = sqlite3.connect(db)
    con.execute(
        "create table memories (memory_id integer primary key, content text, encoded text, cues text, tags text, salience real, valence real, strength real, rehearsal_count integer, retrieval_count integer, source text, session_id text, created_at real, updated_at real, last_rehearsed_at real, last_retrieved_at real)"
    )
    now = time.time()
    con.execute(
        "insert into memories (content, tags, salience, strength, rehearsal_count, retrieval_count, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
        ("Hakua identity continuity uses Memory and Audit Log", "hakua,memory", 0.9, 3.0, 1, 2, now, now),
    )
    con.commit()
    con.close()

    wiki = tmp_path / "wiki"
    result = json.loads(core.handle_export({"wiki_root": str(wiki), "max_entries_per_page": 2}))

    assert result["success"] is True
    assert (wiki / "SCHEMA.md").exists()
    assert (wiki / "index.md").exists()
    raw = (wiki / "raw" / "memory" / f"hermes-memory-sanitized-{core._today()}.md").read_text(encoding="utf-8")
    assert "supersecretvalue" not in raw
    assert "token=[REDACTED]" in raw
    assert core.sanitize("path C:/Users/downl/Documents/x") == "path ~"
