"""Regression tests for the holographic memory provider.

These tests cover production issues found during the 2026-05-30 memory audit:
- fact_store(search) bypassed retrieval_count telemetry
- strict FTS5 AND semantics made natural-language queries return no results
- migrated/imported rows could have NULL hrr_vector forever unless manually rebuilt
- relaxed FTS OR fallback could over-rank generic conversational terms like "store"
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


CURATED_BASELINE_CASES = [
    {
        "content": "10x10 workspace is ~/Developer/10x10; repos: app-flutter, portal-api-node, 10x10-platform; skills live in .agents/skills and are projected to .claude/skills by symlink.",
        "category": "project",
        "tags": "10x10,workspace,repos,skills",
        "query": "where is the 10x10 workspace and which repos are there?",
        "expected": "~/Developer/10x10",
    },
    {
        "content": "Hermes ops: ~/.local/bin/hermes points to ~/.hermes/hermes-agent/venv/bin/hermes source checkout. Prefer `hermes update` for updates; if WebUI blocks, use manual git fast-forward plus `uv pip install --python venv/bin/python -e .`. Restart gateway with `hermes gateway restart`.",
        "category": "tool",
        "tags": "hermes,upgrade,terminal,command,update,source-checkout",
        "query": "how should we update Hermes from the source checkout?",
        "expected": "prefer `hermes update`",
    },
    {
        "content": "mise Java 17 path is /Users/erichsu/.local/share/mise/installs/java/17.0.2; use it for Flutter/Gradle failures; prefer mise env for fastlane/bundle.",
        "category": "tool",
        "tags": "java,mise,flutter,gradle,fastlane",
        "query": "which Java 17 should Flutter Gradle use via mise?",
        "expected": "17.0.2",
    },
    {
        "content": "10x10-platform requires Node >=20.19.0 <21; local shell may be v25.9.0 and pnpm can warn about engines.",
        "category": "project",
        "tags": "10x10-platform,node,pnpm",
        "query": "what Node version does 10x10 platform require?",
        "expected": "Node >=20.19.0",
    },
    {
        "content": "portal-api-node: avoid patch editing api_backend/models/user.ts because it can truncate; use raw-file plus git restore. Clear stale .worktrees if TS lint scans them.",
        "category": "project",
        "tags": "portal-api-node,typescript,worktree,pitfall",
        "query": "which portal api node user model file should not be patch edited?",
        "expected": "api_backend/models/user.ts",
    },
    {
        "content": "Hermes default is openai-codex/gpt-5.5 with fallback anthropic/claude-sonnet-4-6; empty delegation.model/provider inherits current session; Opus delegation needs explicit per-call model.",
        "category": "tool",
        "tags": "hermes,routing,delegation,models,model,default,inherit,inherits",
        "query": "what is Hermes default model and how does empty delegation inherit?",
        "expected": "openai-codex/gpt-5.5",
    },
    {
        "content": "Opus effort policy: default high; complex tasks use xhigh via `hermes config set agent.reasoning_effort xhigh`, then restore afterward; routing.yaml is source of truth.",
        "category": "tool",
        "tags": "opus,reasoning,routing",
        "query": "how do we set Opus effort xhigh?",
        "expected": "reasoning_effort xhigh",
    },
    {
        "content": "Local Claude CLI is /opt/homebrew/bin/claude; it does not support --acp --stdio; credential name is tingyao.hsu--claude-code-laptop; skills model aliases are opus/sonnet.",
        "category": "tool",
        "tags": "claude,cli,credentials",
        "query": "does local Claude CLI support acp stdio?",
        "expected": "does not support --acp --stdio",
    },
    {
        "content": "GitHub PR watcher script is ~/.hermes/scripts/github_pr_monitor.py; Ops topic is 64623; cron route: Hermes to thread 78842, portal-api-node/10x10 to Ops 64623, task closeout to origin thread.",
        "category": "tool",
        "tags": "cron,telegram,ops,github",
        "query": "where is the GitHub PR watcher and Ops topic routing?",
        "expected": "github_pr_monitor.py",
    },
    {
        "content": "agent-browser CDP flag is --cdp 9222; commands include open/click/scroll/screenshot/eval/snapshot/fill; use eval JS click for radio and fill for textbox; wait 4-6s after open.",
        "category": "tool",
        "tags": "agent-browser,cdp,github,asc",
        "query": "what is the agent browser CDP flag for radio and textbox work?",
        "expected": "--cdp 9222",
    },
    {
        "content": "portal-api-node production API is https://portal-api-node.onrender.com/api/.",
        "category": "project",
        "tags": "portal-api-node,production,api",
        "query": "what is the portal api node production API URL?",
        "expected": "portal-api-node.onrender.com",
    },
    {
        "content": "Mongoose findOneAndUpdate with $set silently ignores fields not in schema; verify exact schema field names, especially camelCase vs snake_case date/datetime fields; affects portal-api-node schedulers.",
        "category": "project",
        "tags": "mongoose,portal-api-node,scheduler,pitfall",
        "query": "what Mongoose findOneAndUpdate schema pitfall affects portal schedulers?",
        "expected": "silently ignores",
    },
    {
        "content": "Eric prefers Traditional Chinese by default.",
        "category": "user_pref",
        "tags": "language,eric",
        "query": "what language does Eric prefer by default?",
        "expected": "Traditional Chinese",
    },
    {
        "content": "Telegram style: one topic at a time; use `新議題：…` for new topics; plain text; emoji-prefix set includes ✅❌📊⚠️; do not switch topics due to background notifications.",
        "category": "user_pref",
        "tags": "telegram,style,eric",
        "query": "what is Eric's Telegram one topic and emoji prefix style?",
        "expected": "one topic at a time",
    },
    {
        "content": "Eric workflow preference: research-first; use Opus for planning/high-risk, Sonnet for short verdicts, Codex for implementation; prefer small verified slices and proactive continuation.",
        "category": "user_pref",
        "tags": "workflow,models,eric",
        "query": "what is Eric's research first Opus Codex workflow preference?",
        "expected": "research-first",
    },
    {
        "content": "Eric release preference: release fully automated; App Store should prefer API key and avoid Apple ID/2FA.",
        "category": "user_pref",
        "tags": "release,app-store,eric",
        "query": "what does Eric prefer for App Store release credentials?",
        "expected": "avoid Apple ID/2FA",
    },
    {
        "content": "Review workflow preference: triage intent first, choose Hermes/Claude/Codex, then Hermes synthesis. Skills governance upgrades prefer Opus+Codex+Hermes debate before landing.",
        "category": "user_pref",
        "tags": "review,governance,eric",
        "query": "what is the review workflow with triage Hermes Claude Codex synthesis?",
        "expected": "triage intent",
    },
    {
        "content": "Any code change must open a GitHub issue before code changes; order is open issue, plan, implement, PR; PR must reference the issue.",
        "category": "user_pref",
        "tags": "github,issue,workflow,eric",
        "query": "what must happen before code changes and PR?",
        "expected": "open a GitHub issue",
    },
    {
        "content": "Subagent dispatch rule: before subagent use, present A/B/C route options for Eric confirmation; Wave/PR slice endings need Opus advisor judgment first, unless Opus times out and Hermes explicitly says it is acting.",
        "category": "user_pref",
        "tags": "subagent,dispatch,opus,eric",
        "query": "what route options are required before subagent dispatch?",
        "expected": "A/B/C route options",
    },
    {
        "content": "Hermes memory architecture on this install: MEMORY.md/USER.md are always-on prompt memory with strict watermarks; holographic fact_store is enabled as second-line searchable memory with auto_extract=false. Healthy fact_store does not replace prompt-memory compression.",
        "category": "tool",
        "tags": "hermes,memory,governance,holographic,prompt-memory,fact_store",
        "query": "how does holographic fact store relate to prompt memory and auto extract?",
        "expected": "auto_extract=false",
    },
]


def test_fact_store_search_increments_retrieval_count(tmp_path):
    db_path = tmp_path / "memory_store.db"
    provider = HolographicMemoryProvider(
        config={"db_path": str(db_path), "default_trust": 0.5, "hrr_dim": 128}
    )
    provider.initialize("test-session")

    fact_id = provider._store.add_fact(  # type: ignore[union-attr]
        "Codex binary lives at /opt/homebrew/bin/codex.",
        category="tool",
        tags="codex,binary,path",
    )

    payload = json.loads(
        provider.handle_tool_call(
            "fact_store",
            {"action": "search", "query": "Codex binary", "category": "tool", "limit": 5},
        )
    )

    assert payload["count"] == 1
    assert payload["results"][0]["fact_id"] == fact_id
    row = provider._store._conn.execute(  # type: ignore[union-attr]
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert row["retrieval_count"] == 1


def test_natural_language_search_falls_back_when_fts5_returns_no_candidates(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    fact_id = store.add_fact(
        "Codex binary lives at /opt/homebrew/bin/codex.",
        category="tool",
        tags="codex,binary,path,cli",
    )
    retriever = FactRetriever(store=store, hrr_dim=128)

    # FTS5 treats whitespace as AND, so this query previously returned zero
    # because the fact does not contain every natural-language word.
    results = retriever.search("Where is the Codex CLI path fix?", category="tool", limit=3)

    assert results
    assert results[0]["fact_id"] == fact_id
    row = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert row["retrieval_count"] == 1


def test_relaxed_fts_filters_generic_conversational_terms(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    codex_id = store.add_fact(
        "Codex binary lives at /opt/homebrew/bin/codex.",
        category="tool",
        tags="codex,binary,path,cli",
    )
    store.add_fact(
        "Hermes memory store keeps searchable facts in SQLite.",
        category="tool",
        tags="memory,store,sqlite",
    )
    retriever = FactRetriever(store=store, hrr_dim=128)

    results = retriever.search(
        "where did we store the codex executable location",
        category="tool",
        limit=3,
    )

    assert results
    assert results[0]["fact_id"] == codex_id


def test_initialize_repairs_missing_hrr_vectors_for_migrated_rows(tmp_path):
    if not hrr._HAS_NUMPY:
        return

    db_path = tmp_path / "memory_store.db"
    store = MemoryStore(db_path=db_path, hrr_dim=128)
    fact_id = store.add_fact("Eric prefers Traditional Chinese by default.", category="user_pref")
    store._conn.execute("UPDATE facts SET hrr_vector = NULL WHERE fact_id = ?", (fact_id,))
    store._conn.commit()
    store.close()

    provider = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 128})
    provider.initialize("test-session")

    row = provider._store._conn.execute(  # type: ignore[union-attr]
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert row["hrr_vector"] is not None


def test_search_records_query_log_sample(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    fact_id = store.add_fact(
        "Codex binary lives at /opt/homebrew/bin/codex.",
        category="tool",
        tags="codex,binary,path,cli",
    )
    retriever = FactRetriever(
        store=store,
        hrr_dim=128,
        query_log_sample_rate=1.0,
    )

    results = retriever.search("where is the codex binary?", category="tool", limit=3)

    assert results[0]["fact_id"] == fact_id
    rows = store._conn.execute(
        """
        SELECT query, category, result_count, top_fact_id
        FROM query_log
        ORDER BY query_id ASC
        """
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "query": "where is the codex binary?",
            "category": "tool",
            "result_count": 1,
            "top_fact_id": fact_id,
        }
    ]


def test_search_does_not_record_query_log_when_sample_rate_zero(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    store.add_fact(
        "Codex binary lives at /opt/homebrew/bin/codex.",
        category="tool",
        tags="codex,binary,path,cli",
    )
    retriever = FactRetriever(
        store=store,
        hrr_dim=128,
        query_log_sample_rate=0.0,
    )

    results = retriever.search("where is the codex binary?", category="tool", limit=3)

    assert results
    row = store._conn.execute("SELECT COUNT(*) AS n FROM query_log").fetchone()
    assert row["n"] == 0


def test_auto_extract_string_false_is_hard_guard(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "auto_extract": "false",
            "hrr_dim": 128,
        }
    )
    provider.initialize("test-session")

    provider.on_session_end(
        [
            {
                "role": "user",
                "content": "I prefer every durable preference to stay explicit unless manually saved.",
            }
        ]
    )

    row = provider._store._conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()  # type: ignore[union-attr]
    assert row["n"] == 0


def test_stale_fact_candidates_apply_structured_filters(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    fragment = "upgrade Hermes with `uv tool upgrade hermes-agent`"
    old_cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")

    actionable_id = store.add_fact(
        f"Deprecated guidance: {fragment}; use source checkout update flow instead.",
        category="tool",
        tags="hermes,upgrade,stale",
    )
    recent_id = store.add_fact(
        f"Recent note mentioning {fragment} while documenting a fixed regression.",
        category="tool",
        tags="hermes,upgrade,regression-note",
    )
    low_trust_id = store.add_fact(
        f"Low-trust imported note: {fragment}.",
        category="tool",
        tags="hermes,upgrade,imported",
    )
    user_pref_id = store.add_fact(
        f"User quoted a stale command literally: {fragment}.",
        category="user_pref",
        tags="hermes,quote,user",
    )
    store._conn.execute(
        "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id IN (?, ?, ?)",
        (old_cutoff, old_cutoff, actionable_id, low_trust_id, user_pref_id),
    )
    store._conn.execute(
        "UPDATE facts SET trust_score = 0.1 WHERE fact_id = ?",
        (low_trust_id,),
    )
    store._conn.commit()

    candidates = store.find_stale_fact_candidates(
        fragments=(fragment,),
        stale_days=60,
        min_trust=0.3,
        categories=("project", "tool"),
        limit=10,
    )

    assert [candidate["fact_id"] for candidate in candidates] == [actionable_id]
    assert recent_id not in [candidate["fact_id"] for candidate in candidates]
    assert low_trust_id not in [candidate["fact_id"] for candidate in candidates]
    assert user_pref_id not in [candidate["fact_id"] for candidate in candidates]


def test_stale_fact_candidates_escape_wildcards_and_casefold_matches(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", hrr_dim=128)
    fragment = "Hermes 100%_DONE"
    old_cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")

    match_id = store.add_fact(
        "Deprecated note: hermes 100%_done command.",
        category="tool",
        tags="hermes,stale,literal",
    )
    wildcard_only_id = store.add_fact(
        "Deprecated note: Hermes 100XADONE command.",
        category="tool",
        tags="hermes,stale,wildcard-decoy",
    )
    store._conn.execute(
        "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id IN (?, ?)",
        (old_cutoff, old_cutoff, match_id, wildcard_only_id),
    )
    store._conn.commit()

    candidates = store.find_stale_fact_candidates(
        fragments=(fragment,),
        stale_days=60,
        min_trust=0.3,
        categories=("tool",),
        limit=10,
    )

    assert [candidate["fact_id"] for candidate in candidates] == [match_id]
    assert candidates[0]["matched_fragments"] == [fragment]
