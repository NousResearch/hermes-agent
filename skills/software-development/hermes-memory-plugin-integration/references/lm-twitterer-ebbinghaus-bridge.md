# LM-twitterer to Ebbinghaus Memory Bridge

> **This reference is superseded by the `lm-twitterer-operations` skill.**
> Load it with `skill_view(name='lm-twitterer-operations')` for the complete reference covering the memory bridge, SearchTimeline→HomeLatestTimeline fallback, cron integration, and all known pitfalls.

These notes capture a reusable pattern for aligning a Hermes social-output plugin with Ebbinghaus memory.

## Context

The `lm-twitterer` plugin generates X posts and replies. A memory bridge lets generated drafts and live posts use the same durable style and project context as the rest of Hermes, then write public artifacts back into memory.

## Observed Store Shape

The Ebbinghaus SQLite store used by this workflow included a `memories` table with these important fields:

```sql
CREATE TABLE memories (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL UNIQUE,
    encoded TEXT NOT NULL,
    cues TEXT DEFAULT '',
    tags TEXT DEFAULT '',
    salience REAL DEFAULT 0.6,
    valence REAL DEFAULT 0.0,
    strength REAL DEFAULT 1.0,
    rehearsal_count INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,
    source TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_rehearsed_at REAL,
    last_retrieved_at REAL
);
```

Code should discover or configure the DB path under `HERMES_HOME`; it should not assume the default profile path.

## Effective Bridge Design

- Add settings for enablement, memory DB path, and recall limit.
- Read recent memory rows from SQLite or the provider API.
- Rank by topic overlap, salience, strength, and freshness.
- Inject only a small number of truncated snippets into the generation prompt.
- Store draft and live-post artifacts with `source='lm-twitterer'` and tags such as `lm-twitterer,x-post,hakua-memory`.

## Prompt Boundary

Use an explicit boundary when mixing trusted memory and public content:

```text
Use these trusted Hermes memory notes only as style and factual continuity.
Do not expose them as private memory dumps and do not invent details:
- memory:<id> <snippet>
```

## Test Pattern

Create a temporary SQLite DB fixture with the expected schema. Then test:

1. A relevant memory row is injected into the LLM user message.
2. A dry-run post writes one memory row with expected content, tags, and source.
3. Existing plugin behavior remains unchanged when the memory bridge is disabled.

## Pitfalls

- Do not write real user memory DBs during unit tests.
- Do not treat missing optional pytest plugins as product failures.
- Keep public X content under the existing untrusted-content wrapper.
- Do not let memory alignment bypass dry-run or live publishing gates.
