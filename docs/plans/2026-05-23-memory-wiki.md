# Memory Wiki Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a dashboard page that turns Hermes session history into a navigable “memory wiki” with subject pages and daily logs, so a user can click into subjects, conversations, and work completed together.

**Architecture:** Add a read-only analytics layer over the existing `state.db` session/message tables, exposed through new dashboard REST endpoints. The backend computes deterministic subjects from existing session titles, previews, user messages, assistant summaries, tool names, and date buckets; the React dashboard renders `/memory` with Overview, Subjects, Daily Logs, and detail panes that deep-link into existing session/message views.

**Tech Stack:** Python/FastAPI dashboard backend (`hermes_cli/web_server.py`), SQLite `state.db` through `SessionDB`, React/Vite dashboard (`web/src`), TypeScript API client (`web/src/lib/api.ts`), existing dashboard UI components.

---

## Product Shape

### Core UX

1. New sidebar item: **Memory Wiki** at `/memory`.
2. `/memory` shows:
   - Search box across subjects and indexed conversations.
   - Subject list/cards: subject name, aliases/keywords, conversation count, last touched, representative snippets.
   - Daily logs: grouped by date, showing sessions, topics, and concrete “work done” summaries.
   - Recent activity panel linking to existing session details.
3. Clicking a subject opens `/memory/subjects/:slug`:
   - Subject overview.
   - Related sessions.
   - Related messages/snippets.
   - Timeline of days where that subject appeared.
4. Clicking a day opens `/memory/days/:yyyy-mm-dd`:
   - All sessions from that day.
   - Subject chips for the day.
   - “What we did” bullets derived from user/assistant messages and tool calls.
5. Clicking any session/message reuses existing session APIs and route patterns where possible.

### Backend Data Model — MVP

No schema migration for v1. Compute on demand from `sessions` + `messages`:

```ts
type MemorySubject = {
  slug: string;
  name: string;
  keywords: string[];
  session_count: number;
  message_count: number;
  first_seen: number;
  last_seen: number;
  sessions: Array<{
    id: string;
    title?: string;
    source: string;
    started_at: number;
    last_active: number;
    preview: string;
  }>;
  snippets: Array<{
    message_id: number;
    session_id: string;
    role: string;
    timestamp: number;
    text: string;
  }>;
};

type DailyMemoryLog = {
  date: string;
  started_at_min: number;
  last_active_max: number;
  session_count: number;
  message_count: number;
  subjects: Array<{ slug: string; name: string; count: number }>;
  sessions: SessionInfo[];
  work_items: Array<{
    session_id: string;
    kind: "conversation" | "tool" | "coding" | "research" | "planning";
    text: string;
    timestamp: number;
  }>;
};
```

### Subject Extraction — MVP Heuristic

Implement deterministic, local-only extraction first:

1. Prefer session `title` when present.
2. Add noun-ish keyword phrases from first user messages and session previews.
3. Weight explicit project/file/tool mentions:
   - paths like `foo/bar.py`, `web/src/App.tsx`
   - package/repo names
   - tool names from `tool_calls` / `tool_name`
   - CamelCase identifiers and slash commands
4. Remove generic stopwords and Hermes boilerplate.
5. Normalize aliases into slugs (`memory wiki`, `Memory Wiki`, `memory-wiki` → `memory-wiki`).
6. Return enough subjects for navigation, not perfect taxonomy. Later v2 can add an LLM summarization/indexing cron job.

---

## Task 1: Add Backend Memory Aggregation Helpers

**Objective:** Create pure helper functions that query `state.db` and aggregate subjects/days without changing schema.

**Files:**
- Create: `hermes_cli/memory_wiki.py`
- Test: `tests/hermes_cli/test_memory_wiki.py`

**Step 1: Write failing tests for slugging and keyword extraction**

Create `tests/hermes_cli/test_memory_wiki.py` with tests for:

```python
def test_slugify_subject_name():
    from hermes_cli.memory_wiki import slugify_subject
    assert slugify_subject("Memory Wiki") == "memory-wiki"
    assert slugify_subject("web/src/App.tsx") == "web-src-app-tsx"


def test_extract_subject_candidates_prefers_titles_and_paths():
    from hermes_cli.memory_wiki import extract_subject_candidates
    session = {"title": "Build memory wiki", "preview": "I want to build a memory wiki"}
    messages = [
        {"role": "user", "content": "Add /memory to web/src/App.tsx"},
        {"role": "tool", "tool_name": "search_files", "content": ""},
    ]
    candidates = extract_subject_candidates(session, messages)
    names = [c.name for c in candidates]
    assert "memory wiki" in names
    assert "web/src/App.tsx" in names
    assert "search_files" in names
```

Run:

```bash
python -m pytest tests/hermes_cli/test_memory_wiki.py -q -o 'addopts='
```

Expected: FAIL because `hermes_cli.memory_wiki` does not exist.

**Step 2: Implement helper module**

In `hermes_cli/memory_wiki.py`, add:

- `SubjectCandidate` dataclass.
- `slugify_subject(name: str) -> str`.
- `extract_subject_candidates(session: dict, messages: list[dict]) -> list[SubjectCandidate]`.
- `_STOPWORDS` set.
- Regexes for paths, CamelCase, quoted phrases, slash commands, and tool names.

Keep dependencies stdlib-only.

**Step 3: Verify tests pass**

Run:

```bash
python -m pytest tests/hermes_cli/test_memory_wiki.py -q -o 'addopts='
```

Expected: PASS.

---

## Task 2: Add Subject and Day Aggregators

**Objective:** Build functions that turn real `SessionDB` rows into API-ready dictionaries.

**Files:**
- Modify: `hermes_cli/memory_wiki.py`
- Test: `tests/hermes_cli/test_memory_wiki.py`

**Step 1: Add tests using a temporary `SessionDB`**

Create tests that:

1. Instantiate `SessionDB(db_path=tmp_path / "state.db")` if the constructor supports it; otherwise follow existing test patterns for temp Hermes home.
2. Create sessions/messages.
3. Assert `build_memory_overview(db)` returns subjects and daily logs.

Search existing tests for `SessionDB(` to copy the repo’s temp-db pattern.

**Step 2: Implement functions**

Add:

```python
def build_memory_subjects(db, *, limit: int = 100, query: str | None = None) -> list[dict]: ...

def get_memory_subject(db, slug: str) -> dict | None: ...

def build_daily_logs(db, *, limit_days: int = 60) -> list[dict]: ...

def get_daily_log(db, date: str) -> dict | None: ...

def build_memory_overview(db, *, subject_limit: int = 50, day_limit: int = 30) -> dict: ...
```

Use `db.list_sessions_rich(limit=..., include_children=False, order_by_last_active=True)` and `db.get_messages(session_id)`.

**Step 3: Verify targeted tests**

```bash
python -m pytest tests/hermes_cli/test_memory_wiki.py -q -o 'addopts='
```

Expected: PASS.

---

## Task 3: Add Dashboard API Endpoints

**Objective:** Expose the memory wiki data through protected dashboard APIs.

**Files:**
- Modify: `hermes_cli/web_server.py`
- Test: `tests/hermes_cli/test_web_server.py` or new `tests/hermes_cli/test_web_server_memory_wiki.py`

**Step 1: Add failing endpoint tests**

Test these routes:

- `GET /api/memory/overview`
- `GET /api/memory/subjects`
- `GET /api/memory/subjects/{slug}`
- `GET /api/memory/days`
- `GET /api/memory/days/{date}`

Follow existing `web_server` test auth-token patterns.

**Step 2: Implement endpoints**

Add near the sessions endpoints in `hermes_cli/web_server.py`:

```python
@app.get("/api/memory/overview")
async def get_memory_overview(...): ...

@app.get("/api/memory/subjects")
async def get_memory_subjects(q: str = "", limit: int = 100): ...

@app.get("/api/memory/subjects/{slug}")
async def get_memory_subject(slug: str): ...

@app.get("/api/memory/days")
async def get_memory_days(limit: int = 60): ...

@app.get("/api/memory/days/{date}")
async def get_memory_day(date: str): ...
```

All endpoints should instantiate `SessionDB()`, call `hermes_cli.memory_wiki`, close DB in `finally`, log exceptions with `_log.exception`, and raise `HTTPException(500, ...)` on errors.

**Step 3: Verify tests**

```bash
python -m pytest tests/hermes_cli/test_web_server_memory_wiki.py tests/hermes_cli/test_web_server.py -q -o 'addopts='
```

Expected: PASS.

---

## Task 4: Add TypeScript API Client Types

**Objective:** Make memory APIs available to React with typed response shapes.

**Files:**
- Modify: `web/src/lib/api.ts`

**Step 1: Add interfaces**

Add `MemorySubject`, `DailyMemoryLog`, `MemoryOverviewResponse`, and endpoint response types near the existing session types.

**Step 2: Add API methods**

Add to `api`:

```ts
getMemoryOverview: () => fetchJSON<MemoryOverviewResponse>("/api/memory/overview"),
getMemorySubjects: (q = "", limit = 100) => fetchJSON<MemorySubjectsResponse>(...),
getMemorySubject: (slug: string) => fetchJSON<MemorySubjectResponse>(...),
getMemoryDays: (limit = 60) => fetchJSON<MemoryDaysResponse>(...),
getMemoryDay: (date: string) => fetchJSON<MemoryDayResponse>(...),
```

**Step 3: Verify TypeScript**

```bash
npm run typecheck
```

Expected: PASS.

---

## Task 5: Build Memory Wiki React Page

**Objective:** Create `/memory` overview page with subjects, daily logs, search, and clickable session links.

**Files:**
- Create: `web/src/pages/MemoryWikiPage.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/i18n/en.ts` if nav labels require translation entries

**Step 1: Create page skeleton**

`MemoryWikiPage.tsx` should:

- Load `api.getMemoryOverview()` on mount.
- Show loading/error states matching other pages.
- Render subject cards and daily log cards.
- Provide a search input that calls `api.getMemorySubjects(q)` after debounce.
- Link sessions to existing Sessions UI where available.

**Step 2: Register route/nav**

In `web/src/App.tsx`:

- Import `MemoryWikiPage`.
- Add `"/memory": MemoryWikiPage` to `BUILTIN_ROUTES_CORE`.
- Add nav item `{ path: "/memory", labelKey: "memory", label: "Memory Wiki", icon: Database }` after Sessions.

**Step 3: Verify typecheck/build**

```bash
npm run typecheck
npm run build
```

Expected: PASS.

---

## Task 6: Add Subject and Day Detail Routes

**Objective:** Support deep links for subject and daily-log detail pages.

**Files:**
- Modify: `web/src/pages/MemoryWikiPage.tsx` or create:
  - `web/src/pages/MemorySubjectPage.tsx`
  - `web/src/pages/MemoryDayPage.tsx`
- Modify: `web/src/App.tsx`

**Step 1: Add route components**

Add detail components that read params via `useParams()` and fetch:

- `/api/memory/subjects/{slug}`
- `/api/memory/days/{date}`

**Step 2: Register routes**

Add routes:

- `/memory/subjects/:slug`
- `/memory/days/:date`

If `BUILTIN_ROUTES_CORE` only supports static path strings, extend route construction minimally or handle subroutes inside a single `/memory/*` route.

**Step 3: Verify browser navigation**

```bash
npm run typecheck
npm run build
```

Expected: PASS.

---

## Task 7: Add Docs and Manual QA Checklist

**Objective:** Document how to use the memory wiki and how privacy works.

**Files:**
- Create: `website/docs/user-guide/features/memory-wiki.md`
- Modify docs sidebar if needed.

**Content requirements:**

- Explain that v1 is generated from local `~/.hermes/state.db` session history.
- Explain no remote upload is required.
- Explain subjects are deterministic heuristics and may be imperfect.
- Explain daily logs are derived from sessions/messages and tool calls.
- Include launch command:

```bash
hermes dashboard
```

or the repo’s current web command if different.

**Manual QA:**

1. Start dashboard.
2. Open `/memory`.
3. Confirm subjects load from real history.
4. Search a known topic.
5. Click subject.
6. Click daily log.
7. Click a session and confirm existing transcript view works.

---

## Task 8: Full Verification

**Objective:** Run targeted backend/frontend checks before PR or merge.

**Commands:**

```bash
python -m pytest tests/hermes_cli/test_memory_wiki.py tests/hermes_cli/test_web_server_memory_wiki.py -q -o 'addopts='
npm run typecheck
npm run build
```

If the dashboard has Playwright/e2e tests, add one smoke test for `/memory` and run the existing frontend test command.

---

## Future v2 Ideas

- Persistent `memory_wiki_subjects`, `memory_wiki_subject_sessions`, and `memory_wiki_daily_logs` cache tables for faster large histories.
- Background cron job to summarize each day and improve taxonomy with an auxiliary model.
- Manual subject rename/merge/split UI.
- Export to Markdown/Obsidian.
- Backlinks from session pages to related subject pages.
- Timeline graph of subject co-occurrence.
