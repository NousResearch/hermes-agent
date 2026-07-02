# Dashboard API Integrations

Developer reference for the three backend services consumed by the Hermes web
dashboard. Covers base URL configuration, endpoints, request/response shapes,
and how to use the corresponding React hooks.

---

## Quick setup

All three services are hosted on the **same Hermes dashboard server** — no
separate origins to configure. The only env var you need is:

```
HERMES_DASHBOARD_URL=http://127.0.0.1:9119   # default; change for remote host
```

Set it in your shell or in `web/.env.local` before running the Vite dev server.

```bash
cd web
HERMES_DASHBOARD_URL=http://your-remote-host:9119 npm run dev
```

In production (when `hermes dashboard` serves the built bundle directly), the
Python server injects `window.__HERMES_SESSION_TOKEN__` and
`window.__HERMES_BASE_PATH__` into `index.html` automatically — no further
configuration needed.

All API calls use the shared `fetchJSON<T>()` helper in `web/src/lib/api.ts`,
which prepends `window.__HERMES_BASE_PATH__` to every URL and attaches the
session token header (`X-Hermes-Session-Token`) to all `/api/` requests.

---

## 1. Agent Profiles service

### Base URL

Relative to the dashboard origin. No separate env var — served on the same
host as the dashboard.

### API client module

`web/src/lib/api.ts` — `api.getProfiles()` and the other `api.profiles*`
methods. Import `fetchJSON` from the same file if you need a one-off call.

### Endpoints

#### List profiles

```
GET /api/profiles
```

Response — `{ profiles: ProfileInfo[] }`:

```ts
export interface ProfileInfo {
  name: string;              // profile directory name, e.g. "default"
  path: string;              // absolute path to profile directory
  is_default: boolean;
  gateway_running: boolean;
  model: string | null;      // currently configured model
  provider: string | null;
  has_env: boolean;          // whether a .env file exists for this profile
  skill_count: number;
  alias_path?: string | null;
  distribution_name?: string | null;
  distribution_version?: string | null;
  distribution_source?: string | null;
  description?: string;
  description_auto?: boolean;
}
```

#### Create profile

```
POST /api/profiles
Content-Type: application/json

{ "name": "my-profile", "clone_from_default": true }
```

Response — `{ ok: boolean; name: string; path: string }`.

#### Rename profile

```
PATCH /api/profiles/{name}
Content-Type: application/json

{ "new_name": "better-name" }
```

#### Delete profile

```
DELETE /api/profiles/{name}
```

Response — `{ ok: boolean }`.

### Hook: `useAgentProfiles`

File: `web/src/hooks/useAgentProfiles.ts`

```ts
import { useAgentProfiles } from "@/hooks/useAgentProfiles";

function MyComponent() {
  const { profiles, loading, error, refreshProfiles } = useAgentProfiles();

  if (loading) return <Spinner />;
  if (error)   return <ErrorBanner message={error} />;

  return (
    <ul>
      {profiles.map(p => <li key={p.name}>{p.name}</li>)}
    </ul>
  );
}
```

**State shape** (`AgentProfilesState`):

```ts
type AgentProfilesState = {
  profiles: ProfileInfo[];
  loading: boolean;
  error: string | null;
  // Call with keepExisting: true for background refresh without list flicker.
  refreshProfiles: (options?: { keepExisting?: boolean }) => Promise<void>;
};
```

**Adapter injection** — the hook accepts an optional `AgentProfilesClient`
argument, making it easy to swap transport or mock in tests:

```ts
type AgentProfilesClient = {
  listProfiles: () => Promise<{ profiles: ProfileInfo[] }>;
};

// Use default (live REST):
useAgentProfiles()

// Use a mock in tests:
const mockClient = { listProfiles: async () => ({ profiles: [] }) };
useAgentProfiles(mockClient)
```

**Loading / error contract**:

- `loading: true` from mount until the first response arrives.
- On success: `loading` drops to `false`, `profiles` is populated, `error`
  is `null`.
- On failure: `loading` drops to `false`, `error` holds the message string,
  `profiles` is reset to `[]` (unless `keepExisting: true` was passed).
- Stale-request guard: if a newer request is already in-flight the stale
  response is silently discarded.

---

## 2. Kanban board state service

### Base URL

Relative path: `/api/plugins/kanban/board`. Served by the Kanban dashboard
plugin on the same Hermes server. No separate env var.

### Endpoint

#### Get board state

```
GET /api/plugins/kanban/board
```

Optional query parameters:

| Param | Type | Description |
|---|---|---|
| `board` | string | Board slug (default: active board) |
| `tenant` | string | Filter tasks by tenant namespace |
| `include_archived` | boolean | Include archived tasks (default: false) |

Response — raw board JSON with `columns[]` each containing a `name` (column
id string) and `tasks[]`. The hook normalises `name` → `status` for consumers.

Raw server shape:

```ts
type RawBoardResponse = {
  columns: Array<{
    name: string;   // "triage" | "todo" | "ready" | "running" | "blocked" | "done" | ...
    tasks: KanbanTask[];
  }>;
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;      // server-side Unix timestamp (seconds)
};
```

### Hook: `useKanbanState`

File: `web/src/hooks/useKanbanState.ts`

```ts
import { useKanbanState } from "@/hooks/useKanbanState";

function KanbanWidget() {
  const { data, loading, error, refresh } = useKanbanState({
    board: "ai-dev",
    pollingIntervalMs: 30_000,
  });

  if (loading) return <Spinner />;
  if (error)   return <ErrorBanner message={error} />;

  const running = data?.columns.find(c => c.status === "running")?.tasks.length ?? 0;

  return <div>{running} tasks running</div>;
}
```

**State shape** (`KanbanStateResult`):

```ts
type KanbanStateResult = {
  data: KanbanStateSnapshot | null;
  loading: boolean;
  error: string | null;      // string, not Error instance
  refresh: () => Promise<void>;
};

type KanbanStateSnapshot = {
  columns: KanbanColumn[];
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;
};

/** Note: the hook normalises raw `name` → `status` on each column. */
type KanbanColumn = {
  status: string;   // "triage" | "todo" | "ready" | "running" | "blocked" | "done" | ...
  tasks: KanbanTask[];
};

type KanbanTask = {
  id: string;
  title: string | null;
  status: string;
  assignee: string | null;
  priority: number;
};
```

**Options** (`KanbanStateOptions`):

```ts
type KanbanStateOptions = {
  pollingIntervalMs?: number;  // 0 or omitted = no polling
  board?: string;              // board slug; omit for server's active board
};
```

**Loading / error contract**:

- `loading: true` from mount until first response.
- On success: `loading: false`, `data` populated, `error: null`.
- On failure: `loading: false`, `error` is a string message, `data: null`.
  Errors are suppressed silently in the sidebar — callers inspect `error`
  for optional error badges.
- Stale-request guard: interleaved responses from overlapping `refresh()`
  calls are discarded via a per-call request ID.
- `refresh()` resets `loading` and re-fetches; safe to call on user demand
  or on a polling interval.

**Polling**:

```ts
// Poll every 30 seconds automatically:
const { data } = useKanbanState({ pollingIntervalMs: 30_000 });

// Poll manually:
const { data, refresh } = useKanbanState();
setInterval(refresh, 30_000);
```

---

## 3. Memory service

Surfaces the agent's built-in memory stores (`memory` and `user` targets)
so the dashboard can display and edit the durable facts the agent writes about
itself and the user.

### Base URL

Relative paths: `/api/memory` for provider/status and `/api/memory/content`
for per-target built-in memory read/write. No separate env var.

### Endpoints

#### Get memory status

```
GET /api/memory
```

Response — `MemoryStatus`:

```ts
export interface MemoryStatus {
  active: string;                  // active provider name, or "" for built-in
  providers: MemoryProviderInfo[];
  builtin_files: { memory: number; user: number };  // file sizes in bytes
}

export interface MemoryProviderInfo {
  name: string;
  description: string;
  configured: boolean;
}
```

#### Reset memory

```
POST /api/memory/reset
Content-Type: application/json

{ "target": "all" | "memory" | "user" }
```

Response — `{ ok: boolean; deleted: string[] }`.

#### Set memory provider

```
PUT /api/memory/provider
Content-Type: application/json

{ "provider": "honcho" }
```

Response — `{ ok: boolean; active: string }`.

#### Read memory content

```
GET /api/memory/content?target=memory
```

`target` must be `"memory"` (MEMORY.md) or `"user"` (USER.md). Missing files
return an empty store.

Response — `MemorySnapshot`.

#### Write memory content

```
PUT /api/memory/content
Content-Type: application/json

{ "target": "memory", "content": "raw markdown" }
```

Response — `{ ok: boolean; char_count: number; char_limit: number }`.

### Hook: `useMemory` / `useMemoryData`

File: `web/src/hooks/useMemoryData.ts` (core implementation)
Re-export: `web/src/hooks/useMemory.ts` (`useMemory` = alias for `useMemoryData`)

`useMemory` is a thin re-export of `useMemoryData` for import-path alignment.
Both names refer to the same hook.

```ts
import { useMemory } from "@/hooks/useMemory";
// or equivalently:
import { useMemoryData } from "@/hooks/useMemoryData";

function MemoryWidget() {
  const { data, loading, error, refetch, save } = useMemory("memory");

  if (loading) return <Spinner />;
  if (error)   return <ErrorBanner message={error} />;
  if (!data)   return null;

  return (
    <div>
      <p>{data.char_count} / {data.char_limit} chars used</p>
      <ul>
        {data.entries.map((e, i) => <li key={i}>{e.text}</li>)}
      </ul>
    </div>
  );
}
```

**State shape** (`MemoryDataState`):

```ts
type MemoryDataState = {
  data: MemorySnapshot | null;
  loading: boolean;
  error: string | null;
  refetch: (options?: { keepExisting?: boolean }) => Promise<void>;
  save: (content: string) => Promise<void>;
};

type MemorySnapshot = {
  content: string;           // raw markdown content
  entries: MemoryEntry[];    // parsed entries
  char_count: number;
  char_limit: number;
  target: MemoryTarget;
};

type MemoryEntry = {
  text: string;
};

type MemoryTarget = "memory" | "user";
```

**Adapter injection** — the hook accepts an optional `MemoryDataClient`
argument for testing or transport swapping:

```ts
type MemoryDataClient = {
  fetchMemoryState: (target: MemoryTarget) => Promise<MemorySnapshot>;
  updateMemoryState: (
    target: MemoryTarget,
    content: string,
  ) => Promise<{ success: boolean; char_count: number; char_limit: number }>;
};

// Use default (live REST):
useMemory("memory")

// Use a mock in tests:
const mockClient = {
  fetchMemoryState: async () => mockSnapshot,
  updateMemoryState: async () => ({ success: true, char_count: 0, char_limit: 2200 }),
};
useMemoryData("memory", mockClient)
```

The default live adapter calls `api.getMemoryContent()`
(`GET /api/memory/content`) and `api.updateMemory()`
(`PUT /api/memory/content`).

**Loading / error contract**:

- `loading: true` from mount until first response.
- On success: `loading: false`, `data` populated, `error: null`.
- On failure: `loading: false`, `error` holds the message string, `data: null`
  (unless `keepExisting: true` was passed to `refetch`).
- Auto-refetches when `target` changes.
- `save(content)` sends the updated string to the backend, then calls
  `refetch({ keepExisting: true })` so local state stays consistent.
- Stale-request guard: overlapping `refetch()` calls are deduplicated via
  request IDs; stale responses are silently discarded.

---

## Authentication

The dashboard uses a short-lived session token injected by the Python server
into `window.__HERMES_SESSION_TOKEN__`. The `fetchJSON` helper reads this value
and attaches it to every `/api/` request as:

```
X-Hermes-Session-Token: <token>
```

You do not need to handle this manually — `fetchJSON` (and therefore all `api.*`
methods and the hooks above) does it automatically. Endpoints that require
elevated auth (e.g. `api.revealEnvVar`) call `getSessionToken()` explicitly and
pass the header directly.

**Errors**: any non-2xx response throws `new Error("<status>: <body>")`. All
hooks surface this as the `error` field (string for `useAgentProfiles` and
`useMemory`; Error instance for `useKanbanState`).

---

## Local vs production

| Mode | Backend origin | How to configure |
|---|---|---|
| Local dev (Vite) | `http://127.0.0.1:9119` (default) | Set `HERMES_DASHBOARD_URL` env var or `web/.env.local` |
| Local dev (remote) | your remote Hermes host | `HERMES_DASHBOARD_URL=http://remote:9119 npm run dev` |
| Production | same origin as the dashboard | No config needed — `hermes dashboard` handles it |

The Vite dev plugin (`hermesDevToken` in `web/vite.config.ts`) automatically
scrapes the session token from the running backend's `index.html` so dev-mode
`/api/` calls are authenticated without manual token management.

---

## Cross-reference

| Service | Hook file | Client methods | Endpoint |
|---|---|---|---|
| Agent profiles | `web/src/hooks/useAgentProfiles.ts` | `api.getProfiles()`, `api.createProfile()`, `api.renameProfile()`, `api.deleteProfile()` — `web/src/lib/api.ts` | `GET /api/profiles` |
| Kanban board state | `web/src/hooks/useKanbanState.ts` | `fetchJSON("/api/plugins/kanban/board")` — called directly inside the hook | `GET /api/plugins/kanban/board` |
| Memory | `web/src/hooks/useMemoryData.ts` (re-exported as `useMemory` from `useMemory.ts`) | `api.getMemoryContent()`, `api.updateMemory()` — `web/src/lib/api.ts` | `GET/PUT /api/memory/content` |

TypeScript interfaces for all three services:

- `ProfileInfo` — `web/src/lib/api.ts`
- `KanbanStateSnapshot`, `KanbanTask`, `KanbanColumn` — `web/src/hooks/useKanbanState.ts`
- `MemorySnapshot`, `MemoryEntry`, `MemoryTarget`, `MemoryDataState` — `web/src/hooks/useMemoryData.ts`
- `MemoryStatus`, `MemoryProviderInfo` — `web/src/lib/api.ts`
