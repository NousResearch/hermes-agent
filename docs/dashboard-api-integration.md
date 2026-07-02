# Dashboard API Integration

Developer reference for the three backend services consumed by the Hermes web
dashboard. Covers the base URL env var, endpoints, request/response shapes,
hook usage, and error handling.

All three services live on the **same Hermes dashboard server** and are reached
via a shared `fetchJSON<T>()` helper (`web/src/lib/api.ts`) that handles auth
and base-path injection automatically.

---

## Quick start: point the dashboard at a local backend

```bash
cd web

# Default: connects to http://127.0.0.1:9119 (same host as hermes dashboard)
npm run dev

# Remote host:
HERMES_DASHBOARD_URL=http://remote-host:9119 npm run dev
```

The Vite dev plugin (`hermesDevToken` in `web/vite.config.ts`) scrapes the
session token from the running backend's `index.html` automatically — no
manual token management needed in dev.

In production (`hermes dashboard`) the Python server injects
`window.__HERMES_SESSION_TOKEN__` and `window.__HERMES_BASE_PATH__` into
`index.html`; no extra configuration is required.

### Env vars

| Env var | Default | Purpose |
|---|---|---|
| `HERMES_DASHBOARD_URL` | `http://127.0.0.1:9119` | Backend origin for the Vite dev proxy. Set in shell or `web/.env.local`. Not used in the built bundle (same-origin). |

---

## Authentication

All `/api/` requests must carry a short-lived session token issued by the
Python server. The `fetchJSON<T>()` helper handles this transparently:

```
X-Hermes-Session-Token: <token>
```

- In loopback mode the token is injected via `window.__HERMES_SESSION_TOKEN__`.
- In gated mode (OAuth, `hermes dashboard --auth`) cookies are used; the
  helper adds `credentials: "include"` on every call.
- A 401 in loopback mode triggers a one-shot page reload to pick up a
  fresh token (server restart rotates the token). A 401 in gated mode
  redirects to `/login` via the `login_url` field in the error body.

### Error format

Any non-2xx response throws:

```ts
new Error(`${status}: ${body}`)
```

All three hooks surface this message as the `error: string | null` field in
their state. Consumers can render it directly or inspect it to distinguish
network errors from API validation failures.

---

## 1. Agent Profiles service

Manages Hermes profiles (named configuration directories under `~/.hermes/`).

### API client module

`web/src/lib/api.ts` — `api.getProfiles()` and the other `api.*Profile*`
methods. Import `fetchJSON` from the same file for one-off calls.

### Endpoints

#### List profiles

```
GET /api/profiles
```

No request body or query parameters.

Response — `{ profiles: ProfileInfo[] }`:

```ts
export interface ProfileInfo {
  name: string;        // profile directory name, e.g. "default"
  path: string;        // absolute path to the profile directory
  is_default: boolean;
  model: string | null;      // currently configured model slug
  provider: string | null;   // currently configured provider
  has_env: boolean;          // true when a .env file exists for this profile
  skill_count: number;
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

Response — `{ ok: boolean; name: string; path: string }`.

#### Delete profile

```
DELETE /api/profiles/{name}
```

Response — `{ ok: boolean }`.

#### Get setup command

```
GET /api/profiles/{name}/setup-command
```

Response — `{ command: string }`.

#### Read profile soul (system prompt)

```
GET /api/profiles/{name}/soul
```

Response — `{ content: string; exists: boolean }`.

#### Update profile soul

```
PUT /api/profiles/{name}/soul
Content-Type: application/json

{ "content": "You are a helpful assistant." }
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

State shape — `AgentProfilesState`:

```ts
type AgentProfilesState = {
  profiles: ProfileInfo[];
  loading: boolean;
  error: string | null;
  // Call with keepExisting: true for background refresh without list flicker.
  refreshProfiles: (options?: { keepExisting?: boolean }) => Promise<void>;
};
```

Adapter interface — `AgentProfilesClient`:

```ts
type AgentProfilesClient = {
  listProfiles: () => Promise<{ profiles: ProfileInfo[] }>;
};

// Live REST (default):
useAgentProfiles()

// Injected mock for tests or alternate transports:
const mockClient = { listProfiles: async () => ({ profiles: [] }) };
useAgentProfiles(mockClient)
```

Loading/error contract:

- `loading: true` from mount until the first response arrives.
- Success: `loading` false, `profiles` populated, `error` null.
- Failure: `loading` false, `error` holds the message, `profiles` reset to
  `[]` (unless `keepExisting: true` was passed).
- Stale-request guard: a newer in-flight request causes the stale response to
  be silently discarded — no interleaved state writes.

---

## 2. Kanban board state service

Reads the active Kanban board from the Kanban plugin API.

### Base URL

`/api/plugins/kanban/board` — served by the Kanban plugin on the same Hermes
server. No separate env var.

### API client module

`web/src/hooks/useKanbanState.ts` — the hook calls `fetchJSON<RawBoardResponse>`
directly (not via `api.*`). Import `fetchJSON` from `web/src/lib/api.ts` for
imperative calls.

### Endpoint

#### Get board state

```
GET /api/plugins/kanban/board
```

Optional query parameters:

| Param | Type | Description |
|---|---|---|
| `board` | string | Board slug. Omit to use the server's active board. |

Raw response — `RawBoardResponse` (internal to the hook):

```ts
type RawBoardResponse = {
  columns: Array<{ name: string; tasks: KanbanTask[] }>;
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;
};
```

The hook normalises `name` to `status` before exposing the data:

```ts
export type KanbanTask = {
  id: string;
  title: string | null;
  status: string;
  assignee: string | null;
  priority: number;
};

export type KanbanColumn = {
  status: string;  // e.g. "running", "blocked", "done" — mapped from raw `name`
  tasks: KanbanTask[];
};

export type KanbanStateSnapshot = {
  columns: KanbanColumn[];
  tenants: string[];
  assignees: string[];
  latest_event_id: number;  // monotonically increasing; useful for change detection
  now: number;              // server-side Unix timestamp (seconds)
};
```

### Hook: `useKanbanState`

File: `web/src/hooks/useKanbanState.ts`

```ts
import { useKanbanState } from "@/hooks/useKanbanState";

function SidebarWidget() {
  const { data, loading, error, refresh } = useKanbanState({
    pollingIntervalMs: 30_000,
  });

  const running =
    data?.columns.find(c => c.status === "running")?.tasks.length ?? 0;

  if (loading && !data) return <Skeleton />;

  return (
    <div>
      <p>{running} tasks running</p>
      <button onClick={refresh}>Refresh</button>
    </div>
  );
}
```

Options — `KanbanStateOptions`:

```ts
type KanbanStateOptions = {
  /** Poll the board every N milliseconds. 0 or omitted = no automatic polling. */
  pollingIntervalMs?: number;
  /** Board slug. Omit to use the server's active board. */
  board?: string;
};
```

State shape — `KanbanStateResult`:

```ts
type KanbanStateResult = {
  data: KanbanStateSnapshot | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
};
```

Loading/error contract:

- `loading: true` from mount until the first response.
- Success: `loading` false, `data` populated, `error` null.
- Failure: `loading` false, `error` holds the message. `data` is NOT reset on
  failure — the last successful snapshot is preserved so the UI doesn't blank
  on a transient network glitch. This makes the hook safe for sidebars that
  should degrade gracefully rather than show a hard error.
- Stale-request guard: uses a monotonic `requestId` ref; interleaved responses
  from concurrent `refresh()` calls are discarded.
- Polling: a `setInterval` runs `refresh()` every `pollingIntervalMs` ms. The
  interval is cleared on unmount. Polling is skipped when `pollingIntervalMs`
  is 0 or omitted.

---

## 3. Memory service

Surfaces the agent's persistent memory stores and allows provider management.

### Base URL

`/api/memory` — served directly by the Hermes FastAPI server. No separate env var.

### API client module

`web/src/lib/api.ts` — `api.getMemory()`, `api.getMemoryContent()`,
`api.updateMemory()`, `api.setMemoryProvider()`, `api.resetMemory()`.

### Endpoints

#### Get memory status

```
GET /api/memory
```

No request body or query parameters.

Response — `MemoryStatus`:

```ts
export interface MemoryStatus {
  active: string;   // name of the currently active memory provider, e.g. "builtin" or ""
  providers: MemoryProviderInfo[];
  builtin_files: {
    memory: number;  // byte size of MEMORY.md (0 if not present)
    user: number;    // byte size of USER.md   (0 if not present)
  };
}

export interface MemoryProviderInfo {
  name: string;         // e.g. "honcho", "mem0", "supermemory"
  description: string;
  configured: boolean;  // true when the provider's credentials are set
}
```

#### Set memory provider

```
PUT /api/memory/provider
Content-Type: application/json

{ "provider": "honcho" }
```

Pass `"builtin"`, `"none"`, or `""` to revert to the built-in file store.

Response — `{ ok: boolean; active: string }`.

Error: 400 if the named provider is not installed.

#### Reset memory

```
POST /api/memory/reset
Content-Type: application/json

{ "target": "all" }
```

`target` must be one of `"all"`, `"memory"` (MEMORY.md only), or `"user"`
(USER.md only). Deletes the specified built-in memory file(s).

Response — `{ ok: boolean; deleted: string[] }`.

Error: 400 on invalid `target`.

#### Read memory content

```
GET /api/memory/content?target=memory
```

`target` must be `"memory"` (MEMORY.md) or `"user"` (USER.md). Missing files
return an empty store rather than 404.

Response — `MemorySnapshot`:

```ts
{
  content: string;
  entries: Array<{ text: string }>;
  char_count: number;
  char_limit: number;
  target: "memory" | "user";
}
```

#### Write memory content

```
PUT /api/memory/content
Content-Type: application/json

{ "target": "memory", "content": "raw markdown" }
```

Response — `{ ok: boolean; char_count: number; char_limit: number }`.

Error: 400 on invalid `target`.

### Hook: `useMemory` / `useMemoryData`

File: `web/src/hooks/useMemory.ts` re-exports `useMemoryData` from
`web/src/hooks/useMemoryData.ts` under the hook-trio name.

```ts
import { useMemory } from "@/hooks/useMemory";

function MemoryWidget() {
  const { data, loading, error, refetch, save } = useMemory("memory");

  if (loading) return <Skeleton />;
  if (error)   return <ErrorBanner message={error} />;
  if (!data)   return null;

  return (
    <div>
      <p>{data.char_count} / {data.char_limit} chars used</p>
      <textarea defaultValue={data.content} />
      <button onClick={() => refetch({ keepExisting: true })}>Reload</button>
    </div>
  );
}
```

The hook accepts a `target` argument (`"memory"` or `"user"`) and an optional
`MemoryDataClient` adapter for test injection.

Adapter interface — `MemoryDataClient` (alias: `MemoryClient`):

```ts
type MemoryDataClient = {
  fetchMemoryState: (target: MemoryTarget) => Promise<MemorySnapshot>;
  updateMemoryState: (target: MemoryTarget, content: string) => Promise<{
    success: boolean;
    char_count: number;
    char_limit: number;
  }>;
};
```

The default adapter is wired to `api.getMemoryContent()`
(`GET /api/memory/content`) and `api.updateMemory()`
(`PUT /api/memory/content`). Pass a custom implementation to swap transport or
mock in tests.

`MemoryTarget`:

```ts
type MemoryTarget = "memory" | "user";
```

`MemorySnapshot` (as the hook expects from the adapter):

```ts
type MemorySnapshot = {
  content: string;   // raw memory file contents
  entries: Array<{ text: string }>;  // parsed entry list
  char_count: number;
  char_limit: number;
  target: MemoryTarget;
};
```

State shape — `MemoryDataState` (alias: `MemoryState`):

```ts
type MemoryDataState = {
  data: MemorySnapshot | null;
  loading: boolean;
  error: string | null;
  // Pull fresh data. keepExisting: true avoids clearing the stale snapshot
  // while the request is in-flight (no flicker for background refreshes).
  refetch: (options?: { keepExisting?: boolean }) => Promise<void>;
  // Persist new content, then auto-refetch to stay in sync.
  save: (content: string) => Promise<void>;
};
```

Loading/error contract:

- `loading: true` from mount until the first response.
- Success: `loading` false, `data` populated, `error` null.
- Failure: `loading` false, `error` holds the message, `data` null (unless
  `keepExisting: true` was passed to `refetch`).
- `save(content)` sets `loading: true`, calls `updateMemoryState`, then calls
  `refetch({ keepExisting: true })` to sync state. On error it sets `error` and
  drops `loading`.
- Stale-request guard: a monotonic `requestId` ref discards interleaved responses.
- Re-fetches automatically whenever `target` changes.

---

## Cross-reference

| Service | Hook file | Default API calls | Endpoint |
|---|---|---|---|
| Agent profiles | `web/src/hooks/useAgentProfiles.ts` | `api.getProfiles()` — `web/src/lib/api.ts` | `GET /api/profiles` |
| Kanban board | `web/src/hooks/useKanbanState.ts` | `fetchJSON` directly | `GET /api/plugins/kanban/board` |
| Memory | `web/src/hooks/useMemory.ts` (re-exports `useMemoryData`) | `api.getMemoryContent()`, `api.updateMemory()` — `web/src/lib/api.ts` | `GET/PUT /api/memory/content` |

TypeScript interfaces for service API responses live in `web/src/lib/api.ts`:
`ProfileInfo`, `MemoryStatus`, `MemoryProviderInfo`.

Kanban-specific types (`KanbanTask`, `KanbanColumn`, `KanbanStateSnapshot`)
are defined in `web/src/hooks/useKanbanState.ts`.

Memory hook-layer types (`MemorySnapshot`, `MemoryEntry`, `MemoryTarget`,
`MemoryDataClient`, `MemoryDataState`) are defined in
`web/src/hooks/useMemoryData.ts` and re-exported from
`web/src/hooks/useMemory.ts`.
