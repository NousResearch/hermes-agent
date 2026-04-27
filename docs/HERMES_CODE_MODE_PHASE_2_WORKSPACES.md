# Hermes Code Mode — Fase 2: CodeWorkspaceService

## 1. Resumo

Phase 2 introduces code workspace management to Hermes Agent. A workspace is a registered local project directory with detected stack, package manager, available commands, and Git metadata. Workspaces are persisted in SQLite and exposed via REST endpoints for HermesWeb to consume.

Implemented:
- `WorkspaceDB` class in `hermes_state.py` — SQLite persistence for workspaces (schema v9)
- `hermes_cli/code/workspace_service.py` — `CodeWorkspaceService` + standalone detection helpers
- 4 REST endpoints on `web_server.py`: list, open, get, refresh
- WebSocket `code_workspace.updated` event on open/refresh
- 47 tests covering stack detection, package manager, commands, Git, persistence, and REST endpoints

## 2. Arquivos alterados

| File | Why |
|------|-----|
| `hermes_state.py` | Added `WorkspaceDB` class, `code_workspaces` table in `SCHEMA_SQL`, v9 migration in `_init_schema`, bumped `SCHEMA_VERSION` to 9 |
| `hermes_cli/web_server.py` | Added 4 `/api/code/workspaces` endpoints, `_OpenWorkspaceBody` Pydantic model |
| `hermes_cli/code/__init__.py` | New package init |
| `hermes_cli/code/workspace_service.py` | New — `CodeWorkspaceService`, `detect_stack()`, `detect_package_manager()`, `detect_commands()`, `detect_git_info()` |
| `tests/hermes_cli/test_workspace_service.py` | New — 47 tests |
| `tests/test_artifacts.py` | Updated schema version assertion (8→9) |
| `tests/test_hermes_state.py` | Updated schema version assertions (8→9) |

## 3. Banco/migration

Schema version bumped 8 → 9. Migration is additive/idempotent.

```sql
CREATE TABLE IF NOT EXISTS code_workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    repo_url TEXT,
    is_git_repo INTEGER DEFAULT 0,
    branch TEXT,
    detected_stack_json TEXT DEFAULT '[]',
    package_manager TEXT,
    commands_json TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_workspaces_path ON code_workspaces(path);
CREATE INDEX IF NOT EXISTS idx_code_workspaces_updated_at ON code_workspaces(updated_at DESC);
```

- `path UNIQUE` enforces no duplicates per directory
- `detected_stack_json` / `commands_json` stored as JSON strings, deserialized on read
- `is_git_repo` stored as INTEGER 0/1, returned as Python bool
- Existing sessions/messages/artifacts/approvals tables untouched

## 4. Serviço criado

`hermes_cli/code/workspace_service.py` — `CodeWorkspaceService`

**Responsibilities:**
- Validate path (exists, is directory)
- Detect stack, package manager, commands, Git info
- Delegate persistence to `WorkspaceDB`
- Expose clean interface for REST endpoints

**Key methods:**
- `inspect_path(path)` — detect metadata without persisting
- `open_workspace(path)` — validate + detect + upsert, returns workspace dict
- `list_workspaces()` — return all workspaces ordered by `updated_at DESC`
- `get_workspace(workspace_id)` — lookup by id
- `refresh_workspace(workspace_id)` — re-detect and update existing workspace

### Stack detection (`detect_stack`)

| Signal | Stacks detected |
|--------|----------------|
| `package.json` | `node` |
| `tsconfig.json` | `typescript` |
| `vite.config.ts/js` | `vite` |
| `next.config.js/ts` | `next` |
| `tailwind.config.js/ts` | `tailwind` |
| `package.json` deps: `react`, `zustand`, etc. | `react`, `zustand`, … |
| `go.mod` | `go`, `go-module` |
| `pyproject.toml`, `requirements.txt`, etc. | `python` |
| `requirements.txt` with `fastapi` | `fastapi` |
| `Dockerfile` | `docker` |
| `docker-compose.yml` / `compose.yml` | `compose` |
| `Makefile` | `make` |

### Git detection (`detect_git_info`)

Uses subprocess (no `shell=True`, timeout=3s):
1. Checks if `.git` directory/file exists; if not, probes `git rev-parse --is-inside-work-tree`
2. Gets branch via `git branch --show-current`, falls back to `git rev-parse --abbrev-ref HEAD`
3. Gets remote URL via `git remote get-url origin`
4. Never raises — returns `{is_git_repo: false, branch: null, repo_url: null}` on any failure

### Command detection (`detect_commands`)

Sources:
- **package.json scripts**: reads `scripts` key, classifies by name/content keywords
- **Makefile**: extracts known targets (`build`, `test`, `dev`, `lint`, `typecheck`, `format`, `clean`, `run`, `install`)
- **Go**: suggests `go test ./...`, `go build ./...`, `go vet ./...` when `go.mod` exists

## 5. Endpoints adicionados

All endpoints require `Authorization: Bearer <token>` (same middleware as all `/api/*` routes).

### GET /api/code/workspaces

```json
{
  "workspaces": [
    {
      "id": "abc123",
      "name": "hermes-agent",
      "path": "/home/andrey/dev/hermes-agent",
      "is_git_repo": true,
      "branch": "feature/hermes-code-mode",
      "repo_url": "git@github.com:...",
      "detected_stack": ["python", "fastapi", "pytest"],
      "package_manager": "uv",
      "commands": [],
      "created_at": "2026-04-24T...",
      "updated_at": "2026-04-24T..."
    }
  ],
  "total": 1
}
```

Empty response: `{"workspaces": [], "total": 0}`

### POST /api/code/workspaces/open

```json
{ "path": "/home/andrey/dev/myproject" }
```

- 400 if path doesn't exist or isn't a directory
- 200 with `{"workspace": {...}}` on success
- Upserts — same path returns same id with updated metadata
- Emits `code_workspace.updated` WebSocket event

### GET /api/code/workspaces/{workspace_id}

- 404 if not found
- 200 with `{"workspace": {...}}`

### POST /api/code/workspaces/{workspace_id}/refresh

- 404 if workspace_id not found
- Re-runs detection against the stored path
- 200 with updated `{"workspace": {...}}`
- Emits `code_workspace.updated` WebSocket event

## 6. Detecção de stack

Stacks detected in this phase:

`node`, `typescript`, `react`, `next`, `vite`, `tailwind`, `zustand`, `go`, `go-module`, `python`, `fastapi`, `flask`, `django`, `pytest`, `docker`, `compose`, `make`

Additional frameworks detected via package.json devDependencies/dependencies or requirements.txt content.

## 7. Detecção de comandos

**package.json**: Reads `scripts` map, builds `{pm} run {name}` commands.

Classification by keyword matching on script name + command:
- `build`, `tsc` → `build`
- `test`, `jest`, `vitest` → `test`
- `lint`, `eslint` → `lint`
- `typecheck`, `tsc --noEmit` → `typecheck`
- `dev`, `start` → `dev`
- `format`, `prettier` → `format`
- others → `other`

**Makefile**: Parses lines matching `target:` pattern for known interesting names.

**Go**: Fixed set of `go test ./...`, `go build ./...`, `go vet ./...`.

No commands are executed. All detection is read-only filesystem inspection.

## 8. Git metadata

Detected via subprocess calls with `cwd=workspace_path, timeout=3, capture_output=True, text=True` — no `shell=True`.

| Field | Detection method |
|-------|-----------------|
| `is_git_repo` | `.git` exists, or `git rev-parse --is-inside-work-tree` returns `true` |
| `branch` | `git branch --show-current`, fallback `git rev-parse --abbrev-ref HEAD` |
| `repo_url` | `git remote get-url origin` |

All failures handled silently — safe defaults returned.

## 9. Testes executados

```
uv run pytest tests/hermes_cli/test_workspace_service.py -v
→ 47 passed

uv run pytest tests/test_artifacts.py tests/test_hermes_state.py \
             tests/hermes_cli/test_web_server.py \
             tests/hermes_cli/test_workspace_service.py
→ 318 passed, 0 failed
```

## 10. Compatibilidade

Phase 1 fully intact:
- `GET /api/sessions/{id}/artifacts` — unchanged, 271 tests still passing
- `artifact.created` WebSocket event — unchanged
- `WorkspaceDB` is a separate class; `SessionDB` untouched except schema version bump

Schema version bump 8→9 updated in 3 test assertions. Existing databases migrate cleanly via the `if current_version < 9` block.

## 11. Próximos passos — Fase 3: CodeSessionService

- Create `CodeSession` entity that links: workspace + Hermes session + task + provider/model
- `POST /api/code/sessions/start` — start a coding session in a workspace
- `GET /api/code/sessions/{id}` — session status, commands run, artifacts produced
- Track which commands were executed in the session
- Associate artifacts from Phase 1 with the CodeSession
- Emit `code_session.started`, `code_session.completed` WebSocket events
- HermesWeb can then display a full coding session timeline: task → commands → diffs
