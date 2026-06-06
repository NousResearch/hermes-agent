# CE Plan: Dashboard Organization Chart (OC) + Crew Control Implementation Breakdown

Generated: 2026-06-05 13:52 WAST
Planner: ce-planner
Kanban task: t_3904da4f
Source plan: C:/Users/alibs/Obsidian/Hermes Vault/docs/plans/hermes-dashboard-organization-chart-crew-control-plan.md
Repo/workspace: C:/Users/alibs/AppData/Local/hermes/hermes-agent

## Goal restatement

Add two new read-only Hermes Dashboard menus, `Organization Chart (OC)` and `Crew Control`, that show Lord Haikall/Jarvis a shared visual view of installed profiles, managers, departments, worker mappings, runtime/gateway status, and basic health without changing the existing Profiles page behavior or exposing secrets.

## Compatibility notes from source inspection

Current backend/dashboard structure confirmed:

- Main backend: `hermes_cli/web_server.py`
  - FastAPI app object: `app = FastAPI(...)` at module top.
  - All non-public `/api/*` endpoints are already protected by `auth_middleware`; new `/api/crew/*` endpoints should not be added to `PUBLIC_API_PATHS`, so they inherit the session-token/OAuth gate.
  - Session token header used by frontend: `X-Hermes-Session-Token`.
  - Existing profile endpoint block starts around `@app.get("/api/profiles")` in `hermes_cli/web_server.py`.
  - Existing reusable helpers near that block:
    - `_profile_to_dict(p)`
    - `_fallback_profile_dicts(profiles_mod)`
    - `_resolve_profile_dir(name)`
    - `_cron_profile_dicts()`
    - `_cron_profile_home(profile)`
    - `_call_cron_for_profile(profile, func_name, *args, **kwargs)`
- Profile discovery: `hermes_cli/profiles.py`
  - `list_profiles() -> List[ProfileInfo]` includes default and named profiles.
  - `ProfileInfo` fields already include: `name`, `path`, `is_default`, `gateway_running`, `model`, `provider`, `has_env`, `skill_count`, `alias_path`, `distribution_*`, `description`, `description_auto`.
  - `profiles._check_gateway_running(profile_dir)` uses `gateway.status.get_running_pid(profile_dir / "gateway.pid", cleanup_stale=False)`.
  - Safe config metadata helpers available: `_read_config_model(profile_dir)`, `_count_skills(profile_dir)`, `read_profile_meta(profile_dir)`.
- Gateway/runtime helpers:
  - `gateway.status.get_running_pid(...)` and `gateway.status.read_runtime_status(...)` are imported in `web_server.py` for default dashboard status.
  - For per-profile MVP, prefer `ProfileInfo.gateway_running` / `gateway.pid` existence-derived state, and optionally read only non-secret keys from `<profile_dir>/gateway_state.json` if present. Do not read logs or credentials.
- Frontend entry points:
  - `web/src/App.tsx` contains route map `BUILTIN_ROUTES_CORE` and nav array `BUILTIN_NAV_REST`.
  - Existing `ProfilesPage` is `web/src/pages/ProfilesPage.tsx`; leave behavior unchanged.
  - API client: `web/src/lib/api.ts`, with `fetchJSON<T>()` automatically adding the dashboard session token for `/api/*` calls.
  - Current `ProfileInfo` interface in `api.ts` mirrors `/api/profiles`: `name`, `path`, `is_default`, `model`, `provider`, `has_env`, `skill_count`.
- Test locations:
  - Existing backend dashboard tests: `tests/hermes_cli/test_web_server.py`, `tests/hermes_cli/test_web_server_cron_profiles.py`, etc.
  - New focused tests should be in `tests/hermes_cli/test_web_server_crew.py`.
- Frontend build command confirmed from `web/package.json`: `npm run build` = `tsc -b && vite build`.

Current profile inventory from `hermes profile list` during planning: 24 profiles: `default`, `analytics-revops`, `atlas`, `ce-debugger`, `ce-orchestrator`, `ce-planner`, `ce-reviewer`, `ce-shipper`, `ce-tester`, `ce-worker`, `content`, `creative-visual`, `cro-web`, `influencer-kol`, `lifecycle-crm`, `marketplace-ops`, `muse`, `paid-ads`, `qa`, `research`, `research-marketing`, `seo-content`, `social-community`, `supervisor`.

## Scope boundaries

In scope for MVP:

1. Add read-only backend helpers and endpoints under `/api/crew/*`.
2. Add two read-only frontend routes/pages:
   - `/organization-chart`
   - `/crew-control`
3. Add a local non-secret crew metadata overlay at `C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml`.
4. Always include every installed Hermes profile from live discovery, even if absent from metadata.
5. Display missing metadata profiles as `Unassigned / New Profiles` or `Unassigned / Legacy` with `Needs classification` reason.
6. Add backend tests for payload shape, profile inclusion, metadata fallback, invalid metadata warning, auth behavior, and no-secret fields.
7. Verify frontend build and local dashboard route visibility.

Out of scope / non-goals:

- Do not add start/stop/restart buttons for profiles or gateways.
- Do not implement metadata editing in the UI.
- Do not change `ProfilesPage` behavior, deletion, creation, SOUL editing, or styling except safe shared type/API reuse if absolutely needed.
- Do not expose `.env` contents, credential values, auth tokens, cookies, Telegram tokens, raw `auth.json`, or raw logs.
- Do not read or display private log contents; at most show counts or file pointers in later work.
- Do not push, merge, publish, deploy, or restart services without Lord Haikall approval through Jarvis.
- Do not implement trading actions or any operational buttons that could execute external commitments.

## Data contracts to implement

Create frontend types in `web/src/types/crew.ts` and keep backend payloads aligned.

### CrewProfileSnapshot

```ts
export interface CrewProfileSnapshot {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  gateway_status: "running" | "stopped" | "starting" | "failed" | "unknown";
  has_env: boolean;
  has_soul: boolean;
  skill_count: number;
  toolsets?: string[];
  last_seen_at?: string | null;
  current_task?: CrewTaskSummary | null;
  recent_error_count?: number;
}
```

Implementation note: `gateway_status` can be computed from `ProfileInfo.gateway_running` first. If a safe `gateway_state.json` read is added, read only high-level status fields and timestamps, never tokens/platform secrets.

### CrewMetadata

```ts
export interface CrewMetadata {
  display_name?: string;
  role?: string;
  level?: "main" | "manager" | "worker" | "qa" | "unknown";
  department?: string;
  manager?: string | null;
  board?: string | null;
  lanes?: string[];
  telegram_bot?: string | null;
  telegram_topic?: string | null;
}
```

Safe metadata keys allowed from YAML: `display_name`, `role`, `level`, `department`, `manager`, `board`, `lanes`, `telegram_bot`, `telegram_topic`.

### CrewNode

```ts
export interface CrewNode {
  profile: CrewProfileSnapshot;
  display_name: string;
  role: string;
  level: "main" | "manager" | "worker" | "qa" | "unknown";
  department: string;
  manager: string | null;
  board?: string | null;
  telegram_bot?: string | null;
  telegram_topic?: string | null;
  metadata_status: "classified" | "inferred" | "missing";
  health: "green" | "yellow" | "red" | "gray";
  health_reasons: string[];
}
```

### API responses

```ts
export interface CrewOrganizationResponse {
  generated_at: string;
  source: {
    profiles: string;
    metadata: string;
    metadata_exists: boolean;
    warnings: string[];
  };
  summary: CrewSummary;
  nodes: CrewNode[];
  departments: CrewDepartment[];
  unassigned: CrewNode[];
}

export interface CrewControlResponse {
  generated_at: string;
  source: CrewOrganizationResponse["source"];
  summary: CrewSummary;
  profiles: CrewNode[];
}

export interface CrewProfileDetail {
  generated_at: string;
  node: CrewNode;
}
```

## Backend implementation sequence

### Unit 1: Metadata loader

Files:

- Modify: `hermes_cli/web_server.py`
- Test: `tests/hermes_cli/test_web_server_crew.py`

Add helpers near the existing profile dashboard helpers, before `@app.get("/api/profiles")`:

- `_crew_metadata_path() -> Path`
  - Use default Hermes root, not active profile home, so the overlay is shared by Jarvis/default and named crew profiles.
  - Recommended implementation: import `hermes_cli.profiles as profiles_mod` and return `profiles_mod._get_default_hermes_home() / "crew" / "organization.yaml"`.
  - Fallback: `get_hermes_home() / "crew" / "organization.yaml"` only if default root resolution fails.
- `_load_crew_metadata() -> tuple[dict[str, Any], list[str]]`
  - Missing file returns `({}, [])`.
  - Invalid YAML or non-dict top-level returns `({}, [warning])` and must not break the dashboard.
  - Normalize missing `profiles` to `{}`.
  - Accept only safe metadata fields listed above.

Tests:

- Missing metadata returns empty metadata and no exception.
- Valid metadata returns profile records.
- Invalid YAML/non-dict returns warning and empty metadata.

### Unit 2: Safe profile snapshots

Files:

- Modify: `hermes_cli/web_server.py`
- Test: `tests/hermes_cli/test_web_server_crew.py`

Add helpers:

- `_crew_profile_dicts() -> list[dict[str, Any]]`
  - Use `profiles_mod.list_profiles()` and `_profile_to_dict(p)`.
  - On exception, use `_fallback_profile_dicts(profiles_mod)`.
- `_read_profile_soul_status(profile_dir: Path) -> bool`
  - Existence check only: `(profile_dir / "SOUL.md").exists()`.
  - Do not read the file contents for crew APIs.
- `_profile_gateway_status(profile: dict[str, Any]) -> str`
  - If `_profile_to_dict` includes `gateway_running`, use it; otherwise use known `has_env`/fallback as appropriate.
  - Suggested status: `"running"` if true, else `"stopped"`; `"unknown"` only on exception/missing profile path.
- `_profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]`
  - Return only fields from `CrewProfileSnapshot`.
  - Include `has_env` boolean from profile discovery; never read `.env` contents.
  - Include `has_soul` existence boolean.
  - Set `current_task` to `None` for MVP unless a cheap/safe Kanban summary is implemented.

Tests:

- Snapshot includes no keys named `env`, `env_values`, `token`, `secret`, `auth`, `cookie`, `auth_json`, `raw_log`, or similar raw secret fields.
- Snapshot includes `has_env` and `has_soul` booleans but no contents.
- All profiles from fake `profiles_mod.list_profiles()` are included.

### Unit 3: Merge metadata into nodes and compute health

Files:

- Modify: `hermes_cli/web_server.py`
- Test: `tests/hermes_cli/test_web_server_crew.py`

Add helpers:

- `_infer_profile_role(name: str, metadata: dict[str, Any]) -> dict[str, Any]`
  - If metadata exists for the profile, use it with `metadata_status="classified"`.
  - Defaults:
    - `default`: display `Jarvis`, role `COO / Main Agent`, level `main`, department `Executive`, manager `None`, metadata_status `inferred` if absent from YAML.
    - `atlas`: display `Atlas`, role `IT Team Manager`, level `manager`, department `IT Team`, manager `default`.
    - `muse`: display `Muse`, role `Marketing Team Manager`, level `manager`, department `Marketing Team`, manager `default`.
    - profile name `qa`: level `qa`, department from metadata if available, otherwise `Unassigned / New Profiles`.
    - all other missing metadata: display title-cased profile name, role `Unclassified profile`, level `unknown`, department `Unassigned / New Profiles`, manager `None`, metadata_status `missing`.
- `_compute_profile_health(snapshot: dict, role_meta: dict) -> tuple[str, list[str]]`
  - Green: classified/inferred and has SOUL; gateway running or not required for worker-only profile.
  - Yellow: stopped gateway, missing `.env`, missing metadata classification, missing SOUL, or stale/unknown runtime.
  - Red: config parse failure if detected, gateway status `failed`, or repeated failure summary if later added.
  - Gray: profile exists but missing metadata and no useful runtime signal.
  - Keep reasons human-readable and non-secret.
- `_build_crew_nodes() -> tuple[list[dict], dict]`
  - Return nodes plus source/warnings.
  - Sort stable by level order (`main`, `manager`, `worker`, `qa`, `unknown`) then department/name.
- `_crew_summary(nodes: list[dict]) -> dict[str, int]`
  - Include at least: `total`, `main`, `managers`, `workers`, `qa`, `unknown`, `running`, `stopped`, `green`, `yellow`, `red`, `gray`, `unassigned`.
- `_crew_departments(nodes: list[dict]) -> list[dict]`
  - Group by `department`, include manager names and node counts.

Tests:

- `default`, `atlas`, `muse` get expected inferred roles if metadata is absent.
- Unknown new profile appears in `unassigned` and has `metadata_status="missing"` / `Needs classification` reason.
- Summary counts match nodes.
- Department grouping is stable.

### Unit 4: Read-only `/api/crew/*` endpoints

Files:

- Modify: `hermes_cli/web_server.py`
- Test: `tests/hermes_cli/test_web_server_crew.py`

Add endpoints:

```python
@app.get("/api/crew/organization")
async def get_crew_organization(): ...

@app.get("/api/crew/control")
async def get_crew_control(): ...

@app.get("/api/crew/profiles/{name}")
async def get_crew_profile(name: str): ...
```

Endpoint behavior:

- `GET /api/crew/organization`
  - Return `generated_at`, `source`, `summary`, `nodes`, `departments`, `unassigned`.
- `GET /api/crew/control`
  - Return `generated_at`, `source`, `summary`, `profiles` sorted for operational table/cards.
- `GET /api/crew/profiles/{name}`
  - Validate using `profiles_mod.validate_profile_name(name)` or existing `_resolve_profile_dir(name)`.
  - Return 404 if node not found.

Auth behavior:

- Do not add these paths to `PUBLIC_API_PATHS`.
- Existing middleware will require session token for loopback and OAuth/cookie for gated mode.
- Tests can use `TestClient` with `X-Hermes-Session-Token` header from `web_server._SESSION_TOKEN` for protected calls.

### Unit 5: Initial metadata seed

File to create:

- `C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml`

This is outside the repo but explicitly approved by the source plan. Create parent directory if missing.

Seed contents should include all 24 current profiles discovered during planning:

- Executive/main:
  - `default`: Jarvis / COO / Main Agent
- Top management:
  - `atlas`: IT Team Manager, department `IT Team`, manager `default`, board `business-crew`
  - `muse`: Marketing Team Manager, department `Marketing Team`, manager `default`, board `marketing-team`
- CE/IT under Atlas:
  - `ce-orchestrator`, `ce-planner`, `ce-worker`, `ce-debugger`, `ce-tester`, `ce-reviewer`, `ce-shipper`
- Marketing under Muse:
  - `research-marketing`, `seo-content`, `creative-visual`, `social-community`, `influencer-kol`, `paid-ads`, `marketplace-ops`, `cro-web`, `lifecycle-crm`, `analytics-revops`, `qa`
- Legacy/unassigned:
  - `research`, `content`, `supervisor`

Safety requirements for metadata:

- Metadata may include bot handles/topic labels if already non-secret in the plan.
- Do not include tokens, chat IDs, cookies, API keys, `.env` values, auth file contents, or private log snippets.

Verification command:

```bash
python - <<'PY'
from pathlib import Path
import yaml
p = Path('C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml')
data = yaml.safe_load(p.read_text(encoding='utf-8'))
profiles = data.get('profiles', {})
print(len(profiles))
assert 'default' in profiles
assert 'atlas' in profiles
assert 'ce-worker' in profiles
assert 'paid-ads' in profiles
PY
```

## Frontend implementation sequence

### Unit 6: Types and API client

Files:

- Create: `web/src/types/crew.ts`
- Modify: `web/src/lib/api.ts`

Changes:

- Add exported interfaces in `web/src/types/crew.ts` for crew payloads.
- In `web/src/lib/api.ts`, import crew response types and add methods to `api`:

```ts
getCrewOrganization: () => fetchJSON<CrewOrganizationResponse>("/api/crew/organization"),
getCrewControl: () => fetchJSON<CrewControlResponse>("/api/crew/control"),
getCrewProfile: (name: string) =>
  fetchJSON<CrewProfileDetail>(`/api/crew/profiles/${encodeURIComponent(name)}`),
```

Keep existing `ProfileInfo` unchanged unless TypeScript requires only additive reuse.

### Unit 7: Reusable crew components

Files:

- Create: `web/src/components/crew/CrewHealthBadge.tsx`
- Create: `web/src/components/crew/CrewProfileCard.tsx`
- Create: `web/src/components/crew/CrewProfileDrawer.tsx`

Requirements:

- Use existing UI primitives where possible: `Badge`, `Card`, `CardContent`, `Button`, `Spinner` patterns from current pages.
- `CrewHealthBadge` maps:
  - green -> success tone/green styling
  - yellow -> warning tone
  - red -> destructive tone
  - gray -> neutral/secondary styling
- `CrewProfileCard` displays only safe fields: display name, profile name, role, department, manager, gateway status, health, model/provider, skill count, `.env` exists/missing, SOUL exists/missing.
- `CrewProfileDrawer` is read-only. If a drawer primitive is unavailable, implement a right-side fixed panel/modal using existing `Button`, `Card`, and `X` close patterns; do not introduce a new dependency.
- Drawer sections/tabs can be simple headings for MVP: Overview, Runtime, Telegram, Kanban, Skills & Tools, Config Health, Guardrails.
- Do not add action buttons for start/stop/restart.

### Unit 8: Organization Chart page and route

Files:

- Create: `web/src/pages/OrganizationChartPage.tsx`
- Modify: `web/src/App.tsx`

Page behavior:

- Fetch with `api.getCrewOrganization()` on mount.
- Show loading and error states consistent with existing pages.
- Header text:
  - Title: `Organization Chart (OC)`
  - Subtitle: `Live crew structure from Hermes profiles + crew metadata`
- Render summary cards: total profiles, managers, workers, unassigned/new, running/stopped.
- Render top node: `default` / Jarvis.
- Render manager row: `atlas`, `muse`, and future `level="manager"` profiles.
- Render department swimlanes and workers.
- Render `Unassigned / New Profiles` section with CTA text: `Classify this profile in crew metadata`.
- Include last refreshed timestamp and refresh button.

`web/src/App.tsx` changes only:

- Import page: `import OrganizationChartPage from "@/pages/OrganizationChartPage";`
- Import icons: add `UsersRound` or reuse `Users` if `UsersRound` is unavailable in current `lucide-react` version.
- Add route to `BUILTIN_ROUTES_CORE`:
  - `"/organization-chart": OrganizationChartPage,`
- Add nav item in `BUILTIN_NAV_REST` without removing/reordering existing critical items unless necessary:
  - `{ path: "/organization-chart", label: "Organization Chart (OC)", icon: UsersRound },`

### Unit 9: Crew Control page and route

Files:

- Create: `web/src/pages/CrewControlPage.tsx`
- Modify: `web/src/App.tsx`

Page behavior:

- Fetch with `api.getCrewControl()` on mount.
- Header: `Crew Control` with refresh button.
- Health summary cards: total, running, stopped, green/yellow/red/gray, unassigned.
- Filters:
  - department
  - manager
  - gateway status
  - health
  - missing env / missing SOUL toggles if easy; otherwise include as card badges.
- Render all profiles in cards/table.
- Click a profile card opens `CrewProfileDrawer`.
- Drawer data may use the already-loaded node for MVP; optionally call `api.getCrewProfile(name)` for detail refresh.

`web/src/App.tsx` changes only:

- Import page: `import CrewControlPage from "@/pages/CrewControlPage";`
- Import icon: add `SlidersHorizontal` or use an available current icon such as `Settings` if unavailable.
- Add route:
  - `"/crew-control": CrewControlPage,`
- Add nav item:
  - `{ path: "/crew-control", label: "Crew Control", icon: SlidersHorizontal },`

## Acceptance criteria / verification checklist

Backend acceptance:

- `GET /api/crew/organization` returns a 200 payload with `generated_at`, `source`, `summary`, `nodes`, `departments`, `unassigned`.
- `GET /api/crew/control` returns a 200 payload with `summary` and `profiles` containing every installed profile.
- `GET /api/crew/profiles/{name}` returns one node or 404 for an unknown valid profile name.
- New profile directories absent from metadata still appear as unassigned/missing classification.
- Invalid/missing metadata YAML does not crash the dashboard.
- Endpoints are protected by existing dashboard auth; unauthenticated `/api/crew/*` calls return 401 in loopback session-token mode.
- No response includes raw `.env` values, `auth.json`, cookies, API keys, tokens, raw Telegram tokens/chat IDs, or raw logs.

Frontend acceptance:

- Sidebar shows `Organization Chart (OC)` and `Crew Control`.
- Existing `Profiles` menu remains present and its behavior is unchanged.
- `/organization-chart` renders Jarvis/default, managers, departments, workers, and unassigned/new section.
- `/crew-control` renders health summary, filters, all profiles, and read-only detail drawer.
- Frontend build passes.
- No start/stop/restart or destructive action controls exist in the new pages.

Metadata acceptance:

- `C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml` exists and parses.
- It classifies the known 24 profiles from the planning-time inventory.
- It contains only non-secret labels/mappings.

Local verification acceptance:

- Backend tests pass.
- Frontend build passes.
- Local dashboard loads both routes at `http://127.0.0.1:9119/organization-chart` and `http://127.0.0.1:9119/crew-control`.
- Browser/manual check confirms no secret values are visible.

## Expected commands for ce-worker / ce-tester

Run from repo root:

```bash
cd C:/Users/alibs/AppData/Local/hermes/hermes-agent
python -m pytest tests/hermes_cli/test_web_server_crew.py -v --tb=short -n 0
cd web && npm run build
```

If pytest-xdist option `-n 0` is not available in this environment, retry without it:

```bash
python -m pytest tests/hermes_cli/test_web_server_crew.py -v --tb=short
```

Manual/local dashboard verification:

```bash
cd C:/Users/alibs/AppData/Local/hermes/hermes-agent
hermes dashboard
```

Then open:

```text
http://127.0.0.1:9119/organization-chart
http://127.0.0.1:9119/crew-control
```

Use the browser session token injected in the dashboard page for API calls. Do not hardcode or print tokens.

## Risks and approval gates

Risks:

- `web_server.py` is large; keep helpers isolated and additive to reduce merge conflicts.
- Existing profile dashboard code has fallback scanners; reuse them rather than building a second profile discovery path.
- Per-profile gateway status is only as good as `gateway.pid` / safe runtime state; label unknown/stopped conservatively.
- Metadata seed is outside repo; ce-worker must ensure directory creation is intentional and limited to `C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml`.
- The `qa` profile name may belong to Marketing in the source plan; keep it under Muse/Marketing unless Lord Haikall reclassifies.

Approval gates:

- Lord Haikall/Jarvis approval required before any gateway start/stop/restart controls are added.
- Approval required before push/merge/deploy/publish.
- Approval required before reading or displaying any raw logs, `.env` values, auth files, chat IDs, credential files, or private folders.
- Approval required for any behavior that changes or deletes existing profiles, modifies `ProfilesPage` behavior, or affects live gateway processes.

## Open questions

No blocker for MVP. Minor implementation decisions ce-worker can resolve safely:

1. Whether to use `UsersRound`/`SlidersHorizontal` icons or fall back to existing imported icons if the current lucide package lacks them.
2. Whether `current_task` remains `null` in MVP or uses a safe summary from existing Kanban/cron helpers. If implemented, expose only task id/title/status/profile, never prompt bodies containing secrets.
3. Exact visual layout of the org chart can be card/swimlane based first; no graph library is required for MVP.

## Concrete ce-worker handoff

Implement the MVP in this order:

1. Add backend tests in `tests/hermes_cli/test_web_server_crew.py` for metadata load, profile snapshot, unassigned inclusion, endpoint payloads, auth, and no-secret response keys.
2. Add crew metadata/profile helper functions in `hermes_cli/web_server.py` near the existing profile helper/endpoint block.
3. Add read-only `/api/crew/organization`, `/api/crew/control`, and `/api/crew/profiles/{name}` endpoints; do not add them to public paths.
4. Create `C:/Users/alibs/AppData/Local/hermes/crew/organization.yaml` with the approved 24-profile seed mapping.
5. Add `web/src/types/crew.ts` and additive API methods in `web/src/lib/api.ts`.
6. Add reusable crew components under `web/src/components/crew/`.
7. Add `OrganizationChartPage` and the `/organization-chart` route/nav item.
8. Add `CrewControlPage` and the `/crew-control` route/nav item.
9. Run backend tests and frontend build.
10. Verify local dashboard routes in browser and confirm no secrets/actions are exposed.
11. Leave a review-required handoff for ce-reviewer if code changed, with changed files and command outputs.

Do not modify `web/src/pages/ProfilesPage.tsx` unless a type-only/shared import is absolutely unavoidable. Do not implement profile/gateway lifecycle controls in this MVP.
