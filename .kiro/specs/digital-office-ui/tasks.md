# Tasks — Hermes Digital Office (Web UI)

> Spec id: `digital-office-ui`
> Phase:   3 — Implementation tasks (Kiro workflow)
> Tracks: `requirements.md`, `design.md`

Each task ID maps back to one or more user stories ("Story 4.x") from
`requirements.md`. Implementations should reference the design module names from
`design.md` (e.g. `hermes_office/store.py`).

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

---

## T-0  Spec sign-off

* [x] T-0.1  Phase 1 — write `requirements.md`
* [x] T-0.2  Phase 1 — write `design.md`
* [x] T-0.3  Phase 1 — write `tasks.md` (this file)
* [ ] T-0.4  User reviews & accepts the spec triplet

---

## T-1  Repo / packaging plumbing  *(Stories 4.1, NFR-7, NFR-8)*

* [ ] T-1.1  Create `hermes_office/` package with `__init__.py` exposing
        `app`, `Store`, `SkillResolver`, `compute_capacity`.
* [ ] T-1.2  Add `[project.optional-dependencies]` extra `office = ["fastapi>=0.115,<1", "uvicorn[standard]>=0.30,<1", "python-multipart>=0.0.9,<1"]` in `pyproject.toml`.
* [ ] T-1.3  Add `package_data` / `tool.setuptools.package-data` so the built
        `hermes_office/frontend/dist/` ships with the wheel.
* [ ] T-1.4  Update `AGENTS.md` "Project Structure" with a one-paragraph mention of
        `hermes_office/`.
* [ ] T-1.5  Add a top-level `hermes_office/README.md` (English + 中文 quickstart).

## T-2  Data model layer  *(Stories 4.5, 4.6, 4.10, 4.11; NFR-6)*

* [ ] T-2.1  `models.py`: implement `Activity`, `Zone`, `AvatarStyle`, `Employee`,
        `Department`, `Task`, `ActivityEvent`, `ResolvedRole`, `HostProfile`,
        `ModelProfile`, `CapacityReport` per `design.md` §3.
* [ ] T-2.2  Validators: `Employee.id` `^emp_[a-z0-9]+$`; `Department.color`
        `^#[0-9a-fA-F]{6}$`; `model` non-empty; `Department.name` 1..50 chars;
        cross-ref check `employee.department_id ∈ store.departments`.
* [ ] T-2.3  Tests: `tests/test_models.py` — round-trip, default factories, validator
        failures.

## T-3  Persistent store  *(Stories 4.5, 4.6, 4.11; NFR-6, NFR-9, NFR-10)*

* [ ] T-3.1  `store.py`: `Store` class wrapping `~/.hermes/office/{departments,employees,tasks,activity}/`.
* [ ] T-3.2  `atomic_write_text(path, text)` per design §3.1; tests for crash safety
        (interrupt mid-write simulated via monkey-patched `os.replace` raising).
* [ ] T-3.3  `Store.boot_from_disk()`: scans both folders, validates, **quarantines
        bad files** to `.quarantine/<ts>/`, returns counts.
* [ ] T-3.4  `Store.export()` / `Store.import_(payload)` for full state round-trip.
* [ ] T-3.5  Tests: `tests/test_store.py` — CRUD, quarantine, profile-aware path
        resolution (fixture sets `HERMES_HOME`), export/import round-trip.

## T-4  Capacity model (Python math)  *(Story 4.9; NFR-9)*

* [ ] T-4.1  `capacity.py`: `compute(host, model, roster) → CapacityReport` per
        design §4.1.
* [ ] T-4.2  Bundled `MODEL_TABLE` with `kv_bytes_per_token` & TPS for the 8 most
        common models (Gemma-4 e2b/e4b, Llama-3 8B/70B, Mistral-7B, GPT-4o, Claude
        Opus, GPT-5.4); each row carries a source URL comment.
* [ ] T-4.3  Auto-detect host: `HostProfile.detect()` reads `os.cpu_count()`,
        `psutil.virtual_memory()` if available else `/proc/meminfo` else
        `wmic ComputerSystem` (Windows). Tests mock all three paths.
* [ ] T-4.4  Tests: `tests/test_capacity.py` — 6 reference inputs with hand-checked
        outputs locked to 3 decimals (e.g. *1× Gemma-4 e2b on 16 GB RAM, no GPU,
        roster of 5*).
* [ ] T-4.5  CLI sanity: `python -m hermes_office.capacity` prints a table for the
        current host using the configured Hermes model.

## T-5  Skill resolver  *(Stories 4.4, 4.14; NFR-9)*

* [ ] T-5.1  `data/role_keywords.json`: full curated keyword bundle (≥ 50 keywords
        across EN + zh-CN).
* [ ] T-5.2  `skill_resolver.py`: `resolve(text) → ResolvedRole`; pure function,
        deterministic.
* [ ] T-5.3  `optimize(telemetry_path) → weights.json`: tiny SGD per design §4.2.3,
        fixed seed.
* [ ] T-5.4  `presets.py`: 8 built-in role pictograms with hand-tuned bundles
        (Researcher, Coder, Writer, Designer, Analyst, Translator, Tutor, Helper).
* [ ] T-5.5  Tests: `tests/test_skill_resolver.py` — locked outputs for 10 sample
        prompts in EN + zh-CN; threshold edges; weights load order; `optimize()`
        deterministic with seed.

## T-6  Event bus  *(Stories 4.7, 4.8)*

* [ ] T-6.1  `eventbus.py`: `EventBus.{subscribe,unsubscribe,publish}` per design §4.4.
* [ ] T-6.2  Redaction wrapper using `hermes_cli.config.redact_secrets` (or local
        fallback if not importable).
* [ ] T-6.3  Tests: `tests/test_eventbus.py` — multi-sub fan-out, queue overflow
        drops, redaction.

## T-7  Runtime layer  *(Story 4.10)*

* [ ] T-7.1  `runtime/__init__.py`: `Runtime` Protocol + `make_runtime(kind)` factory.
* [ ] T-7.2  `runtime/simulated.py`: deterministic synthetic event sequence; total
        time `5 + len(text)//30` s; seeded by `task.id`.
* [ ] T-7.3  `runtime/hermes.py`: spins `AIAgent` in a worker thread; wires existing
        callbacks → `EventBus`. Read-only fallback if `AIAgent` import fails (still
        starts, just refuses to switch to real mode).
* [ ] T-7.4  Tests: `tests/test_runtime_simulated.py`; the hermes runtime is
        smoke-tested via `tests/test_server_api.py` (mocked `AIAgent`).

## T-8  REST + WebSocket server  *(Stories 4.1, 4.5–4.8, 4.11, 4.12)*

* [ ] T-8.1  `server.py`: FastAPI app, lifespan that boots `Store` and `EventBus`.
* [ ] T-8.2  Endpoints from design §5 (health, capacity, skills, toolsets,
        skills/resolve, departments × CRUD, employees × CRUD,
        employees/{id}/activity, employees/{id}/cli-command, tasks × CRUD,
        export/import).
* [ ] T-8.3  WS endpoint `/ws/office`; per-connection queue.
* [ ] T-8.4  RFC 9457 problem+json error envelope handler.
* [ ] T-8.5  Static-files mount serves `frontend/dist/` at `/`.
* [ ] T-8.6  Tests: `tests/test_server_api.py` covering ≥ 1 happy + ≥ 1 sad path per
        endpoint, plus a 1-second WS subscribe assertion.

## T-9  CLI launcher  *(Story 4.1)*

* [ ] T-9.1  `hermes_cli/office_cmd.py`: `cmd_office(args)` — picks a free port in
        `8765..8800`, launches `uvicorn.Server` programmatically (so we can SIGINT
        it), opens browser with `webbrowser.open` unless `--no-browser`.
* [ ] T-9.2  Add subparser block to `hermes_cli/main.py`:
        `hermes office [--port N] [--host 127.0.0.1] [--no-browser] [--reload]`.
* [ ] T-9.3  `hermes office optimize` subsubcommand for Story 4.14.
* [ ] T-9.4  Pre-flight check: if `fastapi` import fails, print install hint and
        exit 1 (no traceback).
* [ ] T-9.5  Tests: `tests/hermes_cli/test_office_cmd.py` — port-pick logic with
        sockets, install-hint branch via `monkeypatch`.

## T-10  Frontend scaffold  *(Stories 4.2–4.13)*

* [ ] T-10.1  Vite + React 18 + TypeScript template under `hermes_office/frontend/`.
* [ ] T-10.2  Tailwind + shadcn-style component primitives (`button`, `input`,
        `dialog`, `tabs`, `badge`, `tooltip`, `dropdown`).
* [ ] T-10.3  `api.ts` typed client; `ws.ts` reconnecting subscriber.
* [ ] T-10.4  `state.ts` Zustand store: `employees`, `departments`, `tasks`,
        `capacity`, `activity[employeeId]`.
* [ ] T-10.5  `i18n.ts` en + zh-CN bundles; `useT()` hook; locale resolution per
        Story 4.13.
* [ ] T-10.6  Build pipeline: `npm run build` outputs to `frontend/dist/`; the
        Python wheel includes that dist (T-1.3).

## T-11  Office canvas  *(Story 4.2)*

* [ ] T-11.1  `game/scene.ts`: zone rectangles, draw loop via `requestAnimationFrame`.
* [ ] T-11.2  `game/sprite.ts`: SVG-derived avatar sheet (8 base bodies × 4 walk
        frames), tinted by `avatar.hue`.
* [ ] T-11.3  `game/pathing.ts`: A* on a 24 × 14 grid; tests in `pathing.test.ts`.
* [ ] T-11.4  Hover tooltip; click → opens EmployeeDrawer.
* [ ] T-11.5  Reduced-motion mode (snap to destination).

## T-12  Hire wizard  *(Stories 4.3, 4.4)*

* [ ] T-12.1  `ui/HireWizard.tsx` 3-step modal with big-button styling.
* [ ] T-12.2  Avatar grid (12 pictograms).
* [ ] T-12.3  Role picker (8 pictograms + free-text).
* [ ] T-12.4  Calls `POST /api/skills/resolve` for free-text.
* [ ] T-12.5  Confirm screen lists `recommended_skills` + chips for "📦 will install".
* [ ] T-12.6  On confirm: `POST /api/employees`; sprite spawns in *Rest* and walks
        through *Learn* zone for the duration of background skill installs.

## T-13  Department creator + tabs  *(Story 4.5)*

* [ ] T-13.1  `ui/DepartmentDialog.tsx` (name, mission, color picker).
* [ ] T-13.2  `ui/DepartmentTabs.tsx` (vertical sidebar; add `+ Department` button).
* [ ] T-13.3  Filtering canvas by selected dept (others fade to 30 % opacity).

## T-14  Employee editor drawer  *(Story 4.6)*

* [ ] T-14.1  `ui/EmployeeDrawer.tsx` per design §6.3.
* [ ] T-14.2  Validation: model dropdown sourced from `/api/skills` + the Hermes
        config's known providers.
* [ ] T-14.3  "Suggest from text…" reopens `HireWizard` step 2 in *edit* mode.
* [ ] T-14.4  "Open in CLI" copies the command to clipboard with toast (Story 4.12).
* [ ] T-14.5  Persist edits via `PATCH /api/employees/{id}`; bump `revision`.

## T-15  Task composer + activity feed  *(Stories 4.7, 4.8)*

* [ ] T-15.1  `ui/TaskComposer.tsx` global bottom bar with `@dept` / `@emp`
        autocomplete chips.
* [ ] T-15.2  Wire to `POST /api/tasks`.
* [ ] T-15.3  Activity feed inside `EmployeeDrawer`: subscribed to `/ws/office`,
        last 50 lines visible, paged via `GET /api/employees/{id}/activity?cursor=`.
* [ ] T-15.4  Speech bubbles on canvas for `assistant` / `clarify`.

## T-16  Capacity badge  *(Story 4.9)*

* [ ] T-16.1  `ui/CapacityBadge.tsx` in topbar; pulls `/api/capacity`.
* [ ] T-16.2  Tooltip explains the breakdown (recommended_concurrency, p95 latency,
        \$/h, headroom) — copy is a one-liner per dimension.
* [ ] T-16.3  Yellow over-subscribe banner if `roster > recommended`.

## T-17  Asset generation  *(NFR-1; nice-to-have)*

* [ ] T-17.1  Generate hero/cover image (cute isometric office of robots) via the
        `GenerateImage` tool; commit under `hermes_office/frontend/public/cover.png`.
* [ ] T-17.2  Generate one tile-able **office floor background** (light wood + carpet
        zones) under `frontend/public/office_bg.png`.
* [ ] T-17.3  Generate **8 character portraits** under `frontend/public/avatars/<id>.png`
        used in HireWizard step 1.

## T-18  Verification & docs  *(NFR-5, NFR-2)*

* [ ] T-18.1  `pytest hermes_office/tests -q --cov=hermes_office` ≥ 90 % on the 4
        core modules.
* [ ] T-18.2  `scripts/office_smoke.ps1` Windows smoke; `office_smoke.sh` Unix.
* [ ] T-18.3  `hermes office --no-browser &` then `curl /api/health` returns 200.
* [ ] T-18.4  Take a screenshot of the office UI (post-implementation) for the
        README.
* [ ] T-18.5  Update `hermes_office/README.md` with screenshots + 中文 + English
        quickstart + `hermes office` reference.
* [ ] T-18.6  Add a one-line entry to project `README.md` linking to the new module.

## T-19  Self-improvement loop  *(Story 4.14)*

* [ ] T-19.1  Wire telemetry append in `runtime.hermes` and `runtime.simulated`
        (`success` heuristic = `task.status == "done" ∧ no error events`).
* [ ] T-19.2  `hermes office optimize`: invokes `skill_resolver.optimize(...)`,
        prints diff of weight changes.
* [ ] T-19.3  Tests: optimisation lowers training-set loss monotonically over 5
        epochs on the synthetic dataset bundled in `tests/data/telemetry_sample.jsonl`.

---

## Dependency graph (informal)

```
  T-1 ──► T-2 ──► T-3 ──► T-8 ──► T-9
                 ├─► T-4 ──► T-16 (UI badge)
                 ├─► T-5 ──► T-12 (wizard)  
                 └─► T-6 ──► T-7 ──► T-10..T-15 (frontend)
                                    │
                                    └─► T-17, T-18, T-19
```

---

## Definition of Done (whole spec)

A feature is "done" when:

1. Its tasks here are all `[x]`.
2. Its referenced user-story acceptance criteria pass via tests or
   `scripts/office_smoke.*`.
3. `pytest hermes_office/tests -q` exits 0 on Windows + Linux CI.
4. `hermes office` boots in < 3 s on a clean profile, shows the empty office, lets
   the user hire one preset employee and see them animate to *Work*.
5. The README screenshots match what the user sees on first run.
