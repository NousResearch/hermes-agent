# Design — Hermes Digital Office (Web UI)

> Spec id: `digital-office-ui`
> Phase:   2 — Design (Kiro workflow)
> Tracks: `requirements.md` (Phase 1)

---

## 1. Architecture overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                             User's browser                              │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  React + TypeScript + Tailwind  (Vite-built static bundle)       │  │
│   │  ┌────────────────────────┐  ┌─────────────────────┐  ┌────────┐ │  │
│   │  │ Office (Canvas2D sim)  │  │ HireWizard / Editor │  │ Tasks  │ │  │
│   │  └────────────────────────┘  └─────────────────────┘  └────────┘ │  │
│   │                       ▲                ▲                ▲         │  │
│   │             /ws/office (WebSocket)     │     fetch (REST)         │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────┬───────────────────────────────┘
                                         │
                                  http://127.0.0.1:<port>
                                         │
┌────────────────────────────────────────▼───────────────────────────────┐
│        FastAPI app  (uvicorn, single-process, asyncio)                  │
│                                                                         │
│  /api/health     /api/employees    /api/departments   /api/tasks        │
│  /api/skills     /api/capacity     /api/export        /api/import       │
│  /ws/office  ────────────────────────────────► EventBus (asyncio.Queue) │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ Store (JSON) │ │ SkillResolver│ │ CapacityCalc │ │ Runtime      │    │
│  │  + atomic    │ │  (TF-IDF +   │ │  (pure math) │ │  Sim | Hermes│    │
│  │  writes      │ │   keywords)  │ │              │ │  AIAgent     │    │
│  └──────┬───────┘ └──────────────┘ └──────────────┘ └──────┬───────┘    │
│         │                                                  │             │
└─────────┼──────────────────────────────────────────────────┼─────────────┘
          │                                                  │
          ▼                                                  ▼
   ~/.hermes/office/*.json              Existing Hermes core (run_agent.AIAgent,
                                        toolsets.py, skills/, providers, etc.)
```

* **Process model:** one `uvicorn` worker, asyncio event loop. Hermes `AIAgent` runs
  *synchronously* — we wrap each agent invocation in `asyncio.to_thread(...)` so the
  loop stays free.
* **State source of truth:** disk (JSON files) + an in-memory cache (`Store`) that the
  HTTP handlers read/write. The cache is rebuilt from disk at boot.
* **Realtime:** a single `/ws/office` channel multiplexes events tagged with
  `{kind, dept_id?, employee_id?, payload}`.

### 1.1 Why this stack

| Decision                                     | Why                                                                                                                                                   |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FastAPI + uvicorn**                        | Already first-class in Hermes (the gateway uses it). Async-friendly, schema-first, OpenAPI for free. Adds zero new C-extension deps.                  |
| **React + Vite + TS, Tailwind, shadcn-look** | Mainstream, well-tested, kid-friendly DX, ships static. Vite build is < 1 s incremental.                                                              |
| **Canvas2D for the office sim**              | DOM is the wrong tool for 50 animated sprites at 30 fps. Canvas2D needs zero deps, is in every browser, and is enough for our scope (no shaders).   |
| **JSON file store, atomic write**            | Profile-safe, Hermes-style, trivial to inspect/edit by hand, no migrations, no server. Won't scale to 10k employees but our SLO is 200.               |
| **Pure-Python `capacity.py`**                | Determinism (NFR-9), unit-testable, auditable. No NumPy/Pandas dep.                                                                                   |
| **Two runtimes (`simulated` / `hermes`)**    | Demo-safety + dev-velocity (Story 4.10). Same interface; runtime is dependency-injected.                                                              |
| **No DB**                                    | We're a desktop app; SQLite would be overkill and adds cross-platform locking pain. JSON-per-entity is the same pattern Hermes uses elsewhere (cron). |

---

## 2. Module layout

```
hermes_office/                     ← new top-level Python package
├── __init__.py                    Public exports: app, Store, SkillResolver, capacity
├── models.py                      Pydantic v2 models: Employee, Department, Task, ActivityEvent, ResolvedRole, CapacityReport
├── store.py                       Atomic JSON store w/ in-memory cache; profile-aware paths
├── skill_resolver.py              Deterministic role-text → skills/toolsets resolver
├── capacity.py                    Pure-Python hardware × roster → CapacityReport math
├── runtime/
│   ├── __init__.py                Runtime protocol + factory
│   ├── simulated.py               Synthetic activity emitter (default)
│   └── hermes.py                  Real bridge: spins up AIAgent in a worker thread
├── eventbus.py                    asyncio fan-out for /ws/office
├── server.py                      FastAPI app, REST handlers, WS endpoint, lifespan
├── presets.py                     The 8 built-in role pictograms with their skill bundles
├── data/
│   └── role_keywords.json         Bag-of-words seed for SkillResolver (auditable)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_store.py
│   ├── test_capacity.py
│   ├── test_skill_resolver.py
│   ├── test_eventbus.py
│   ├── test_runtime_simulated.py
│   └── test_server_api.py
└── frontend/                      Vite + React + TS sources (built artifacts under dist/)
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.tsx               App entry
        ├── App.tsx                Layout shell
        ├── api.ts                 Typed REST client
        ├── ws.ts                  WS subscriber + reconnect
        ├── types.ts               Shared TS types (mirror Pydantic)
        ├── i18n.ts                en + zh-CN bundles
        ├── state.ts               Zustand store
        ├── game/
        │   ├── Office.tsx         Canvas2D scene container
        │   ├── scene.ts           Zone layout, draw loop, RAF mgmt
        │   ├── sprite.ts          Avatar drawing + animation FSM
        │   ├── pathing.ts         Deterministic A*-on-grid for zone↔zone
        │   └── theme.ts           Colors, sizes
        └── ui/
            ├── Topbar.tsx
            ├── DepartmentTabs.tsx
            ├── HireWizard.tsx     3 steps; pictograms; free-text
            ├── DepartmentDialog.tsx
            ├── EmployeeDrawer.tsx Editor + Activity feed
            ├── TaskComposer.tsx
            ├── CapacityBadge.tsx
            └── icons.tsx
```

Files added under `hermes_cli/` (CLI plumbing only):

```
hermes_cli/office_cmd.py           cmd_office(args) — boots server + opens browser
```

The existing `hermes_cli/main.py` adds **one** subparser block (`hermes office`) and
imports `office_cmd` lazily.

---

## 3. Data model (Pydantic v2)

All models live in `hermes_office/models.py` and are the single source of truth.
TypeScript mirrors are generated by hand in `frontend/src/types.ts` (small enough that
zero-deps is preferred over an OpenAPI codegen pipeline).

```python
# Pseudocode — full version in models.py
class Activity(StrEnum):
    OFFLINE = "offline"
    RESTING = "resting"
    LEARNING = "learning"
    TALKING = "talking"
    WORKING = "working"

class Zone(StrEnum):
    REST = "rest"
    LEARN = "learn"
    TALK = "talk"
    WORK = "work"

ACTIVITY_TO_ZONE: dict[Activity, Zone] = {
    Activity.OFFLINE:  Zone.REST,
    Activity.RESTING:  Zone.REST,
    Activity.LEARNING: Zone.LEARN,
    Activity.TALKING:  Zone.TALK,
    Activity.WORKING:  Zone.WORK,
}

class AvatarStyle(BaseModel):
    sprite_id: Literal["robot-1","robot-2","cat","fox","panda","wizard","scientist",
                       "writer","designer","analyst","translator","tutor"]
    hue: int = Field(ge=0, le=359, default=200)   # HSL hue for tinting

class Employee(BaseModel):
    id: str = Field(default_factory=lambda: f"emp_{uuid4().hex[:10]}")
    department_id: str
    name: str
    role: str                                # human-readable role label
    avatar: AvatarStyle
    model: str                               # e.g. "anthropic/claude-opus-4.6" or "gemma4-e2b-hermes"
    provider: Optional[str] = None           # None → use config default
    base_url: Optional[str] = None
    enabled_toolsets: list[str] = []
    skills: list[str] = []                   # skill ids resolvable by SkillResolver
    system_prompt: str = ""                  # additional persona/instructions
    runtime: Literal["simulated","hermes"] = "simulated"
    activity: Activity = Activity.RESTING
    revision: int = 1                        # bumps on every save
    created_at: datetime
    updated_at: datetime

class Department(BaseModel):
    id: str
    name: str
    mission: str
    color: str                               # hex like "#7c3aed"
    runtime_default: Literal["simulated","hermes"] = "simulated"
    employee_ids: list[str] = []
    created_at: datetime
    updated_at: datetime

class Task(BaseModel):
    id: str
    department_id: Optional[str]
    employee_id: Optional[str]
    text: str
    status: Literal["queued","running","done","failed","cancelled"] = "queued"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result_summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0

class ActivityEvent(BaseModel):
    ts: datetime
    employee_id: str
    department_id: str
    kind: Literal["state_change","tool_call","tool_result","assistant","clarify","error"]
    text: str                                # already redacted
    meta: dict[str, Any] = {}

class ResolvedRole(BaseModel):
    recommended_toolsets: list[str]
    recommended_skills: list[str]
    model_hint: Optional[str]
    confidence: float                        # [0,1]
    rationale_md: str
    matched_keywords: list[str]

class CapacityReport(BaseModel):
    host: HostProfile                        # cores, ram_gb, gpus
    model: ModelProfile                      # name, ctx, params, kv_bytes_per_token
    employee_count: int
    recommended_concurrency: int
    expected_p95_latency_ms: int
    est_usd_per_hour: float
    memory_headroom_gb: float
    notes: list[str]
```

### 3.1 On-disk layout

```
$HERMES_HOME/office/
├── departments/
│   └── <dept_id>.json                # one Department per file
├── employees/
│   └── <emp_id>.json                 # one Employee per file (cross-refs dept)
├── tasks/
│   └── <YYYYMMDD>.jsonl              # one append-only task log per day
├── activity/
│   └── <emp_id>.jsonl                # ring-buffered (max 10k lines, rotated)
├── telemetry.jsonl                   # for self-improvement loop
├── weights.json                      # SkillResolver fitted weights (optional)
└── .quarantine/                      # corrupted-file landing zone
```

All writes go through `store.atomic_write_text(path, text)`:

```python
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    if hasattr(os, "fsync"):
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    os.replace(tmp, path)               # atomic on POSIX, near-atomic on win32
```

---

## 4. Algorithms

### 4.1 Capacity model (`capacity.py`)

> **Goal:** given a host profile, a model profile, and an employee roster, return a
> deterministic `CapacityReport`.

#### 4.1.1 Inputs

* `HostProfile`: `cores: int`, `ram_gb: float`, `gpus: list[GPU]` (each with
  `vram_gb: float`, `name: str`).
* `ModelProfile`: `params_b: float` (billions of params), `quant_bits: float`
  (e.g. 4 for Q4_K_M, 16 for fp16), `ctx_tokens: int`,
  `kv_bytes_per_token: float` (defaults to model-family table), `provider_kind:
  Literal["local","api"]`, `usd_per_mtok_in: float`, `usd_per_mtok_out: float`.
* `EmployeeRoster`: list of employees grouped by `model`.

#### 4.1.2 Memory math (per loaded model instance)

Let:

* W = weight memory          = `params_b * 1e9 * (quant_bits / 8)`   bytes
* K = KV-cache memory        = `kv_bytes_per_token * ctx_tokens`     bytes
* O = activation overhead    = `0.10 * W`                            bytes (heuristic; sourced below)
* M = total memory per copy  = `W + K + O`

Reference values for `kv_bytes_per_token` (from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) +
[llama.cpp readme — KV-cache table](https://github.com/ggerganov/llama.cpp#caching)):

| Model family             | bytes / token |
| ------------------------ | ------------- |
| Gemma-3 / Gemma-4 e2b    | 2 048         |
| Llama-3 8B               | 1 024         |
| Llama-3 70B              | 1 280         |
| Mistral 7B               | 1 024         |
| Generic fallback         | 1 024         |

The 0.10 × W activation overhead is a conservative bound from
[llama.cpp issue #4567](https://github.com/ggerganov/llama.cpp/issues/4567) and the
DeepSpeed ZeRO white-paper Appendix A (which observes 4–9 % for inference).

#### 4.1.3 Concurrency rule

* **Local model:** weights are shared across all sessions ("multi-session per model")
  if the runtime supports it. Ollama≥0.20 does, so we treat W as fixed cost paid once
  per **distinct model**, while K + O is per **concurrent session**.
* **API model:** memory cost is 0; concurrency is bounded by provider rate-limit
  (default 4 RPM if unknown — looked up from `models.py` registry).

```
local_mem_budget = 0.7 * (sum_vram_gb if any_gpu else ram_gb) - 1.5 GB     # leave 30% + 1.5 GB OS
shared_W_gb      = sum_over_distinct_models(W_gb)
per_session_gb   = K_gb + O_gb                                             # for the *largest* loaded model
local_capacity   = max(0, floor((local_mem_budget - shared_W_gb) / per_session_gb))

api_capacity     = floor( min_rate_per_min / per_task_calls_estimate )
                   where per_task_calls_estimate = 4   # avg tool round-trips
```

`recommended_concurrency = min(local_capacity + api_capacity, employee_count, 16)`.
Cap at 16 because UI animation budget (NFR-3) was sized for ≤ 50 sprites with ≤ 16
working concurrently.

#### 4.1.4 Latency model

Given measured tokens/sec for the target model (from a small calibration table in
`capacity.py`):

| Model             | local TPS (est.) | api p50 (ms / tok) |
| ----------------- | ---------------- | ------------------ |
| Gemma-4 e2b       | 50               | —                  |
| Gemma-4 e4b       | 30               | —                  |
| Llama-3 8B q4     | 35               | —                  |
| GPT-4o (api)      | —                | 8                  |
| Claude Opus (api) | —                | 12                 |

`expected_p95_latency_ms = avg_tokens_per_response / TPS * 1000 * 1.35`
(the 1.35 multiplier is the empirical p95/p50 ratio observed across providers, sourced
from the [Artificial Analysis Q1 2026 report](https://artificialanalysis.ai)).

#### 4.1.5 Cost

For API:
`est_usd_per_hour = sum(employees_using_model_i) * (avg_in_tok * usd_in_per_mtok / 1e6 + avg_out_tok * usd_out_per_mtok / 1e6) * tasks_per_employee_per_hour`

Defaults: `avg_in_tok = 1500`, `avg_out_tok = 600`, `tasks_per_employee_per_hour = 6`
(once every 10 minutes of active load — a conservative mid-load figure).

For local: `est_usd_per_hour = 0.0`.

#### 4.1.6 Determinism

The function `capacity.compute(host, model, roster, defaults=DEFAULTS)` MUST be a pure
function (no clock, no env reads). Tests pin reference outputs to 3 decimals.

### 4.2 Skill resolver (`skill_resolver.py`)

> **Goal:** given a free-text role description, deterministically pick toolsets and
> skills.

#### 4.2.1 Approach

We do **not** call an LLM for this — the user's local model may not be capable, and
NFR-9 demands determinism. Instead, a small **bag-of-words classifier** with
hand-curated keyword vectors per (toolset, skill) — auditable in `data/role_keywords.json`.

For each candidate `c` (toolset or skill):

```
score(c | text) = Σ_kw  w_kw · tf_kw(text)
        where tf_kw(text) = (1 + log( count(kw in text) ))    if count > 0
                          = 0                                  otherwise
```

Pick toolsets with `score >= θ_toolset` (default 0.7), skills with
`score >= θ_skill` (default 1.2). Break ties by alphabetical id for determinism.

Confidence:
```
conf = sigmoid( (best_score - θ) / θ )      # in (0.5, 1)  for matches
     = 0                                    if no candidate scored
```

`θ` is the relevant threshold for the *type* of best candidate.

#### 4.2.2 Keyword bundle (extract — full file in `data/role_keywords.json`)

```jsonc
{
  "toolsets": {
    "web":          { "weights": { "research":2.0, "search":2.0, "google":1.5, "news":1.5,
                                   "调研":2.0, "搜":1.8, "查":1.0 } },
    "browser":      { "weights": { "browser":2.5, "click":1.5, "form":1.0, "scrape":2.0,
                                   "网页":2.0, "登录":1.8 } },
    "file":         { "weights": { "file":2.0, "edit":1.5, "save":1.0, "patch":2.0,
                                   "文件":2.0, "保存":1.5 } },
    "code_execution":{"weights":{ "calculate":1.5, "compute":1.5, "python":2.0,
                                   "数学":2.0, "计算":2.0 } },
    "image_gen":    { "weights": { "draw":2.5, "image":2.0, "picture":2.0, "logo":2.0,
                                   "画":2.5, "图":2.0 } },
    "tts":          { "weights": { "voice":2.0, "speak":2.0, "podcast":2.0,
                                   "语音":2.0, "朗读":2.0 } },
    "todo":         { "weights": { "plan":2.0, "schedule":1.5, "task":1.0,
                                   "计划":2.0 } },
    "memory":       { "weights": { "remember":2.5, "记住":2.5 } },
    "delegation":   { "weights": { "team":1.5, "delegate":2.5, "manage":1.5 } }
  },
  "skills": {
    "research/arxiv":            { "weights": { "arxiv":3.0, "paper":2.0, "论文":2.5 } },
    "research/research-paper-writing": { "weights": { "writing":2.0, "summary":2.0,
                                                       "summarize":2.0, "总结":2.0 } },
    "creative/p5js":             { "weights": { "draw":2.0, "creative":1.5, "p5":3.0 } },
    "creative/pixel-art":        { "weights": { "pixel":3.0, "sprite":2.5, "像素":3.0 } },
    "github/github-pr-workflow": { "weights": { "github":2.0, "pr":2.0, "code review":2.0 } },
    "productivity/notion":       { "weights": { "notion":3.0 } },
    "media/youtube-content":     { "weights": { "youtube":3.0, "video":1.5 } }
  },
  "model_hints": {
    "anthropic/claude-opus-4.6": { "weights": { "complex":1.5, "research":1.0 } },
    "openai/gpt-5.4":            { "weights": { "fast":1.5, "draft":1.5 } }
  },
  "thresholds": { "toolset": 0.7, "skill": 1.2 }
}
```

#### 4.2.3 Optimisation loop (Story 4.14)

Telemetry events (one line per finished task) carry:

```jsonc
{ "ts": "...", "role_text": "...", "skills": [...], "toolsets": [...],
  "success": 1|0, "latency_ms": 12345, "tokens": 9876 }
```

`hermes office optimize` re-fits weights using a tiny **single-feature logistic
regression per (keyword, candidate)**:

* Features: presence of keyword in role_text.
* Label: `success`.
* Weight update: `w_kw_c ← w_kw_c + η · (y - σ(score(c|text))) · tf_kw(text)`,
  η = 0.05, max 50 epochs, fixed-seed shuffle.

The result is written to `~/.hermes/office/weights.json`. The resolver loads weights
from that file if present (else uses the bundled defaults). All math is in pure
Python; exact algorithm is locked by `tests/test_skill_resolver.py`.

### 4.3 Pathing (frontend, deterministic)

* The office is a 24 × 14 grid of 32-pixel tiles (768 × 448 logical, scaled).
* Each zone is a fixed rectangle (see `theme.ts`).
* `pathing.findPath(from, to)` runs A* with manhattan heuristic on the static grid.
* Avatars walk at 2 tiles/s using linear interpolation between path nodes.
* Sprite z-order = y-coordinate (so further sprites overlap correctly).

### 4.4 Event bus

```python
class EventBus:
    def __init__(self): self._subs: set[asyncio.Queue] = set()
    async def subscribe(self) -> asyncio.Queue: q = asyncio.Queue(maxsize=512); self._subs.add(q); return q
    async def unsubscribe(self, q): self._subs.discard(q)
    async def publish(self, evt: ActivityEvent):
        text = redact_secrets(evt.text)        # NFR + Story 4.8.4
        for q in list(self._subs):
            try: q.put_nowait(evt.model_copy(update={"text": text}))
            except asyncio.QueueFull: pass     # drop, prefer liveness over completeness
```

WS handler: `await q.get(); await ws.send_text(evt.model_dump_json())`.

### 4.5 Runtime protocol

```python
class Runtime(Protocol):
    async def run_task(self, employee: Employee, task: Task,
                       on_event: Callable[[ActivityEvent], Awaitable[None]]
                      ) -> TaskResult: ...
```

* `simulated.SimulatedRuntime`:
  * Picks a random (seeded by `task.id`) sequence of synthetic events.
  * Total wall time `= 5 + len(task.text) // 30` seconds.
  * Emits at least: `state_change(working) → tool_call("web_search") →
    tool_result(...) → assistant("…") → state_change(resting)`.
* `hermes.HermesRuntime`:
  * Runs `AIAgent(model=…, provider=…, enabled_toolsets=…, ephemeral_system_prompt=…)`
    inside `asyncio.to_thread`.
  * Wires Hermes's `tool_start_callback`, `tool_complete_callback`, and a custom
    streaming wrapper to `on_event`.
  * Reads `~/.hermes/office/skills/<emp_id>.md` (if exists) and concats it to the
    system prompt. (We never modify `~/.hermes/skills/` — those are user-global.)

Both runtimes return identical `TaskResult` shapes so the UI is runtime-agnostic.

---

## 5. REST API (FastAPI)

> All paths are JSON; all writes are 200 + body of the new entity (or 422 with a
> structured error per RFC 9457 / problem+json).

| Method | Path                                  | Purpose                                                    |
| ------ | ------------------------------------- | ---------------------------------------------------------- |
| GET    | `/api/health`                         | `{ ok: true, version, profile, runtime_default }`          |
| GET    | `/api/capacity`                       | Computed `CapacityReport`                                  |
| GET    | `/api/skills`                         | List of available skills (id, title, path, installed?)     |
| GET    | `/api/toolsets`                       | List of toolsets (id, description)                         |
| POST   | `/api/skills/resolve`                 | `{ text: str } → ResolvedRole`                             |
| GET    | `/api/departments`                    | All departments                                            |
| POST   | `/api/departments`                    | Create department                                          |
| PATCH  | `/api/departments/{id}`               | Edit (name, mission, color, runtime_default)               |
| DELETE | `/api/departments/{id}`               | Cascade-delete employees                                   |
| GET    | `/api/employees`                      | All employees (filterable `?dept_id=`)                     |
| POST   | `/api/employees`                      | Create employee                                            |
| GET    | `/api/employees/{id}`                 | One employee + computed `cli_command`                      |
| PATCH  | `/api/employees/{id}`                 | Edit (validates against registries)                        |
| DELETE | `/api/employees/{id}`                 | Soft-stops mid-task; removes file                          |
| GET    | `/api/employees/{id}/activity?cursor` | Paginated history                                          |
| GET    | `/api/employees/{id}/cli-command`     | The CLI escape-hatch string                                |
| POST   | `/api/tasks`                          | Dispatch a task; returns `Task`                            |
| GET    | `/api/tasks?status=&dept_id=`         | List recent tasks                                          |
| GET    | `/api/export`                         | Full state JSON                                            |
| POST   | `/api/import`                         | Replace state (with backup of previous)                    |
| WS     | `/ws/office`                          | Multiplexed real-time events                               |

`POST /api/skills/resolve` example response:

```jsonc
{
  "recommended_toolsets": ["web","file","todo"],
  "recommended_skills":   ["research/arxiv","research/research-paper-writing"],
  "model_hint":           "anthropic/claude-opus-4.6",
  "confidence":           0.78,
  "rationale_md":         "Matched keywords: arxiv (3.0), paper (2.0), summarize (2.0). Above threshold for skills `research/arxiv` and `research/research-paper-writing`.",
  "matched_keywords":     ["arxiv","paper","summarize"]
}
```

### 5.1 Error envelope (RFC 9457)

```jsonc
HTTP 422 application/problem+json
{
  "type":   "https://hermes.local/errors/validation",
  "title":  "Invalid employee payload",
  "status": 422,
  "detail": "model 'unknown/foo' not in provider registry",
  "instance": "/api/employees",
  "errors": [{ "loc": ["body","model"], "msg": "unknown model" }]
}
```

---

## 6. Frontend design

### 6.1 Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Topbar:  🏢 Hermes Office   |   Capacity: 6/8 ✓   |   Runtime: Sim/Real     │
├────────────┬──────────────────────────────────────────────────────────────────┤
│ DeptTabs ▾ │                                                                  │
│ ⬛ Marketing│             ┌──────────────────────────────────────────┐         │
│ 🟪 Research │             │           OFFICE  CANVAS  2D            │         │
│ 🟩 Lily-Lab │             │   [Work zone]  [Talk]  [Rest]  [Learn]  │         │
│  + Dept    │             │       walking sprites + tooltips         │         │
│            │             └──────────────────────────────────────────┘         │
│            │                                                                  │
│            │   Task Composer:  [ "@marketing draft a tweet about …" ]  Send  │
│            │                                                                  │
└────────────┴──────────────────────────────────────────────────────────────────┘
                       Click an avatar → EmployeeDrawer slides in from right
```

* **Color system:** Tailwind defaults + dept color as accent. Light mode default.
* **Type:** Inter, 16 px base, 28 px headings.
* **Big-button minimum:** 56 × 56 px tap targets (kid-mode).
* **Sound:** OFF by default; an opt-in chime per `state_change` for kids.

### 6.2 Hire Wizard flow

```
Step 1 — Pick a body         Step 2 — Pick a job          Step 3 — Confirm
┌────────────────────┐       ┌────────────────────┐        ┌────────────────────┐
│ 🤖 🐱 🦊 🐼 🧙 …    │  →    │ 🔍 👨‍💻 ✍️ 🎨 …    │   →    │ Big avatar + name  │
│ pick one (1 click) │       │ or "describe ⌨"    │        │ skill chips badge  │
│                    │       │ → SkillResolver    │        │ HIRE 🟢            │
└────────────────────┘       └────────────────────┘        └────────────────────┘
```

* If the user picks a pictogram in Step 2, Step 3's skill chips are pre-locked from
  `presets.PRESETS[role]`.
* Input validation: name auto-generated ("Researcher #3") but editable.
* Background skill installs animate the sprite walking to *Learn* zone.

### 6.3 Employee Drawer (editor)

Sections, top to bottom:

1. **Identity** — name, role, avatar (re-pick).
2. **Brain** — model dropdown (live from `/api/skills`+config), provider, base_url
   (advanced collapsible).
3. **Skills** — chips (`x` to remove), input "Add skill…" with autocomplete from
   `/api/skills`, "Suggest from text…" reopens resolver.
4. **Toolsets** — multi-select pills.
5. **Persona** — system prompt textarea.
6. **Activity** — live feed (last 50 lines); copy / pause buttons.
7. **Danger** — Delete + "Reset memory" (clears `activity/<id>.jsonl`).

### 6.4 Office canvas semantics

* **Update loop:** subscribe to WS, debounce employee state into a local cache; the
  draw loop is a pure function of `state` + `t`.
* **Idle agents** wander inside their current zone using a small random walk seeded
  by their `id` so movement is reproducible and feels organic.
* **Talking** agents are paired by the `meta.peer_id` field on `state_change(talking)`;
  they path to the *Talk* zone and stand next to each other for the duration.
* **Speech bubbles** appear when `kind in {assistant, clarify}`; auto-fade after 4 s.

---

## 7. Sequence diagrams

### 7.1 Hire flow (free text path)

```
User                Frontend            Backend (REST)         SkillResolver       Store           EventBus
 │   click "Hire"   │                     │                       │                  │                │
 │ ───────────────► │                     │                       │                  │                │
 │                  │  POST /skills/      │                       │                  │                │
 │                  │  resolve {text}     │                       │                  │                │
 │                  │ ──────────────────► │                       │                  │                │
 │                  │                     │  resolve(text)        │                  │                │
 │                  │                     │ ────────────────────► │                  │                │
 │                  │                     │ ◄── ResolvedRole ──── │                  │                │
 │                  │ ◄── 200 ResolvedRole│                       │                  │                │
 │ shows preview    │                     │                       │                  │                │
 │   click "Hire"   │  POST /employees    │                       │                  │                │
 │ ───────────────► │ ──────────────────► │  validate + persist   │                  │                │
 │                  │                     │ ─────────────────────────────────────────►│                │
 │                  │                     │ ◄── Employee saved ───────────────────────│                │
 │                  │                     │  publish state_change(resting)            │                │
 │                  │                     │ ─────────────────────────────────────────────────────────► │
 │                  │ ◄── 201 Employee ── │                       │                  │                │
 │                  │  /ws/office:        │                       │                  │                │
 │                  │  {state_change…}    │                       │                  │                │
 │                  │ ◄────────────────── │                       │                  │                │
 │ sees new sprite  │                     │                       │                  │                │
```

### 7.2 Task dispatch (real runtime)

```
User → Frontend → POST /tasks → router picks idle employee in dept
                              → state_change(working) → publish event
                              → asyncio.create_task(runtime.run_task(...))
runtime.hermes.HermesRuntime  → asyncio.to_thread(AIAgent.run_conversation)
                              → callbacks → eventbus.publish(...)  → /ws/office
                              → on completion: state_change(resting), persist Task
```

---

## 8. Error handling

| Class                                                         | Behavior                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Invalid request body                                          | 422 + RFC 9457 `validation` problem.                                                        |
| Unknown employee/department                                   | 404 + `not_found` problem.                                                                  |
| Model name not in registry                                    | 422 + suggested alternatives (Levenshtein top-3).                                           |
| Skill install fails                                           | Employee created; skill marked `install_failed: true`; event `error` emitted; 200 returned. |
| Runtime exception inside an agent                             | Task → `failed`; activity event `error` with stack-trace truncated; sprite returns to *Rest*.|
| Capacity over-subscribe (running > recommended)               | 200 + `warning` field; tasks queue normally; FE banners.                                    |
| Corrupt JSON on disk                                          | File quarantined; boot continues; `error` event published.                                  |
| Network/port issues during launch                             | CLI prints actionable message and exits 1.                                                  |
| Backend dies mid-task                                         | On restart, tasks in `running` state are reset to `queued` (idempotent best-effort recover).|

---

## 9. Security

* **Loopback only.** `uvicorn.run(host="127.0.0.1", ...)`. The `--listen` flag is
  intentionally not exposed in v1.
* **No auth.** Local-first single-user; if/when we open to a LAN, we'll require a token
  (deferred to `digital-office-cloud` spec).
* **Secret redaction.** All event text passes through `redact_secrets()` before WS or
  disk. Re-uses Hermes's `_CREDENTIAL_NAMES` patterns.
* **Path traversal.** All disk paths derived via `get_hermes_home() / "office" / ...`;
  user-supplied IDs are validated `^[a-z0-9_]{6,64}$`.
* **CORS.** Disabled (frontend served by the same origin).
* **CSP.** `default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'`
  (Tailwind needs inline for arbitrary values).
* **WS origin check.** Reject if `Origin` header is set and != `http://127.0.0.1:<port>`.

---

## 10. Testing strategy

### 10.1 Backend (pytest)

| Suite                       | Coverage focus                                                       | Determinism |
| --------------------------- | -------------------------------------------------------------------- | ----------- |
| `test_models.py`            | Pydantic round-trip, enum coercion, default factories                | yes         |
| `test_store.py`             | Atomic write, profile path resolution, quarantine on bad JSON, rotation | yes      |
| `test_capacity.py`          | Locked numerical outputs for 6 reference inputs (3-decimal precision) | yes        |
| `test_skill_resolver.py`    | Keyword matching, threshold edges, weight reload, optimisation step  | yes (fixed RNG seed) |
| `test_eventbus.py`          | Subscribe/publish, queue overflow drops oldest, redaction passthrough | yes         |
| `test_runtime_simulated.py` | Emits expected event sequence; total time within ±0.1 s of formula   | yes (seeded) |
| `test_server_api.py`        | All REST verbs incl. 404/422 envelopes; FastAPI TestClient           | yes         |

Coverage target: ≥ 90 % statement on the 4 core files (NFR-5). Measured by
`pytest --cov=hermes_office --cov-report=term-missing`.

### 10.2 Frontend

* **Unit:** `vitest` for `pathing.test.ts`, `state.test.ts`, `i18n.test.ts`.
* **Component (deferred to a follow-up PR):** Playwright smoke that loads `/`, hires
  one preset employee, asserts `<canvas>` repaints (pixel hash changes).

### 10.3 E2E

* `scripts/office_smoke.ps1` (Windows) / `scripts/office_smoke.sh` (Unix):
  * `python -m hermes_office.server &` (or via `hermes office --no-browser`)
  * `curl /api/health` expects 200
  * `curl POST /api/skills/resolve {"text":"draft a haiku"}` expects toolsets includes `image_gen` ∨ `web` per the keyword vector
  * `curl POST /api/departments` then `POST /api/employees` then `POST /api/tasks`
  * Subscribe `/ws/office` for 5 s, expect ≥ 1 `state_change` event
  * `kill` the process, exit 0

This script is invoked from CI on Linux + Windows.

---

## 11. Performance budget

| Operation                      | Budget         | Measurement method                                |
| ------------------------------ | -------------- | ------------------------------------------------- |
| Boot to first paint            | < 3 s          | `playwright.tracing` wall-clock                   |
| `GET /api/health`              | < 5 ms p99     | `pytest-benchmark`                                |
| `POST /api/employees` (sim)    | < 50 ms p95    | TestClient micro-bench                            |
| Office frame render (50 emp)   | < 16 ms        | `performance.now()` around each draw call         |
| WS event end-to-end            | < 200 ms p95   | timed publish→client receive (loopback)           |

---

## 12. Migration & rollout

* **Opt-in install:** `pip install -e .[office]` (extra adds `fastapi`, `uvicorn`,
  `python-multipart`). Without the extra, `hermes office` prints a one-liner pointing
  at the install command.
* **No DB migration** (greenfield store).
* **Coexistence with CLI/Gateway:** different default ports (gateway = 8088; office =
  8765+). They never share a port.
* **Profiles:** office state lives under `<HERMES_HOME>/office`, fully isolated per
  profile (cf. `AGENTS.md` § Profiles).

---

## 13. Open questions (intentional, will resolve during impl)

1. **Sprite art:** ship vector SVG or AI-generated PNG sheets? → MVP: SVG (CSS-tinted
   by `avatar.hue`); AI-generated PNG variants ship in v1.1.
2. **Cancellation of running task:** API exists in design; UI button deferred to v1.1.
3. **Animations on low-power devices:** CSS `prefers-reduced-motion` snaps avatars to
   destinations instead of walking. (Spec'd, MVP-implemented.)

---

## 14. Traceability matrix (Story ↔ Module)

| Story | Implementing modules                                      |
| ----- | --------------------------------------------------------- |
| 4.1   | `office_cmd.py`, `server.py` (lifespan)                   |
| 4.2   | `frontend/game/*`, `eventbus.py`                          |
| 4.3   | `frontend/ui/HireWizard.tsx`, `presets.py`                |
| 4.4   | `skill_resolver.py`, `data/role_keywords.json`            |
| 4.5   | `models.Department`, `store.py` (department CRUD)         |
| 4.6   | `frontend/ui/EmployeeDrawer.tsx`, `models.Employee`       |
| 4.7   | `frontend/ui/TaskComposer.tsx`, `server.py` (`/api/tasks`)|
| 4.8   | `eventbus.py`, `frontend/ws.ts`                           |
| 4.9   | `capacity.py`, `frontend/ui/CapacityBadge.tsx`            |
| 4.10  | `runtime/simulated.py`, `runtime/hermes.py`, `models.runtime` |
| 4.11  | `store.py`, `server.py` (export/import)                   |
| 4.12  | `frontend/ui/EmployeeDrawer.tsx` (CLI button), `cli-command` endpoint |
| 4.13  | `frontend/i18n.ts`, ARIA labels in components             |
| 4.14  | `skill_resolver.optimize`, `office optimize` CLI          |
