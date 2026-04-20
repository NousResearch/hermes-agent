# Requirements — Hermes Digital Office (Web UI for Digital-Employee Management)

> Spec id: `digital-office-ui`
> Owner:   Hermes core team
> Status:  Draft v1.0 (Kiro Phase 1 — Requirements)
> Stack tag: `web-ui`, `multi-agent`, `low-code`, `game-like-ux`

---

## 1. Introduction

This document is the **requirements** phase of the Kiro spec workflow for **Hermes Digital
Office** — a web-based, game-style management UI that lets *non-technical* users (down
to an 8-year-old) create, visualize and orchestrate fleets of Hermes-Agent instances
("digital employees") working in *departments* on real tasks.

The product is **additive**: it lives next to the existing Hermes CLI / Gateway and
re-uses the same `AIAgent` core, the same `~/.hermes/skills/` library, and the same
provider/model configuration. It does **not** modify or replace any existing surface.

### 1.1 Vision (one sentence)

> *"Open a browser, see your AI office, drag a new colleague in, type what they should
> do — they're hired, equipped with the right skills, and walking to their desk."*

### 1.2 Why this exists

* The CLI/Gateway is power-user oriented. New users need a **zero-jargon path** in.
* Building multi-agent workflows ("a research dept + a writing dept") today requires
  hand-editing YAML/JSON, picking toolsets manually, and shelling out — **way too high
  a bar** for the audience Hermes wants to reach.
* A **playful, spatial metaphor** (small-people sim) makes the otherwise abstract
  "agent fleet" *legible at a glance*: you can literally **see** who is busy, who is
  idle, who is talking to whom.
* Hermes already has all the primitives (Agent, Toolsets, Skills, Sessions). What's
  missing is the **front-of-house**.

### 1.3 In scope

| ✅ In scope                                                       | ❌ Out of scope (future spec)               |
| ----------------------------------------------------------------- | ------------------------------------------ |
| Local-first web UI bound to `127.0.0.1`                           | Multi-tenant cloud SaaS                    |
| Create / edit / delete digital employees                          | Org-wide RBAC, SSO                         |
| Create / edit / delete departments                                | Cross-org marketplace of departments       |
| 2-D top-down "office" sim with 4 zones                            | 3-D, VR, voice-pilotable office            |
| Hire wizard with text-only role definition + auto-skill picking   | LoRA / full fine-tuning of employee models |
| Live activity feed (per employee)                                 | Anomaly detection / SRE alerting           |
| Pluggable runtime: simulated *or* real Hermes `AIAgent`           | Distributed execution across machines      |
| Persistent JSON store under `~/.hermes/office/`                   | Postgres / external DB                     |
| Quantitative capacity model (Python-computed) shown to the user   | ML-based capacity predictor                |

---

## 2. Glossary

| Term              | Definition                                                                                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Digital Employee** (or *Employee*) | A persistent configuration that, when activated, instantiates a Hermes `AIAgent` with: chosen model + provider, enabled toolsets, a curated skill bundle, an `ephemeral_system_prompt`, and an avatar. |
| **Department**    | A named group of employees that share a *mission statement* and a *task queue*.  Independent from existing Hermes workflows.                                                                                       |
| **Office**        | The animated 2-D top-down view containing all departments and their employees.                                                                                                                                     |
| **Zone**          | A region inside the office: `work`, `talk`, `rest`, `learn`. Avatars walk between them based on their current activity.                                                                                            |
| **Skill**         | A markdown `SKILL.md` document under `~/.hermes/skills/` (or the bundled `skills/` dir) that gives a model task-specific procedural knowledge. Auto-installable by URL via the existing Skills Hub.                |
| **Toolset**       | A named bundle of low-level Hermes tools (e.g. `web`, `file`, `browser`) defined in `toolsets.py`.                                                                                                                  |
| **Activity**      | One of `working`, `talking`, `resting`, `learning`, `offline`. Drives the sim animation and the activity feed.                                                                                                     |
| **Hire Wizard**   | The 3-step game-like creation flow: pick avatar → pick role pictogram or describe in free text → confirm.                                                                                                          |
| **Capacity Model** | A pure-Python module that, given a hardware profile + employee roster, computes safe concurrency, expected latency, and \$ / hour.                                                                                |

---

## 3. Personas

### P1 — *"Lily"* — 8-year-old curious kid (primary)

* **Goal:** "Make a robot that draws cats."
* **Tech level:** Can use a tablet, has never seen a terminal.
* **Constraints:** Reading speed limited; relies on icons + colors.
* **Success looks like:** Three big buttons, two pictures clicked, a happy avatar walks
  to a desk and starts "working", a cat picture appears in the chat bubble.

### P2 — *"Mark"* — Solo founder / SMB owner (primary)

* **Goal:** Run a 3-person *AI Marketing Department* (researcher, copywriter,
  designer) without hiring real staff.
* **Tech level:** Comfortable installing apps, scared of YAML.
* **Success:** From `hermes office` to a working 3-employee department in < 5 minutes,
  with the dept persisting across restarts.

### P3 — *"Devika"* — Hermes power developer (secondary)

* **Goal:** Prototype new agent topologies visually before scripting them.
* **Tech level:** Reads the codebase. Wants escape hatches.
* **Success:** Can right-click any employee → "Open in CLI" to drop into a `hermes chat`
  session that resumes from the office state. Can edit raw JSON if needed.

---

## 4. User Stories (EARS format)

> Notation: each acceptance criterion uses one of the EARS keywords:
> **WHEN** (event-driven), **WHILE** (state-driven), **WHERE** (feature-conditional),
> **IF…THEN** (conditional), or **the system SHALL** (ubiquitous).

### Story 4.1 — Launch the office

> **As** any user, **I want** to start the office UI with a single command, **so
> that** I don't need to remember ports or paths.

Acceptance criteria:

1. **WHEN** the user runs `hermes office` in any directory, **the system SHALL** start
   a local FastAPI backend on `127.0.0.1:<auto-picked free port in 8765–8800>`.
2. **WHEN** the backend is healthy (responds 200 to `GET /api/health`), **the system
   SHALL** open the user's default browser to `http://127.0.0.1:<port>/`.
3. **IF** the port range is exhausted **THEN** the system **SHALL** print a clear error
   and exit non-zero.
4. **WHILE** the office is running, **the system SHALL** bind only to the loopback
   interface (no external network exposure).
5. **WHEN** the user presses `Ctrl+C` in the launching terminal, **the system SHALL**
   shut down gracefully within 3 seconds and persist any pending state.

### Story 4.2 — See the office

> **As** any user, **I want** to see all current employees as little people moving
> around an office, **so that** I understand what my AI workforce is doing at a glance.

Acceptance criteria:

1. **WHILE** any employee exists, **the system SHALL** render each as an animated
   sprite at 30 fps inside one of four labeled zones (Work / Talk / Rest / Learn).
2. **WHEN** an employee changes activity, **the system SHALL** animate the avatar
   walking from the current zone to the new zone within 2 s using deterministic
   pathing.
3. **WHILE** an employee is `working`, **the system SHALL** display a small spinning
   icon and a `┊`-style activity tag above its sprite.
4. **WHERE** more than 12 employees occupy the same zone, **the system SHALL** use a
   force-directed layout to prevent sprite overlap.
5. **WHEN** the user hovers an avatar, **the system SHALL** pop a tooltip with name,
   role, current activity, and last activity feed line.

### Story 4.3 — Hire a new employee in three clicks ("kid mode")

> **As** Lily (8 y/o), **I want** to make a new helper by tapping pictures, **so
> that** I never have to type unless I want to.

Acceptance criteria:

1. **WHEN** the user clicks the floating green **+ Hire** button on the office canvas,
   **the system SHALL** open the **Hire Wizard** in a modal.
2. **WHILE** in step 1 of the wizard, **the system SHALL** present at least 12 avatar
   pictograms; **WHEN** one is tapped, **the system SHALL** preview the choice and
   advance to step 2.
3. **WHILE** in step 2 of the wizard, **the system SHALL** present at least 8 *role
   pictograms* (Researcher 🔍, Coder 👨‍💻, Writer ✍️, Designer 🎨, Analyst 📊,
   Translator 🌐, Tutor 📚, Helper 🤝) and a "describe in your own words" textarea.
4. **IF** the user taps a pictogram **AND** does not enter free text **THEN** the
   system SHALL apply that role's preset (model + toolsets + skills) without further
   prompts.
5. **IF** the user enters free text (any language), **THEN** the system SHALL invoke
   the **Skill Resolver** (see § 4.4) and present the proposed toolsets + skills for
   one-tap confirmation.
6. **WHEN** the user confirms in step 3, **the system SHALL** create the employee,
   spawn the sprite into the *Rest* zone, then walk them to *Learn* for skill
   installation, then to *Work*.
7. **the system SHALL** allow the entire flow to be completed without typing anything
   if the user picks a role pictogram (kid-mode requirement).

### Story 4.4 — Auto-pick skills from a free-text job description

> **As** Mark, **I want** to write *"someone who can read arXiv papers and write
> tweet-thread summaries"* and get a working employee, **so that** I never read
> `toolsets.py`.

Acceptance criteria:

1. **WHEN** the wizard receives free-text input, **the system SHALL** call the
   `SkillResolver.resolve(text)` API which returns a `ResolvedRole` containing:
   `recommended_toolsets`, `recommended_skills`, `model_hint`, `confidence ∈ [0,1]`,
   and `rationale_md`.
2. **the system SHALL** compute `confidence` deterministically (see Design §4.4 —
   TF-IDF + role-keyword vector) so that the same text always yields the same value.
3. **WHEN** `confidence < 0.4` **THEN** the system SHALL prompt for one clarification
   (multi-choice) before finalizing.
4. **WHEN** a recommended skill is not yet installed locally, **the system SHALL**
   show it with a "📦 will be installed" badge; **upon confirmation**, **the system
   SHALL** install it via the existing Skills Hub (`hermes skills install <id>`) in
   the background and animate the sprite walking to the *Learn* zone for the duration
   of the install.
5. **IF** an install fails, **THEN** the employee SHALL still be created but the
   failed skill SHALL appear with a red banner in the employee card.

### Story 4.5 — Form an independent department

> **As** Mark, **I want** to spin up a brand-new department that does **not** touch my
> existing CLI workflows, **so that** experiments can't break production.

Acceptance criteria:

1. **WHEN** the user clicks **+ Department**, **the system SHALL** open a 1-step modal
   asking for: name, mission statement (free text), and color.
2. **the system SHALL** allocate a new room/area on the office canvas for the
   department, isolated by color.
3. **WHILE** a department exists, **the system SHALL** route its employees' work
   through a **dedicated `task_queue` keyed by `department_id`**; **the system SHALL
   NOT** read from or write to any other department's queue.
4. **the system SHALL** persist departments under `~/.hermes/office/departments/<id>.json`
   independently from any `~/.hermes/sessions/` state used by the CLI.
5. **WHEN** the user deletes a department, **the system SHALL** ask once for
   confirmation, then remove its employees, queue, and JSON record. (Skills installed
   are kept — they belong to the user, not the dept.)

### Story 4.6 — Edit any employee, old or new

> **As** Devika, **I want** to click any employee (created by wizard or imported) and
> change their toolsets / skills / system prompt, **so that** the UI is a real editor,
> not just a creator.

Acceptance criteria:

1. **WHEN** the user clicks an avatar, **the system SHALL** open an **Employee
   Editor** drawer with sections: *Identity*, *Brain* (model + provider), *Skills*
   (chips, removable + adder), *Toolsets* (multi-select), *Persona* (system-prompt
   textarea), *Activity* (live feed), *Danger* (delete).
2. **WHEN** the user saves, **the system SHALL** validate model & toolset names
   against the live registry; invalid values SHALL surface inline errors and SHALL
   NOT persist.
3. **the system SHALL** version each employee with a monotonically-increasing
   `revision` integer; the editor SHALL show the current revision.
4. **WHILE** an employee is mid-task, edits to *Brain* / *Skills* / *Toolsets* SHALL
   be queued and applied at the next idle transition (so we never break a running
   agent loop).

### Story 4.7 — Assign a task to a department or single employee

> **As** Mark, **I want** to type a task into a chat-like box, optionally tag it
> `@dept` or `@employee`, **so that** routing feels obvious.

Acceptance criteria:

1. **WHILE** any department or employee exists, **the system SHALL** show a global
   **Task Composer** at the bottom of the canvas.
2. **WHEN** the input contains `@<dept>`, **the system SHALL** route the task to that
   department's queue (round-robin among idle employees).
3. **WHEN** the input contains `@<employee>`, **the system SHALL** route directly.
4. **IF** no `@` mention is present **THEN** the system SHALL show a quick picker
   listing all departments + a "Pick smartest free agent" auto option.
5. **WHEN** a task is dispatched, **the system SHALL** push a `task.created` event to
   the WebSocket channel; the targeted avatar(s) SHALL animate to *Work* and a chat
   bubble SHALL appear showing the truncated task.

### Story 4.8 — Live activity feed per employee

> **As** any user, **I want** to see what each employee is "saying" / doing in
> near-real-time, **so that** the office feels alive and trustworthy.

Acceptance criteria:

1. **WHILE** an employee is active, **the system SHALL** stream a per-employee log
   over a single WebSocket multiplexed channel (`/ws/office`) at ≤200 ms latency on
   localhost.
2. **the system SHALL** classify each event as one of: `tool_call`, `tool_result`,
   `assistant`, `clarify`, `error`, `state_change`.
3. **the system SHALL** render at most the last 50 lines per employee in the editor
   drawer; **WHILE** the user scrolls up, history SHALL be paged from
   `GET /api/employees/{id}/activity?cursor=…`.
4. **the system SHALL** redact any value matching the configured secret-redaction
   regexes (re-using `hermes_cli.config.redact_secrets`) before sending over WS.

### Story 4.9 — Quantitative capacity readout

> **As** Mark, **I want** to know "can my laptop run 6 of these at once?" *without*
> guessing, **so that** I don't get a frozen UI.

Acceptance criteria:

1. **WHEN** the office boots, **the system SHALL** detect host CPU cores, total RAM,
   GPU(s) (if any) and the currently-configured model's `context_length`.
2. **the system SHALL** compute a capacity report by running the
   `hermes_office.capacity` model (pure-Python, deterministic) and SHALL display:
   `recommended_concurrency`, `expected_p95_latency_ms`, `est_usd_per_hour` (0 for
   local models), and `memory_headroom_gb`.
3. **WHEN** the roster size exceeds `recommended_concurrency`, **the system SHALL**
   show a yellow warning banner ("⚠ N employees, only K can run simultaneously —
   tasks will queue").
4. **the system SHALL** expose the same model under `GET /api/capacity` so external
   tools can introspect it.
5. **the system SHALL** include unit tests that lock the math down to the third
   decimal place for reference inputs (see `tests/test_capacity.py`).

### Story 4.10 — Mock-mode for safe demos

> **As** Lily's parent, **I want** the office to be fun without spending money on
> tokens, **so that** kids can experiment safely.

Acceptance criteria:

1. **the system SHALL** ship with a `runtime: simulated` setting that is the *default*
   on first launch.
2. **WHILE** in `simulated` mode, employees SHALL emit *plausible* but synthetic
   activity events (no real LLM calls, no real tool calls).
3. **WHEN** the user toggles to `runtime: hermes` (real mode), **the system SHALL**
   show a confirm dialog explaining "real models will be used and may incur cost".
4. **the system SHALL** persist the choice per-department so a user can keep one
   "playground" dept simulated and another "production" dept real.

### Story 4.11 — Persistence & disaster recovery

> **As** any user, **I want** my office to look the same after I restart, **so that**
> I can trust I'm not losing work.

Acceptance criteria:

1. **the system SHALL** persist all departments + employees as JSON under
   `~/.hermes/office/` after every successful mutation (atomic write via temp + rename).
2. **WHEN** the office boots, **the system SHALL** load the on-disk state and SHALL
   detect & quarantine any file that fails JSON-Schema validation (move to
   `~/.hermes/office/.quarantine/<timestamp>/`) and SHALL continue booting.
3. **the system SHALL** expose `GET /api/export` and `POST /api/import` for full-state
   round-trip (a single JSON file).
4. **WHEN** Hermes profiles are in use (`hermes -p <name>`), **the system SHALL**
   resolve the office root via `get_hermes_home() / "office"` so each profile gets its
   own isolated office (per the "Profile-safe code" rules in `AGENTS.md`).

### Story 4.12 — Escape hatch to the CLI

> **As** Devika, **I want** to drop from the office into a `hermes chat` session for
> any employee, **so that** I can debug or extend without a context switch.

Acceptance criteria:

1. **WHEN** the user clicks the "💻 Open in CLI" button on the editor drawer, **the
   system SHALL** copy a ready-to-run shell command to the clipboard
   (`hermes chat --skills <…> --model <…>` etc.) and show a toast.
2. **the system SHALL NOT** spawn the CLI itself (we don't own the user's terminal).
3. **the system SHALL** also expose `GET /api/employees/{id}/cli-command` returning
   the exact same string for power-users / scripts.

### Story 4.13 — Accessibility & i18n

> **As** any user, **I want** the UI to be usable with keyboard / screen reader and in
> Chinese or English, **so that** the kid-friendly visual layer doesn't punish a11y.

Acceptance criteria:

1. **the system SHALL** keep all interactive elements reachable via keyboard with
   visible focus rings.
2. **the system SHALL** label all sprite elements with `aria-label` ("Researcher Bot,
   currently working in Marketing department").
3. **the system SHALL** ship `en` and `zh-CN` locale bundles; **the system SHALL**
   pick the locale via `?lang=` query, then `localStorage`, then browser default.
4. **the system SHALL NOT** hard-code user-facing strings outside the locale bundles.

### Story 4.14 — Self-improvement loop ("Hermes optimizes itself")

> **As** Mark, **I want** the office itself to learn from past tasks, **so that**
> next time I hire a "research analyst" the system suggests an even better skill
> bundle.

Acceptance criteria:

1. **WHEN** a task completes, **the system SHALL** record `(role_text, used_skills,
   used_toolsets, success_signal, latency_ms, tokens)` in
   `~/.hermes/office/telemetry.jsonl`.
2. **the system SHALL** expose a CLI `hermes office optimize` that re-fits the
   `SkillResolver` weights from the telemetry (deterministically, with a fixed seed)
   and writes a new `weights.json`.
3. **the system SHALL NOT** transmit telemetry off-machine; the file is purely local.
4. **the system SHALL** keep the resolver pure-Python so users can audit the math.

---

## 5. Non-functional requirements

| #     | Requirement                                                                                                                              |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-1 | **Local-first.** Backend binds only to `127.0.0.1`. No third-party telemetry. No outbound calls except those the agents themselves make. |
| NFR-2 | **Boot < 3 s** on a 2020-era laptop with an empty office.                                                                                |
| NFR-3 | **Animation budget** ≤ 16 ms / frame at 30 fps with 50 employees on screen (Canvas2D).                                                   |
| NFR-4 | **API latency** p95 < 50 ms for read endpoints, < 150 ms for write endpoints (excluding LLM time).                                       |
| NFR-5 | **Test coverage**: ≥ 90 % statement coverage on `hermes_office/{models,store,capacity,skill_resolver}.py`.                                |
| NFR-6 | **Profile-safe.** All paths through `get_hermes_home()`; never hard-codes `~/.hermes`.                                                   |
| NFR-7 | **Backward-compatible.** Existing `hermes`, `hermes chat`, `hermes gateway` behavior unchanged.                                          |
| NFR-8 | **No new mandatory deps.** Frontend ships pre-built; backend uses `fastapi` + `uvicorn` (gated as `[office]` extra).                     |
| NFR-9 | **Determinism for tests.** Capacity model & skill resolver SHALL be deterministic given the same inputs (no clock, no RNG).              |
| NFR-10| **Crash safety.** A backend crash mid-write SHALL never corrupt store files (atomic rename, fsync on Linux/macOS, best-effort on win32). |

---

## 6. Out-of-scope (explicit)

* No multi-user collaboration / multi-cursor.
* No billing / usage-cap enforcement (Hermes already has a token budget; we surface
  it but don't enforce cents-level caps).
* No marketplace for departments (yet — see § 7).
* No "real" social mechanics between employees (they animate towards each other when
  delegating, but they don't have feelings).

---

## 7. Future spec hooks (informative)

* `digital-office-marketplace`: shareable department recipes (`.dept.json`).
* `digital-office-cloud`: optional cloud backplane with auth / RBAC.
* `digital-office-3d`: an opt-in 3-D iso view using Three.js.

---

## 8. Acceptance summary (Phase 1 exit criteria)

| #   | Criterion                                                                                                       |
| --- | --------------------------------------------------------------------------------------------------------------- |
| A1  | All 14 user stories above have at least one runnable test in Phase 4.                                           |
| A2  | The `tasks.md` derived from this document covers each story by ID (e.g. `T-4.3.x ↔ Story 4.3`).                  |
| A3  | The capacity model is implemented as pure Python and produces locked, hand-checked values (Phase 2 / Phase 4). |
| A4  | The Hire Wizard is operable end-to-end with **zero typing** for at least 8 preset roles.                       |
| A5  | The office UI degrades gracefully (no crash) when run with **0** employees, **0** departments.                  |
