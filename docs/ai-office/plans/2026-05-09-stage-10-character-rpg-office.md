# Stage 10+ Character RPG AI Office Implementation Plan

> **For Hermes:** Use `writing-plans` before implementation, then use `subagent-driven-development` only after the user explicitly approves an implementation slice. This plan is product/UX and technical planning. It does not approve renderer dependencies, service restarts, mutation controls, remote exposure, or copying DeskRPG code/assets.

**Goal:** Evolve `/office` from a safe dynamic dashboard into a DeskRPG-like RPG office where each model/agent appears as its own character performing its role, while preserving the read-only, local-first, privacy-safe `OfficeState` boundary.

**Architecture:** Keep the existing safe pipeline as the source of truth: `OfficeState DTO -> character/scene projection helpers -> renderer`. Stage 10 starts with a dependency-free character scene model and CSS/SVG presentation, then gates any Phaser/Pixi/canvas adoption behind a separate dependency, bundle, accessibility, security, and license review. Characters must represent safe role/status/progress metadata, not raw prompts, transcripts, task bodies, cron scripts, logs, auth fields, or secrets.

**Tech Stack:** Hermes web dashboard, React/TypeScript, Vitest, CSS/SVG first. No Phaser/PixiJS/canvas/sprite asset dependency until a later Stage 11 renderer decision is explicitly approved.

**Last updated:** 2026-05-09 12:25 KST

---

## 0. Product north star

The target experience is:

> “각 모델이 자기 캐릭터가 되어, 오피스 안에서 자기 역할을 수행하는 모습을 RPG처럼 보여준다.”

This should feel like DeskRPG in clarity and warmth, but remain Hermes-native:

- Characters are visual projections of safe dashboard state.
- The office remains observability-first, not a fake game simulation.
- The page stays read-only; no task/cron/gateway/Kanban/topic mutation controls are introduced.
- The user should understand at a glance:
  - which model/agent role is active,
  - what kind of work it is doing,
  - where work is flowing,
  - what is blocked/error/unknown,
  - which parts are merely unavailable because a source is missing.

## 1. Non-negotiable boundaries

1. No raw data projection
   - Do not display or inspect raw prompts, transcripts, messages, task bodies, cron scripts, logs, auth fields, tokens, credentials, `.env`, or hidden fields.
   - Character labels must be role/status labels, not user/task content.

2. Read-only only
   - No mutation buttons in RPG layer.
   - No task assignment, cron start/stop, gateway restart, Kanban edits, topic-registry writes, NAS/Obsidian writes, or service restarts.

3. Browser-local dynamics only
   - Live tracking may continue to call the existing read-only `api.getOfficeState()` only when user-enabled.
   - No persistent browser history for sensitive records.

4. DeskRPG is inspiration, not source material
   - Do not copy DeskRPG code, tile maps, sprites, screenshots, or assets without explicit legal/license review.
   - Hermes should use original CSS/SVG placeholders first.

5. Accessibility remains primary
   - RPG motion must not be the only information channel.
   - Provide text equivalents: room badges, recent-change rail, safe inspector, legends.
   - Respect `prefers-reduced-motion: reduce`.

---

## 2. Stage overview

### Stage 10-A: Character role projection model, no new renderer

Status: implemented 2026-05-09 10:52 KST.

Purpose: give each model/agent a stable character identity derived from safe DTO fields.

Output:

- `OfficeCharacter` projection type.
- `buildOfficeCharacters(state, nodes)` pure helper.
- Role taxonomy for model/agent characters.
- Deterministic room placement and safe status labels.
- Tests proving raw-field avoidance.

No new dependencies.

### Stage 10-B: Character sprites with CSS/SVG only

Purpose: replace generic moving markers with recognizable role characters.

Output:

- CSS/SVG character silhouettes or glyph assemblies.
- Per-role visual grammar: researcher, coder, reviewer, automation keeper, router, analyst, unknown.
- Character nameplates/status chips using safe labels only.
- Reduced-motion fallback.

No external image assets.

### Stage 10-C: Role action loops and activity states

Purpose: make characters visibly “perform roles” without inventing private content.

Output:

- `buildOfficeCharacterActivity(character, delta)` helper.
- Safe activities: thinking, coding, reviewing, routing, waiting, blocked, scheduled, alerting.
- Tiny loop animations tied to safe state/delta buckets.
- Recent-change rail links activity text to map highlight.

No fabricated speech bubbles or hidden raw inspection.

### Stage 10-D: Room-to-room RPG flow choreography

Purpose: show work movement across the office like an RPG path system.

Output:

- `buildOfficeCharacterRoutes(previous, next)` or extension to current `changedFlows`.
- Safe path particles or walking trails tied only to room-level count/status/flow deltas.
- Static route badges under reduced-motion.
- Text equivalents in `최근 변화`.

No raw task identity or content shown.

### Stage 10-E: Safe character inspector

Purpose: let the user click a room/character and inspect only safe role metadata.

Output:

- Room buttons remain accessible primary targets.
- Optional character target may be enabled only if safe ARIA/keyboard behavior is clean.
- Inspector fields: role, room, status, source health, safe count, last safe activity bucket, redaction note.
- No prompt/task/log/script/body display.

### Stage 10-F: RPG office usability hardening

Status: implemented 2026-05-09 12:01 KST.

Purpose: ensure the RPG layer improves comprehension rather than becoming visual noise.

Output:

- Small-screen layout smoke notes.
- Dense-state caps and aggregation rules.
- Empty/error/source-unavailable character states.
- Visual regression checklist.
- Korean-first copy pass.

Implemented additions:

- `OfficeUsabilitySummary` and `buildOfficeUsabilitySummary(state, characters, options)`.
- Safe map rail labeled `Stage 10-F 사용성 점검`.
- Browser-local reduced-motion and viewport-width observation for static labels and narrow-screen posture.
- Smoke hooks: `data-office-usability` and `data-office-usability-item`.

### Stage 10-G: CSS/SVG readability modes

Status: implemented 2026-05-09 12:15 KST.

Implemented output:

- `OfficeMapDensityMode` / `OfficeMapDensityPlan`.
- `buildOfficeMapDensityPlan(mode, characters)` for 요약/표준/상세 display plans.
- Browser-local density controls with `data-office-density-controls` / `data-office-density-mode`.
- 요약/표준 character caps and folded-character copy; 요약 mode folds the recent-change rail.
- No new dependency or backend/API/schema change.

### Stage 10-H: Keyboard jump targets and responsive smoke posture

Status: implemented 2026-05-09 12:25 KST.

Implemented output:

- `OfficeMapJumpTarget` and `buildOfficeMapJumpTargets(densityPlan)`.
- Korean quick links for 지도/사용성/최근 변화/안전 정보.
- Stable focusable anchors for map canvas, usability rail, recent rail/collapsed recent notice, and safe inspector.
- Density-aware recent target: 표준/상세 point to `office-map-recent`, 요약 points to `office-map-recent-collapsed`.
- No new dependency, backend/API/schema change, mutation control, persistent storage, renderer dependency, or raw record projection.

### Stage 11: Renderer decision gate

Purpose: decide whether CSS/SVG is enough or whether a true renderer is justified.

Possible outcomes:

1. Keep CSS/SVG and continue refining original Hermes characters.
2. Adopt PixiJS for many animated sprites/particles.
3. Adopt Phaser only if tile-map/game-loop needs outweigh bundle/accessibility cost.
4. Build a tiny custom canvas renderer only if simpler than Pixi/Phaser.
5. Stop at current RPG dashboard and avoid engine maintenance.

Stage 11 requires explicit approval before adding dependencies.

---

## 3. Character role taxonomy

The first character layer should avoid claiming exact internal thoughts. It should show operational roles.

| Safe source | Character role | Korean label | Visual metaphor | Allowed status examples |
|---|---|---|---|---|
| `agents` / active sessions | Operator | 조작자 | small avatar at lobby/console | active, idle, unknown |
| model/provider metadata if safely present | Model | 모델 캐릭터 | role-colored avatar | online, idle, unavailable |
| work item counts/status | Coder / Worker | 작업자 | desk, terminal, note card | working, blocked, done bucket |
| review/quality status if present | Reviewer | 검토자 | clipboard/magnifier | reviewing, warning, clean |
| automation `next_run_at` bucket | Automation keeper | 자동화 관리인 | machine keeper / clock | overdue, soon, later, unknown |
| topics/provenance | Router / Messenger | 전달자 | mail runner / route board | routed, unknown, missing |
| source health | Sentinel | 감시자 | status lamp / guard | ok, partial, error |
| attention items | Medic / Alert | 경보 담당 | red/amber notice | attention, blocked, error |

Important:

- A “model character” may represent a model/provider/session role only when the DTO already exposes a safe label.
- If safe identity is unavailable, use generic labels such as `모델 캐릭터 1`, `작업자 2`, `전달자 1`.
- Never infer a private task title or user content from raw records.

---

## 4. Stage 10-A implementation plan: character projection model

### Task 10-A1: Add character projection types

Status: done 2026-05-09 10:52 KST.

**Objective:** Introduce a safe character model separate from raw DTO records and separate from visual rendering.

**Files:**
- Modify: `web/src/pages/officeView.ts`
- Test: `web/src/pages/OfficePage.test.ts`

**Step 1: Write failing test**

Add a Vitest case that imports `buildOfficeCharacters` before it exists and uses fixture state with raw-looking fields injected into nested records.

Expected assertions:

- characters are returned for safe roles such as model/operator/worker/router/automation.
- labels are Korean-first and generic when source identity is unsafe.
- no character label/detail/action contains raw-looking fixture values.
- every character has a `roomId`, `role`, `status`, `x`, `y`, and `redactionNote`.

**Step 2: Run RED**

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts
```

Expected: fail because `buildOfficeCharacters` is not exported.

**Step 3: Implement minimal helper**

Suggested type shape:

```ts
export type OfficeCharacterRole =
  | "model"
  | "operator"
  | "worker"
  | "reviewer"
  | "automation_keeper"
  | "router"
  | "sentinel"
  | "alert";

export type OfficeCharacterStatus =
  | "active"
  | "idle"
  | "working"
  | "reviewing"
  | "routing"
  | "scheduled"
  | "blocked"
  | "warning"
  | "error"
  | "unknown";

export type OfficeCharacter = {
  id: string;
  role: OfficeCharacterRole;
  roomId: OfficeMapNode["id"];
  label: string;
  status: OfficeCharacterStatus;
  detail: string;
  redactionNote: string;
  x: number;
  y: number;
};
```

Helper:

```ts
export function buildOfficeCharacters(state: OfficeState, nodes: OfficeMapNode[]): OfficeCharacter[]
```

Rules:

- Use only safe DTO counts/status/source health fields.
- Use bounded characters per room, e.g. 3 visible per role plus a safe `+N` aggregate character if needed.
- Deterministic coordinates based on existing room node positions and fixed offsets.
- Prefer Korean generic labels:
  - `모델 캐릭터 1`
  - `작업자 1`
  - `자동화 관리인 1`
  - `전달자 1`
  - `감시자 1`

**Step 4: Run GREEN**

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts
```

Expected: pass.

### Task 10-A2: Add character-to-scene adapter

Status: done 2026-05-09 10:52 KST.

**Objective:** Keep current scene markers compatible while allowing the RPG character layer to grow.

**Files:**
- Modify: `web/src/pages/officeView.ts`
- Test: `web/src/pages/OfficePage.test.ts`

Add:

```ts
export function buildOfficeCharacterSceneObjects(characters: OfficeCharacter[]): OfficeSceneObject[]
```

Rules:

- `model`, `operator`, `worker`, `reviewer` map to avatar/desk-like safe objects.
- `automation_keeper` maps to machine.
- `router` maps to mail.
- `sentinel` / `alert` maps to alert/machine depending on severity.
- Preserve current `buildOfficeSceneMotionTrack` compatibility.

### Task 10-A3: Integrate behind current map without removing existing markers

Status: done 2026-05-09 10:52 KST.

**Objective:** Render characters as the preferred marker source while retaining fallback scene objects.

**Files:**
- Modify: `web/src/pages/OfficePage.tsx`

Implementation shape:

- Build `characters = buildOfficeCharacters(state, nodes)`.
- Build `characterSceneObjects = buildOfficeCharacterSceneObjects(characters)`.
- Use character scene objects when non-empty; otherwise fallback to existing `buildOfficeSceneObjects` output.
- Keep markers `aria-hidden` and `pointer-events-none` in this first slice.
- Keep room buttons as the actual clickable targets.

### Task 10-A4: Korean copy and legend

Status: done 2026-05-09 10:52 KST.

**Objective:** Make the RPG intent explicit without implying raw task visibility.

**Files:**
- Modify: `web/src/pages/OfficePage.tsx`

Add or revise copy near the map:

- `모델 캐릭터가 역할별로 배치됩니다.`
- `캐릭터 움직임은 안전 DTO의 상태/개수/흐름만 반영합니다.`
- Legend roles: `모델`, `작업자`, `검토자`, `자동화`, `전달`, `경보`.

### Task 10-A5: Verify

Status: done 2026-05-09 10:52 KST.

Run:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts && \
  ./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts && \
  npm run build

cd /Users/lidises/dev/hermes-agent
source .venv/bin/activate && \
  scripts/run_tests.sh \
    tests/hermes_cli/test_office_redaction.py \
    tests/hermes_cli/test_office_state_adapters.py \
    tests/hermes_cli/test_office_api.py \
    -q --tb=short && \
  git diff --check
```

Browser smoke:

- Open `http://127.0.0.1:8765/office?stage10a=characters`.
- Confirm Korean shell remains Korean.
- Confirm character labels/legend visible.
- Confirm marker count > 0.
- Confirm no marker exposes raw fixture text.
- Confirm console JS errors: none.
- Confirm reduced-motion disables animation.

---

## 4.1 Stage 10-A implementation notes

Implemented in `web/src/pages/officeView.ts` and `web/src/pages/OfficePage.tsx`.

Added helpers:

- `OfficeCharacterRole`, `OfficeCharacterStatus`, `OfficeCharacter`.
- `buildOfficeCharacters(state, nodes)`.
- `buildOfficeCharacterSceneObjects(characters)`.

Current behavior:

- Models/agents become generic Korean role characters such as `모델 캐릭터 1`.
- Work items become `작업자 N`; blocked work also creates `경보 담당 N`.
- Automations become `자동화 관리인 N`.
- Missing/unhealthy routing/source signals become `전달자 N` and `감시자 N`.
- Dense roles are capped at 3 visible characters plus safe `+N` aggregate.
- The map now prefers character scene objects when available, while falling back to the previous safe scene markers when no characters can be projected.
- Character markers remain decorative/non-interactive: `aria-hidden`, `pointer-events-none`; room buttons remain the real accessible targets.
- Korean map copy and role legend explain that movement reflects only safe DTO role/status/count/flow metadata.

Verification 2026-05-09 10:52 KST:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts && \
  ./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts && \
  npm run build

cd /Users/lidises/dev/hermes-agent
source .venv/bin/activate && \
  scripts/run_tests.sh \
    tests/hermes_cli/test_office_redaction.py \
    tests/hermes_cli/test_office_state_adapters.py \
    tests/hermes_cli/test_office_api.py \
    -q --tb=short && \
  git diff --check
```

Results:

- Frontend focused tests: `19 passed`.
- Frontend lint: passed.
- Frontend build: passed; existing large chunk warning remains.
- Backend focused office tests: `18 passed in 0.99s`.
- Browser smoke `/office?stage10a=characters`: Korean shell/body visible, 12 character markers, RPG copy/legend visible, markers `aria-hidden=true`, `pointer-events=none`, no marker raw leak detected, console errors none.

---

## 5. Stage 10-B implementation plan: original CSS/SVG character style

Status: implemented 2026-05-09 11:06 KST.

### Goal

Make roles recognizable as RPG characters without adding sprite assets.

### Tasks

1. Add pure view helper — done

```ts
export type OfficeCharacterView = {
  glyph: string;
  bodyClassName: string;
  accessoryClassName: string;
  nameplate: string;
  statusLabel: string;
  safeTitle: string;
};

export function buildOfficeCharacterView(character: OfficeCharacter): OfficeCharacterView
```

2. Add tests for each role — done

- model/operator/worker/reviewer/automation_keeper/router/sentinel/alert.
- Assert Korean label and no raw terms.

3. Add CSS-only character components — done

- no image imports.
- use layered div/SVG shapes:
  - head/body/accent/accessory/status light.
- keep visual system warm 2D RPG inside dark Hermes panel.

4. Update `SceneObjectMarker` or create `OfficeCharacterMarker` — done

- marker remains decorative for Stage 10-B.
- room button still handles selection.
- smoke hook: `data-office-character-role`.

5. Browser visual smoke — done

- check the full map after scrolling `main`.
- ensure lower legend and room cards do not overlap.

### Stage 10-B implementation notes

Implemented in `web/src/pages/officeView.ts`, `web/src/pages/OfficePage.tsx`, and `web/src/index.css`.

Added/changed:

- `OfficeCharacterView` and `buildOfficeCharacterView(character)` derive only role/status/room labels, CSS class names, glyphs, and safe titles from `OfficeCharacter`.
- `OfficeCharacterMarker` renders layered CSS shapes: head, body, accessory, status light, and tiny Korean nameplate/status text.
- Smoke hook `data-office-character-role` plus `data-office-character-status` is present on character markers.
- The map legend now says `캐릭터 역할 투영` and uses the same original role glyphs as the helper.
- Existing CSS-only motion layer remains in place and reduced-motion continues to disable animation.

Verification 2026-05-09 11:06 KST:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts && \
  ./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts && \
  npm run build
```

Results:

- RED verified first: Stage 10-B test failed because `buildOfficeCharacterView` did not exist.
- GREEN verified: focused frontend tests `20 passed`, lint passed, build passed; existing Vite large chunk warning remains.
- Browser smoke `/office?stage10b=character-style-2`: Korean shell/body visible, 12 character markers with `data-office-character-role`, first marker title safe, `aria-hidden=true`, `pointer-events=none`, no raw marker leak, role legend visible, console errors none.
- Visual smoke after scrolling `main`: CSS character markers/nameplates visible in the map; no severe overlap with room buttons, flow legend, or recent rail.

---

## 6. Stage 10-C implementation plan: role action loops

Status: implemented 2026-05-09 11:17 KST.

### Goal

Make characters appear to perform their roles while avoiding fake or private semantics.

### Safe action vocabulary

| Action id | Korean label | When allowed |
|---|---|---|
| `thinking` | 생각 중 | active model/operator with no stronger status |
| `working` | 작업 중 | work count/status indicates active/in-progress bucket |
| `reviewing` | 검토 중 | reviewer role or quality/review-safe status exists |
| `routing` | 전달 중 | topics/provenance route change or routing room activity |
| `scheduled` | 예약 대기 | automation next-run bucket later/today |
| `soon` | 곧 실행 | automation `<15m` or `<1h` bucket |
| `blocked` | 막힘 | blocked/error/attention count exists |
| `warning` | 확인 필요 | source partial/missing/warning |
| `idle` | 대기 | no current activity |
| `unknown` | 확인 불가 | source unavailable/unknown |

Add helper:

```ts
export type OfficeCharacterActivity = {
  id: string;
  label: string;
  motion: "walk" | "idle" | "blink" | "pulse" | "none";
  tone: "normal" | "success" | "warning" | "danger" | "muted";
  reducedMotionLabel: string;
};

export function buildOfficeCharacterActivity(
  character: OfficeCharacter,
  delta: OfficeStateDelta,
): OfficeCharacterActivity
```

Rules:

- Activity derives from safe character status, source health, automation timing bucket, and room/flow deltas only.
- No generated speech bubbles that imply real thoughts or content.
- Optional small bubble text is limited to action labels like `작업 중`, `검토 중`, `전달 중`, `확인 필요`.

### Stage 10-C implementation notes

Implemented in `web/src/pages/officeView.ts`, `web/src/pages/OfficePage.tsx`, and `web/src/index.css`.

Added/changed:

- `OfficeCharacterActivityId` and `OfficeCharacterActivity`.
- `buildOfficeCharacterActivity(character, delta)` maps only safe character role/status plus room/flow delta metadata into action ids/labels/motion/tone/reduced-motion copy.
- `OfficeCharacterMarker` now renders a small decorative action chip and smoke hook `data-office-character-activity`.
- CSS action-chip tones were added for normal/success/warning/danger/muted.
- Map copy and legend explain that action chips are also safe DTO/count/status/change decorations, not hidden work content or generated thoughts.

Verification 2026-05-09 11:17 KST:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts && \
  ./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts && \
  npm run build
```

Results:

- RED verified first: Stage 10-C test failed because `buildOfficeCharacterActivity` did not exist.
- GREEN verified: focused frontend tests `21 passed`, lint passed, build passed; existing Vite large chunk warning remains.
- Browser smoke `/office?stage10c=action-loops`: Korean shell/body visible, 12 character markers with `data-office-character-activity`, action ids include `thinking`, `unknown`, `warning`, `scheduled`, `blocked`, first marker title safe, no marker raw leak, action-chip legend visible, console errors none.
- Visual smoke after scrolling `main`: Korean action chips such as `생각 중`, `예약 대기`, and `확인 필요` are visible near character markers; no severe overlap with room buttons, flow legend, or recent rail.

---

## 7. Stage 10-D implementation plan: room-to-room RPG flow choreography

Status: implemented 2026-05-09 11:31 KST.

### Goal

Show work movement across rooms when safe deltas indicate flow changes.

Implementation candidates:

1. CSS/SVG path particles
   - tiny dots moving along existing safe flow SVG paths.
   - only active when `changedFlows` includes that edge.
   - reduced-motion: static `방금 변경` badge on edge.

2. Character route hint
   - one router/worker character receives a temporary path offset.
   - no individual task identity.
   - label: `흐름 변경`, `전달 중`.

3. Recent-change rail synchronization
   - clicking/hovering rail item highlights the corresponding room/flow only.
   - no mutation.

Helper candidate:

```ts
export type OfficeCharacterRoute = {
  id: string;
  fromRoomId: OfficeMapNode["id"];
  toRoomId: OfficeMapNode["id"];
  label: string;
  detail: string;
  tone: "normal" | "warning" | "danger";
  motion: "route" | "alert";
  changed: true;
  reducedMotionLabel: string;
};

export function buildOfficeCharacterRoutes(delta: OfficeStateDelta): OfficeCharacterRoute[]
```

### Stage 10-D implementation notes

Implemented in `web/src/pages/officeView.ts`, `web/src/pages/OfficePage.tsx`, `web/src/index.css`, and `web/src/pages/OfficePage.test.ts`.

Added/changed:

- `OfficeCharacterRoute` and `buildOfficeCharacterRoutes(delta)` project only `OfficeStateDelta.changedFlows` into safe room-to-room route hints.
- Route labels/details are generated from known room IDs (`세션`, `작업`, `자동화`, `라우팅`), not from raw flow labels or record contents.
- `/office` renders decorative `data-office-character-route` route hints with moving dots plus the static `흐름 변경` label.
- `prefers-reduced-motion: reduce` disables route hint/dot animations while preserving the visible label.
- Map copy and legend now explain that room-to-room route choreography is also a safe DTO/count/status/change decoration.

Verification 2026-05-09 11:31 KST:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts && \
  ./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts && \
  npm run build
```

Results:

- RED verified first: Stage 10-D test failed because `buildOfficeCharacterRoutes` did not exist.
- GREEN verified: focused frontend tests `22 passed`, lint passed, build passed; existing Vite large chunk warning remains.
- Backend focused office tests still pass: `18 passed in 1.00s`; `git diff --check` passed.
- Browser smoke `/office?stage10d=routes`: Korean shell/body visible, 12 character markers still present, route legend copy visible, no route raw leak, console errors none. Current live snapshot had no `changedFlows`, so no route hint DOM nodes were expected on first snapshot.

---

## 8. Stage 10-E implementation plan: safe character inspector

Status: implemented 2026-05-09 11:45 KST.

### Goal

Allow user to understand a character’s role without exposing raw records.

Inspector fields allowed:

- 캐릭터: safe label.
- 역할: role label.
- 방: room label.
- 상태: safe status/action label.
- 출처 상태: source health summary.
- 최근 변화: safe delta label if available.
- 안전 경계: `원문/프롬프트/로그/스크립트/인증 정보는 표시하지 않습니다.`

Implementation constraint:

- Character click target should be added only after keyboard/ARIA behavior is tested.
- Until then, room click + inspector summary is enough.

Implemented Stage 10-E:

```ts
export type OfficeCharacterInspector = {
  kind: "RPG 캐릭터";
  title: string;
  ariaLabel: string;
  fields: Array<[string, string]>;
};

export function buildOfficeCharacterInspector(character: OfficeCharacter, delta: OfficeStateDelta): OfficeCharacterInspector
```

Implementation notes:

- `buildOfficeCharacterInspector(character, delta)` uses `buildOfficeCharacterView(character)` and `buildOfficeCharacterActivity(character, delta)` so inspector labels stay aligned with the visible character nameplate/action chip.
- Recent change text is generated only from `OfficeStateDelta.nodeBadges[character.roomId]` and `OfficeStateDelta.changedFlows` involving the character room.
- Route text is regenerated from known room IDs, not copied from raw flow labels.
- `/office` character markers are real `<button>` elements with Korean ARIA labels such as `모델 캐릭터 살펴보기, 방 세션, 상태 활성, 액션 생각 중`.
- Visual child spans remain `aria-hidden` to avoid duplicated screen-reader noise.
- Existing room cards/buttons remain accessible inspection targets; character inspection is additive.

Verification 2026-05-09 11:45 KST:

- RED verified first: `OfficePage.test.ts` failed because `buildOfficeCharacterInspector` did not exist.
- Focused frontend test/lint/build passed: `OfficePage.test.ts` 23 passed; existing Vite large chunk warning remains.
- Backend focused office tests passed: `18 passed in 0.99s`; `git diff --check` passed before docs finalization.
- Browser smoke `/office?stage10e=character-inspector`: 12 `data-office-character-inspect` buttons, safe Korean ARIA labels, character click populates the safe inspector fields, no raw leak regex match, console errors none.

---

## 9. Stage 10-F implementation plan: usability hardening

Add tests and smoke checks for:

1. Dense state
   - Many models/work items/automations aggregate into safe `+N` characters.
   - No overlap that hides room count/status.

2. Missing sources
   - source unavailable creates sentinel/unknown character, not empty silence.

3. Reduced motion
   - animations stop.
   - action labels remain visible.

4. Korean-first copy
   - primary UI labels are Korean.
   - technical IDs remain only where useful for debugging.

5. Browser visual smoke
   - map readable at current desktop viewport.
   - small-screen layout documented.
   - console errors none.

---

## 10. Stage 11 renderer decision matrix

Only open Stage 11 after Stage 10-A through 10-F prove the RPG character model is useful.

| Option | Use when | Pros | Risks | Approval needed |
|---|---|---|---|---|
| Continue CSS/SVG | Characters remain under modest count and map is readable | smallest bundle, best accessibility, no dependency | limited animation/tile authoring | no new dependency approval |
| PixiJS | many animated sprites/particles needed, but not full game mechanics | strong 2D rendering, lighter than full game engine | canvas accessibility fallback needed, bundle cost | dependency/security/a11y review |
| Phaser | tile maps, game loop, camera, pathfinding-like behavior needed | closest to DeskRPG implementation style | heavier, game framework maintenance, accessibility fallback | stronger approval required |
| Custom canvas | tiny renderer is enough and Pixi/Phaser are too heavy | controlled bundle | custom maintenance, a11y fallback | dependency may be none, but architecture review needed |
| Stop renderer work | CSS/SVG already delivers comprehension | avoids complexity | less “game-like” ceiling | product decision only |

Stage 11 acceptance before dependency install:

- dependency bundle estimate documented,
- license/security review documented,
- reduced-motion and non-canvas fallback designed,
- no DeskRPG asset/code reuse unless explicitly cleared,
- rollback path to CSS/SVG retained,
- user explicitly approves dependency addition.

---

## 11. Later roadmap after Stage 11

These are not approved implementation tasks yet.

### Stage 12: Original Hermes sprite/tile design system

- Create original Hermes-safe character/tile visual language.
- Could use CSS/SVG first, then asset files only after approval.
- No external sprite packs without license review.

### Stage 13: Multi-room office expansion

- Add rooms only when DTO semantics justify them:
  - 모델 라운지,
  - 작업실,
  - 검토실,
  - 자동화실,
  - 라우팅/우편실,
  - 경보 게시판,
  - 미확인/격리 구역.

### Stage 14: Timeline playback, browser-local only

- Optional short-lived in-memory replay of last safe deltas.
- No persistent sensitive history.
- No raw record diff.

### Stage 15: Role authoring/settings, only if safe

- User may choose visual names/colors for safe model roles.
- Store only non-sensitive presentation preferences if approved.
- No model credentials or prompts stored in presentation config.

---

## 12. Acceptance criteria for Stage 10 overall

Stage 10 is complete only if:

- `/office` visibly presents models/agents as RPG-like role characters.
- Each character derives only from safe `OfficeState` and helper projections.
- Character roles/actions are understandable in Korean.
- No raw prompt/transcript/task/script/log/auth/secret data appears.
- Existing room badges, recent-change rail, source-health summary, and Safe inspector remain usable.
- Reduced-motion users still get text/static equivalents.
- Focused frontend tests pass.
- Focused backend office redaction/API/adapter tests pass.
- Browser smoke confirms Korean shell, visible characters, no console JS errors, no severe overlap.
- No renderer dependency is added unless Stage 11 is explicitly approved.

## 13. Immediate next implementation recommendation

Start with Stage 10-A only:

1. TDD `buildOfficeCharacters(state, nodes)`.
2. Map characters into existing CSS/SVG marker pipeline.
3. Add Korean role legend.
4. Verify with focused tests/build/backend/browser smoke.

This gives the user the core “각 모델이 캐릭터가 된다” direction without taking on renderer dependency risk yet.
