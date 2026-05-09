# Stage 11 Renderer Decision Gate — AI Office RPG Map

> **For Hermes:** This is a decision-gate plan, not approval to add dependencies. Do not install Phaser, PixiJS, canvas renderers, sprite packs, DeskRPG code/assets, or mutation controls from this document alone. If implementation proceeds later, use TDD and keep the existing safe `OfficeState DTO -> projection helpers -> React/CSS/SVG renderer` boundary.

**Goal:** Decide whether the current CSS/SVG RPG office is sufficient, or whether a true renderer layer is justified for the next AI Office stage.

**Architecture:** Treat Stage 10-H as the baseline. The current dashboard already has safe deltas, character projection, CSS/SVG characters, action chips, route choreography, inspector, usability rail, density modes, and keyboard jump targets. Stage 11 should compare this baseline against renderer alternatives using explicit evidence, not visual ambition alone.

**Tech Stack:** Existing Hermes web dashboard, React/TypeScript, Vitest, CSS/SVG. Current web dependencies already include React, Tailwind, GSAP, Three/react-three stack for other web needs; Stage 11 does not add PixiJS, Phaser, canvas, sprites, or DeskRPG assets without explicit later approval.

**Last updated:** 2026-05-09 13:08 KST

---

## 1. Current baseline from Stage 10-H

The CSS/SVG RPG office already provides:

1. Safe dynamic state
   - `buildOfficeStateDelta(previous, next)`.
   - room `+N` / `-N` / `상태 변경` / `일정 변경` badges.
   - browser-memory-only `최근 변화` rail.
   - browser-local live tracking toggle using the existing read-only OfficeState endpoint.

2. RPG character projection
   - `OfficeCharacter` / `buildOfficeCharacters(state, nodes)`.
   - `buildOfficeCharacterSceneObjects(characters)`.
   - CSS/SVG-like character silhouettes and role/status nameplates.
   - safe action chips and route hints.

3. Readability and accessibility
   - safe character inspector.
   - usability checklist rail.
   - 요약/표준/상세 density modes.
   - keyboard jump targets: 지도, 사용성, 최근 변화, 안전 정보.
   - reduced-motion fallback.
   - Korean-first visible copy.

4. Safety boundary
   - no backend/API/schema changes.
   - no mutation controls.
   - no raw prompts, transcripts, task bodies, cron scripts, logs, auth, secrets, or individual task identity.
   - no persistent browser storage for dynamic history.
   - no Phaser/Pixi/canvas/sprite/DeskRPG dependency or asset copy.

## 2. Decision principle

Default decision: **keep CSS/SVG unless a renderer solves a measured problem that CSS/SVG cannot solve cleanly.**

A renderer is justified only if at least one of these is true:

1. Character count or path density becomes unreadable after density modes and grouping.
2. The user needs tile-map navigation, collision/pathfinding, or camera panning/zoom that is awkward in DOM/CSS.
3. The dashboard needs many concurrent animated objects where DOM layout/paint becomes visibly slow.
4. A renderer can improve, not degrade, accessibility and Korean text equivalents.
5. Bundle/maintenance/security/license cost is acceptable.

A renderer is not justified for:

- nicer movement alone,
- copying DeskRPG style/assets,
- fake speech bubbles,
- exposing hidden model thoughts,
- mutation controls,
- remote control-plane expansion,
- replacing text rails that already explain the state.

## 3. Renderer options

### Option A — Stay CSS/SVG/React

Status: recommended default.

Pros:
- No new dependency.
- Best accessibility path because buttons, links, ARIA labels, and text rails remain native DOM.
- Lowest bundle and maintenance cost.
- Fits current safe DTO projection architecture.
- Easy TDD of pure helpers and smoke selectors.

Cons:
- More complex animation/camera behavior is harder.
- Very high object counts may need more aggregation.
- Tile-map/game-loop features would be custom work.

Use this if:
- Stage 10-H is already understandable in browser smoke.
- The main pain is copy/layout/density, not renderer capability.

### Option B — Add PixiJS later

Status: investigate only if visual object count/particle density becomes a real bottleneck.

Pros:
- Strong 2D rendering performance.
- Better for many sprites/particles than DOM.
- More focused than a full game engine.

Cons:
- New dependency and bundle cost.
- Canvas/WebGL accessibility must be rebuilt with DOM overlays.
- Text and Korean labels likely still need DOM mirrors.
- Requires asset/licensing discipline.

Use this only if:
- DOM/CSS visibly struggles with many safe characters/routes.
- A DOM accessibility overlay remains first-class.

### Option C — Add Phaser later

Status: defer.

Pros:
- Game-loop, scenes, tile maps, camera, input, path systems.
- Natural fit if the office becomes a true interactive RPG simulation.

Cons:
- Heaviest mental/model shift for a read-only dashboard.
- Accessibility and DOM integration are harder.
- More likely to tempt game mechanics/mutation controls.
- Higher maintenance and bundle risk.

Use this only if:
- The product requirement becomes genuine tile-map/gameplay navigation, not dashboard observability.
- User explicitly approves dependency/security/accessibility review.

### Option D — Custom canvas renderer

Status: not recommended now.

Pros:
- Maximum control.
- Could be smaller than full engines if scope stays tiny.

Cons:
- Reinvents accessibility, input, layout, scaling, hit testing, text rendering, and testing.
- Easy to underbuild and hard to maintain.

Use this only if:
- Pixi/Phaser are too large and a tiny, well-bounded visual-only layer is proven simpler.

### Option E — Hybrid CSS/SVG + optional visual overlay

Status: possible later bridge.

Approach:
- Keep DOM as source of truth and accessibility layer.
- Add a non-interactive renderer overlay only for decorative particles/path effects.
- Text, inspection, quick jumps, room buttons, and safety rails remain DOM.

Pros:
- Preserves accessibility and safety.
- Limits renderer scope.

Cons:
- Two rendering layers must stay synchronized.
- Still adds dependency/bundle risk.

Use this if:
- Stage 10 remains useful but needs richer route/motion polish.

## 4. Stage 11-A evidence checklist

Before choosing any renderer, collect evidence from the existing CSS/SVG page:

1. Desktop smoke
   - Load `/office?stage11=desktop`.
   - Verify map, density controls, jump rail, usability rail, recent rail, and inspector are readable.
   - Verify no console JS errors.
   - Verify raw leak regex remains false.

2. 요약 mode smoke
   - Switch to 요약.
   - Verify visible character buttons are capped.
   - Verify `최근 변화 접힘` points to `#office-map-recent-collapsed`.

3. 상세 mode smoke
   - Switch to 상세.
   - Verify all generated safe characters remain understandable or identify crowding.

4. Reduced-motion smoke
   - Verify CSS animation is not the only information channel.
   - Text rails and static labels must remain meaningful.

5. Bundle posture
   - Run `npm run build`.
   - Record built JS/CSS size and existing warnings.
   - Do not add dependency until the baseline cost is recorded.

6. Accessibility posture
   - Verify native focus targets exist for map, usability, recent/collapsed recent, and safe inspector.
   - Verify room/character buttons have Korean ARIA labels.


## 4-A. Stage 11-A evidence pass — 2026-05-09 12:42 KST

Scope: documentation/evidence only. No renderer dependency, renderer import, canvas path, sprite asset, DeskRPG code/asset copy, backend/API/schema change, mutation control, persistent browser storage, cron/Kanban/topic/gateway/NAS/Obsidian write, or raw record projection was added.

Commands and environment:

```bash
cd /Users/lidises/dev/hermes-agent
git status --short --branch
git diff --check
lsof -nP -iTCP:8765 -sTCP:LISTEN

cd /Users/lidises/dev/hermes-agent/web
npm run build
```

Observed state:

- Branch: `ai-office-stage6-7-cleanup-20260508`.
- Baseline commit before Stage 11 docs: `236ae26b feat(office): add Korean RPG dashboard dynamics`.
- Dashboard listener already existed on `127.0.0.1:8765`; no new server was started.
- Browser URL: `http://127.0.0.1:8765/office?stage11=evidence`.
- `git diff --check` passed before evidence edits.

Desktop/standard smoke:

- `data-office-density-controls="true"` present.
- Density modes present for `summary`, `standard`, and `detail`; standard was active initially.
- `data-office-jump-targets="true"` present with 4 jump targets.
- Standard recent target points to `#office-map-recent`.
- `data-office-usability="true"` present.
- `data-office-character-inspect="true"` count: 12.
- Stable focus anchors present with `tabindex="-1"`: `#office-map-canvas`, `#office-map-usability`, `#office-map-recent`, `#office-safe-inspector`.
- First character ARIA label was Korean and safe: `모델 캐릭터 살펴보기, 방 세션, 상태 활성, 액션 생각 중`.
- Raw leak regex result: false for `/prompt|transcript|task_body|script|secret|api[_-]?key|password|token|BEGIN|sk-[A-Za-z0-9]/i`.
- Browser console JS errors: none.

요약 mode smoke:

- Activating 요약 changed `aria-pressed` to summary.
- Character inspect buttons capped at 6.
- Recent jump target changed to `#office-map-recent-collapsed`.
- `data-office-recent-collapsed="true"` appeared.
- `#office-map-recent-collapsed` existed with `tabindex="-1"`.
- Usability rail remained present.
- Raw leak regex remained false.
- Browser console JS errors: none.

상세 mode smoke:

- Activating 상세 changed `aria-pressed` to detail.
- Current fixture/source state showed 12 safe generated character inspect buttons, equal to the standard cap; no additional hidden characters appeared in this fixture.
- Recent jump target returned to `#office-map-recent`.
- Character ARIA labels remained Korean role/status/action labels only.
- Raw leak regex remained false.
- Browser console JS errors: none.

Reduced-motion/accessibility posture:

- CSS contains `prefers-reduced-motion` rules for the office motion layer.
- Normal-motion markers currently animate via CSS, e.g. `office-scene-walk`, but motion labels are safe DTO-based (`... 이동 표시 · 안전 DTO 기반`).
- Meaning is not animation-only: density controls, role/status/action chips, route/flow legend, usability rail, recent-change rail, and safe inspector remain textual DOM equivalents.
- Focus anchors exist for map, usability, recent/collapsed recent, and safe inspector depending on mode.

Visual readability note:

- Browser visual smoke marked the CSS/SVG RPG office as visually functional but borderline dense.
- Density controls and jump targets are readable.
- Main room cards remain understandable.
- Character nameplates/status chips are small and crowded around dense session/automation clusters.
- The lower rail competes visually with the bottom of the map.
- This is a CSS/SVG layout/readability polish signal, not enough evidence for a renderer dependency by itself.

Build-size baseline:

```text
../hermes_cli/web_dist/index.html                     0.47 kB │ gzip:   0.31 kB
../hermes_cli/web_dist/assets/index-f3FOCX5B.css    125.63 kB │ gzip:  19.99 kB
../hermes_cli/web_dist/assets/index-_qQ9_ZzW.js   1,253.73 kB │ gzip: 366.78 kB

(!) Some chunks are larger than 500 kB after minification.
✓ built in 1.96s
```

Stage 11-A conclusion:

- CSS/SVG passes the safety, read-only, accessibility-anchor, and browser-smoke gates.
- The measured issue is visual density/readability, especially small/crowded character labels and lower-rail competition.
- No measured performance or functional blocker currently requires PixiJS, Phaser, custom canvas, sprites, or a hybrid renderer overlay.
- Default decision remains: keep CSS/SVG and continue layout/density polish before any renderer spike.

## 4-B. Stage 11-B CSS/SVG layout-density polish — 2026-05-09 13:03 KST

Scope: implementation stayed inside the existing React/CSS/SVG map. No PixiJS, Phaser, canvas renderer, sprite asset, DeskRPG code/asset copy, backend/API/schema change, mutation control, persistent browser storage, cron/Kanban/topic/gateway/NAS/Obsidian write, or raw record projection was added.

Problem addressed from Stage 11-A evidence:

- Character nameplates/status chips were borderline dense around clustered room areas.
- The lower map legend/rail visually competed with the bottom of the map.
- This remained a CSS/SVG polish problem, not a renderer-adoption trigger.

Implementation:

- `OfficeMapPolishPlan` and `buildOfficeMapPolishPlan(densityPlan)` now derive a polish policy from the existing safe `OfficeMapDensityPlan`.
- Standard/detail dense conditions use compact role-centered nameplates; summary mode uses minimal label posture.
- The lower legend uses a detached, scroll-bounded `office-map-legend` block and the map gets extra bottom breathing room.
- Browser smoke hooks were added: `data-office-polish`, `data-office-polish-label-mode`, `data-office-polish-rail-mode`, and `data-office-polish-legend`.

TDD evidence:

```text
npm test -- --run OfficePage.test.ts
# RED: Stage 11-B test failed because buildOfficeMapPolishPlan was not a function
# GREEN: OfficePage.test.ts 27 passed
```

Verification 2026-05-09 13:03 KST:

- ESLint passed for `OfficePage.tsx`, `officeView.ts`, and `OfficePage.test.ts`.
- `npm run build` passed with the existing Vite large-chunk warning; current build size was JS `1,254.85 kB` / gzip `367.11 kB`, CSS `126.50 kB` / gzip `20.19 kB`.
- Backend focused office tests passed: `18 passed in 0.99s`.
- Browser smoke `/office?stage11b=polish`: standard/detail label mode `compact`, summary label mode `minimal`, rail mode `detached`, polish legend present, recent target adapts to collapsed summary rail, raw leak regex false, console JS errors none.
- Visual smoke after scrolling the `main` container found compact labels readable and no severe lower-legend overlap.

Safety posture:

- The helper consumes only the safe density plan and generated characters already produced by prior stages.
- It emits class names and Korean notes, not raw task/session/automation content.
- Raw-term exclusion is covered in the Stage 11-B test.

Current Stage 11-B conclusion:

- Keep CSS/SVG as the default renderer path.
- The first observed crowding issue was addressable through DOM/CSS polish.
- Re-open renderer research only if a later smoke test records a measured problem that compact/minimal labels, grouping, rail detachment, and density modes cannot solve cleanly.


## 4-C. Stage 11-C renderer decision checkpoint — 2026-05-09 13:08 KST

Scope: decision checkpoint only. No code, dependency, renderer import, canvas path, sprite asset, DeskRPG code/asset copy, backend/API/schema change, mutation control, persistent browser storage, cron/Kanban/topic/gateway/NAS/Obsidian write, or raw record projection was added.

Fresh checkpoint evidence:

- Branch remained `ai-office-stage6-7-cleanup-20260508` with latest pushed commit `a7090c0e feat(office): polish CSS RPG map density` before this documentation pass.
- Working tree was clean at checkpoint start; `git diff --check` passed.
- Existing dashboard listener on `127.0.0.1:8765` was reused; no new server was started.
- Browser URL: `http://127.0.0.1:8765/office?stage11c=decision`.
- Standard mode evidence: `data-office-polish="true"`, label mode `compact`, rail mode `detached`, 4 jump targets, 12 safe character inspect buttons, recent target `#office-map-recent`, raw leak regex false.
- Summary mode evidence after activating 요약: label mode `minimal`, 6 safe character inspect buttons, recent target `#office-map-recent-collapsed`, collapsed recent rail present.
- Browser console JS errors: none.

Decision:

- Close Stage 11 renderer adoption for now with **CSS/SVG retained as the primary renderer path**.
- Do not open a PixiJS, Phaser, custom canvas, hybrid overlay, sprite, or DeskRPG asset/code spike from the current evidence.
- Stage 11-A identified density/readability as the main issue; Stage 11-B solved that class of issue with compact/minimal labels and a detached lower legend; Stage 11-C found no new measured blocker.

Re-open criteria:

1. A later browser smoke records unreadable safe character density after summary/standard/detail modes, compact/minimal labels, grouping, and rail detachment.
2. The product requirement changes to true tile-map navigation, camera/pan/zoom, pathfinding, collision, or high-object-count animation beyond dashboard observability.
3. A separate user-approved spike measures bundle impact and includes DOM accessibility equivalents, safe DTO-only renderer inputs, reduced-motion fallback, license review, and CSS/SVG fallback or explicit degradation approval.

Recommended next phase:

- Move out of renderer-gate work unless new evidence appears.
- Prefer product/dashboard polish that preserves the current CSS/SVG, Korean-first, read-only, safe DTO projection boundary.
- Good next candidates: responsive/mobile readability evidence, Office empty-source copy polish, or a small non-mutating UX handoff/PR summary pass.

## 5. Stage 11 decision rubric

Score each option 1-5:

| Criterion | CSS/SVG | PixiJS | Phaser | Custom canvas | Hybrid overlay |
|---|---:|---:|---:|---:|---:|
| Privacy/safety boundary | 5 | 4 | 3 | 3 | 4 |
| Accessibility | 5 | 2 | 2 | 1 | 4 |
| Korean text/UI clarity | 5 | 3 | 3 | 2 | 5 |
| Bundle/maintenance cost | 5 | 3 | 2 | 2 | 3 |
| Many-object animation capacity | 3 | 5 | 5 | 4 | 4 |
| Dashboard fit | 5 | 3 | 2 | 2 | 4 |
| Game-like future capacity | 3 | 4 | 5 | 3 | 4 |

Initial reading after Stage 10-H: CSS/SVG remains the best default. Hybrid overlay is the only plausible next renderer path if decorative route/motion quality becomes a concrete gap.

## 6. Hard gates before dependency adoption

Do not install or import a renderer unless all are true:

1. User explicitly approves the selected renderer path.
2. A short spike branch measures bundle impact.
3. Accessibility plan includes DOM text/focus equivalents.
4. Safety plan confirms renderer receives only safe projection DTOs.
5. License review confirms no DeskRPG code/assets/sprites are copied.
6. Tests verify raw-term exclusion and reduced-motion fallback.
7. The fallback CSS/SVG view remains available or degradation is acceptable.

## 7. Recommended next action

Stage 11 should close as a **decision and evidence pass**, not as dependency adoption:

1. Keep CSS/SVG as the primary implementation.
2. Stop renderer work for now; do not create PixiJS/Phaser/canvas/hybrid spike branches without a new measured blocker and explicit user approval.
3. Re-open renderer research only if future evidence shows CSS/SVG cannot solve a measured readability, performance, or navigation problem cleanly.

Current recommendation after Stage 11-C evidence: **do not add a renderer. Keep CSS/SVG as the primary implementation; Stage 11-A/B/C did not produce a measured performance, readability, or navigation blocker that justifies PixiJS, Phaser, custom canvas, sprites, or a hybrid overlay.**
