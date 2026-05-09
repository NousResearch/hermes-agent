# Stage 9-E/9-F proposal — Korean-first dashboard and dynamic office map

Last updated: 2026-05-09 12:25 KST

## User problem

The `/office` dashboard is hard to read because most user-facing labels are in English. The 2D office map also feels static: it shows a snapshot, but it does not make it obvious what changed, what is moving, or what should be tracked over time.

## Stage 9-E — Korean-first readable dashboard

Goal: make the dashboard primarily Korean while keeping stable technical identifiers visible when they are useful for debugging.

Translate:
- Page titles, section titles, buttons, helper text, empty-state hints.
- Office map room names: 세션, 작업, 자동화, 라우팅.
- Zone names where they are visual labels: 입구, 작업대, 기계실, 우편실/라우팅.
- Status labels: 정상, 부분 연결, 미연결, 사용 불가, 오류.
- Inspector field labels: 상태, 확인 시각, 항목, 경고, 설명, 안전 개수.
- Safety copy explaining what remains omitted.

Keep as-is or only lightly label:
- DTO, OfficeState, cron, ID-like values, source IDs, platform/status enum values coming from adapters.
- Internal query params, code identifiers, and imported component names.

Safety boundary:
- Koreanization must not introduce raw prompts, transcripts, task bodies, cron scripts, logs, auth, or secrets.
- It is a presentation-only change over the existing redacted DTO.
- No new dependency, no backend schema change.

## Stage 9-F — Dynamic/tracking office map design

Use the current read-only polling endpoint first. Do not add write controls.

Recommended first dynamic layer:
1. Snapshot diff model in the browser
   - Keep previous `OfficeState` in memory.
   - Derive safe deltas from counts/status only:
     - room count changed: +N / -N
     - source health changed: 정상 → 오류, 오류 → 정상
     - attention count changed
     - automation next-run bucket changed
   - Never diff raw records or hidden fields.

2. Visual change badges
   - Room cards show small change chips for the last refresh: `+2`, `-1`, `상태 변경`.
   - Flow lines pulse only when endpoint health/count changed.
   - Attention rail gets a short “방금 변경” marker.

3. Timeline rail
   - Add a compact “최근 변화” rail generated from safe deltas.
   - Example entries:
     - `세션 +2 · 방금 전`
     - `자동화 상태 오류 → 정상`
     - `확인 필요 3 → 1`
   - Keep only a small in-memory ring buffer in the browser for MVP.

4. Tracking modes
   - Manual: existing 새로고침 button.
   - Live local mode: optional poll interval, e.g. 15–30 seconds, disabled by default or explicitly labeled.
   - Pause tracking button affects only browser polling, not backend jobs.

5. Motion/accessibility
   - Respect `prefers-reduced-motion`.
   - Use subtle pulses/highlights, not constant animation.
   - Provide text equivalents in the recent-change rail.

6. Later upgrade path
   - If polling is not enough, add local-only SSE/WebSocket after a separate review.
   - The event payload should be safe deltas, not raw state rows.

## Stage 9-F implementation slice — browser-local safe deltas

Implemented first layer:

1. Pure helper
   - `buildOfficeStateDelta(previous, next)` lives in `web/src/pages/officeView.ts`.
   - It accepts `previous: OfficeState | null | undefined` and `next: OfficeState`.
   - It returns:
     - `hasChanges`
     - `nodeBadges` keyed by `sessions`, `work`, `automation`, `routing`
     - `recentChanges` rail entries
   - First snapshot returns no changes so the dashboard does not fabricate history.

2. Safe comparison scope
   - Room counts from existing safe projection:
     - `agents.length`
     - `work_items.length`
     - `automations.length`
     - `topics.length + provenance.length`
   - Room health from existing safe source-health projection.
   - Attention count from the existing safe `buildOfficeAttentionItems()` helper.
   - The helper does not inspect prompt, transcript, body, script, log, auth, secret-like, or hidden raw fields.

3. Office-map badges
   - Room cards show last-refresh badges only:
     - `+N`
     - `-N`
     - `상태 변경`
   - Badge color/tone is derived from safe directionality:
     - positive count / health improvement
     - negative count
     - warning health degradation
   - Accessible labels include recent badge text, so the change is not animation-only.

4. Recent-change rail
   - The map card includes a `최근 변화` rail below the 2D map.
   - Entries are generated from safe deltas only, e.g. `세션 +2`, `자동화 상태 오류 → 정상`, `확인 필요 1 → 2`.
   - The component keeps only a small browser-memory ring buffer (`CHANGE_LIMIT = 6`).
   - No localStorage/sessionStorage persistence is used.

5. State management boundary
   - Current implementation compares the previous successful browser snapshot with the next successful snapshot.
   - Manual refresh remains available and is the default.
   - Stage 9-F3 adds an explicit opt-in local live tracking toggle; it only calls the existing read-only OfficeState endpoint every 30 seconds from the browser tab.
   - No SSE/WebSocket, backend schema change, mutation control, cron/gateway start-stop, or service restart was added.

## Stage 9-F3 implementation slice — local live tracking and flow hints

Implemented next layer:

1. Flow-level change hints
   - `buildOfficeStateDelta(previous, next)` now also returns `changedFlows` for the existing safe office-map flows.
   - A flow is marked changed when either its derived flow health changes or either endpoint room's safe count/health changes.
   - Flow entries use only endpoint room IDs, labels, safe counts, and safe health labels.
   - The SVG flow stroke gets a reduced-motion-safe `motion-safe:animate-pulse` highlight, while the bottom text legend adds `방금 변경` as the non-animation cue.

2. Recent-change duplicate collapse
   - Added `mergeOfficeRecentChanges(incoming, current, limit)` to keep the browser-memory rail compact.
   - Incoming duplicate IDs replace older rail entries at the front instead of spamming repeated identical deltas.
   - Ring-buffer storage remains React state only; no `localStorage` or `sessionStorage`.

3. Explicit local live tracking toggle
   - Added `실시간 추적 켜기` / `실시간 추적 일시정지` in the safe-mode card.
   - Default remains manual refresh only.
   - When enabled, the browser tab calls the same read-only `api.getOfficeState()` every 30 seconds.
   - Cleanup is local React interval cleanup; no backend, cron, Kanban, gateway, topic registry, NAS, or Obsidian mutation.

4. TDD/verification additions
   - Added helper tests for flow-level safe delta hints, first-snapshot empty `changedFlows`, and duplicate recent-change collapse.
   - Component-level polling behavior remains verified by build/lint/browser smoke because this repo currently has no React DOM test harness dependency.

## Stage 9-F4 implementation slice — timing buckets and live backoff

Implemented hardening layer:

1. Automation timing buckets
   - Added `buildOfficeAutomationTimingSummary(state, now)` in `web/src/pages/officeView.ts`.
   - It buckets only safe `next_run_at` timestamps into:
     - `overdue` / `기한 지남`
     - `<15m`
     - `<1h`
     - `today` / `오늘`
     - `later` / `나중`
     - `unknown` / `알 수 없음`
   - `buildOfficeStateDelta(previous, next, { now })` now adds an automation `일정 변경` badge and a safe recent-change rail entry when the primary next-run bucket changes.
   - It does not compare or expose cron prompts, scripts, outputs, task bodies, logs, auth fields, or secret-like fields.

2. Local live visibility/backoff helper
   - Added `OFFICE_LIVE_TRACKING_BASE_INTERVAL_MS = 30_000` and `resolveOfficeLiveTrackingInterval({ isVisible, consecutiveFailures })`.
   - Visible tab with no failures remains 30 seconds.
   - Hidden tabs slow to 60 seconds.
   - Consecutive failures slow to 60 seconds, then 120 seconds.
   - The live loop uses recursive browser-local timeouts so the next delay can adapt after visibility changes or failed reads.

3. UI copy and boundary
   - The live tracking safety copy now states that hidden tabs or repeated failures slow polling to 60–120 seconds.
   - Polling still calls only the existing read-only OfficeState endpoint.
   - No backend/schema/API, cron/gateway/service, Kanban/topic-registry, NAS/Obsidian, browser storage, renderer dependency, or mutation-control expansion was added.

4. TDD additions
   - Added helper tests for safe automation timing bucket counts, timing-bucket deltas in `buildOfficeStateDelta`, and live interval resolution.

## Stage 9-G implementation slice — fixture/source-health hardening

Implemented hardening layer after Stage 9-F4:

1. Source-health summary helper
   - Added `buildOfficeSourceHealthSummary(state)` in `web/src/pages/officeView.ts`.
   - It summarizes only source IDs, safe source statuses, warning counts, and expected-but-unreported safe source IDs.
   - It produces Korean labels/details for compact UI display, e.g. `주의 필요 · 정상 1 · 주의 2 · 공백/미연결 2 · 경고 3`.
   - It intentionally does not derive aggregate labels from raw adapter error text, stack traces, prompt bodies, scripts, logs, auth, or secret-like fields.

2. Central empty-state hints
   - Added `buildOfficeEmptyStateHints()` to centralize Korean copy for:
     - rooms
     - agents/sessions
     - work items
     - automations
     - topics
     - events
   - The hints clarify source gaps without implying external systems are empty and without exposing hidden raw content.

3. UI integration
   - `/office` source status now shows the compact source-health summary before detailed source cards.
   - Source counters now include `사용 불가` as a first-class visible count.
   - Empty states in the room/session/work/automation/topic/event sections use the helper-generated safe copy.

4. TDD/verification additions
   - Added helper tests for safe source-health summary, missing source IDs, empty-state hints, and empty-map resilience.
   - The empty map remains stable: four rooms, missing flows, and safe decorative scene objects even when DTO arrays are empty.

Safety boundary:
- Browser-facing DTO only.
- No backend/schema/API changes.
- No mutation controls.
- No cron/gateway/service/Kanban/topic-registry/NAS/Obsidian writes.
- No browser storage.
- No Pixi/Phaser/canvas/sprite/DeskRPG dependency or asset copy.

## Stage 9-I implementation slice — DeskRPG-like CSS motion layer

Implemented after the user pointed out that the map still did not move like DeskRPG:

1. Safe marker motion helper
   - Added `OfficeSceneMotionTrack`, `OfficeSceneMotionStyle`, and `buildOfficeSceneMotionTrack(object)` in `web/src/pages/officeView.ts`.
   - The helper derives only deterministic CSS motion hints from already-safe scene-object metadata:
     - avatar/mail markers: small walk/patrol drift
     - desk markers: subtle idle bob
     - machine/alert markers: status-light blink
   - It does not inspect prompt, transcript, body, script, log, auth, secret-like, or hidden raw fields.

2. CSS-only visual motion
   - Added `office-scene-walk`, `office-scene-idle`, and `office-scene-blink` keyframes in `web/src/index.css`.
   - Motion uses CSS custom properties for tiny per-marker offsets, duration, and delay.
   - `prefers-reduced-motion: reduce` disables all marker animation.
   - No Phaser, PixiJS, canvas engine, sprite assets, renderer dependency, or DeskRPG code/assets were added.

3. UI integration
   - `SceneObjectMarker` now applies the motion helper and exposes safe `data-office-motion-label` metadata for browser smoke checks while remaining decorative/non-interactive.
   - Office map copy now states that markers patrol/blink with CSS and stop under reduced-motion.
   - Existing room buttons remain the accessible interaction targets; marker pointer events remain `none`.

4. TDD/verification additions
   - Added a RED/GREEN helper test proving motion labels/classes/styles are safe and deterministic.
   - Browser smoke confirmed 11 scene markers, 11 animated markers in normal motion mode, `aria-hidden="true"`, `pointer-events: none`, safe motion label metadata, visible map markers, and no console errors.

Safety boundary:
- Browser-facing DTO only.
- No backend/schema/API changes.
- No mutation controls.
- No cron/gateway/service/Kanban/topic-registry/NAS/Obsidian writes.
- No browser storage.
- No Pixi/Phaser/canvas/sprite/DeskRPG dependency or asset copy.

## Further dynamic-map design

Next recommended design steps, still before any renderer dependency:

0. Stage 10-F usability layer provides a safe map-level checklist, Stage 10-G adds 요약/표준/상세 density/readability modes, and Stage 10-H adds keyboard jump targets across the map, rails, and safe inspector. Treat these as the boundary between CSS/SVG refinement and renderer research: if density, source fallback, reduced-motion, responsive posture, rail folding, and keyboard movement are understandable through text rails and CSS-only characters, prefer stopping or a documentation pass over a renderer.

1. Flow-level change hints
   - Extend the pure delta helper or add `buildOfficeFlowDelta(previous, next)`.
   - Compare only endpoint node count/health deltas.
   - Render subtle changed-flow strokes plus text equivalents in `최근 변화`.
   - Respect reduced-motion by using static badges when motion is reduced.

2. Automation timing buckets
   - Stage 9-F4 implemented the safe bucketizer for `next_run_at`:
     - `overdue`, `<15m`, `<1h`, `today`, `later`, `unknown`.
   - Delta entries compare only bucket labels, not cron prompt/script/output.
   - Rail example: `자동화 다음 실행 오늘 → <1h`.

3. Local live mode follow-up
   - Stage 9-F3 implemented the explicit browser-local toggle:
     - `실시간 추적 켜기`
     - 30-second interval
     - `실시간 추적 일시정지`
   - The toggle changes only frontend polling; it does not start/stop cron jobs or gateway processes.
   - Stage 9-F4 changed the browser-local live loop from a fixed interval to adaptive timeouts:
     - visible/no failures: 30 seconds
     - hidden tab: 60 seconds
     - repeated read failures: 60–120 seconds
   - Further work should only add richer visual regression or React DOM cleanup tests after dependency/test-harness review.

4. Snapshot identity and duplicate collapse
   - If repeated refreshes produce identical deltas, avoid duplicate rail spam by collapsing same `id` entries within the ring buffer.
   - Keep this in browser memory only.

5. Visual hierarchy
   - Keep room count/status badges primary.
   - Keep object markers decorative/non-interactive.
   - Keep room buttons as the accessible interaction target feeding Safe inspector.
   - Do not let dynamic badges hide zone labels, health labels, or the bottom safety/flow legend.

## Suggested implementation order

1. Stage 9-E: Korean-first labels and helper text.
2. Stage 9-F1/9-F2: completed — pure helper tests for `buildOfficeStateDelta(previous, next)` plus room badges and `최근 변화` rail.
3. Stage 9-F3: completed — opt-in local live polling toggle, safe flow-level change hints, and duplicate recent-change collapse.
4. Stage 9-F4: completed — safe automation timing buckets plus local live visibility/failure backoff.
5. Stage 9-I: completed — DeskRPG-like CSS motion layer for decorative scene markers: walk/idle/blink keyframes, reduced-motion stop, safe motion metadata, no renderer dependency.
6. Stage 10-A: completed — safe RPG character projection (`OfficeCharacter`, `buildOfficeCharacters(state, nodes)`, `buildOfficeCharacterSceneObjects(characters)`) turns model/agent/work/automation/routing/source-health/attention counts into Korean role characters and map legend while preserving decorative non-interactive markers.
7. Stage 10-B: completed — original CSS/SVG-like character silhouettes/nameplates (`OfficeCharacterView`, `buildOfficeCharacterView(character)`, `OfficeCharacterMarker`) make the role characters visible as small layered RPG figures with safe role/status labels and smoke hook `data-office-character-role`.
8. Stage 10-C: completed — safe role action chips (`OfficeCharacterActivity`, `buildOfficeCharacterActivity(character, delta)`, `data-office-character-activity`) show labels such as `생각 중`, `예약 대기`, `확인 필요`, and `막힘` from role/status plus room/flow deltas only.
9. Stage 10-D: completed — room-to-room RPG route choreography (`OfficeCharacterRoute`, `buildOfficeCharacterRoutes(delta)`, `data-office-character-route`) shows decorative `흐름 변경` route hints from `OfficeStateDelta.changedFlows` only.
10. Stage 10-E: completed — safe character inspector (`OfficeCharacterInspector`, `buildOfficeCharacterInspector(character, delta)`, `data-office-character-inspect`) lets keyboard/click users inspect generated role/status/action/recent-delta fields only.
11. Stage 10-F: completed — RPG office usability rail (`OfficeUsabilitySummary`, `buildOfficeUsabilitySummary(state, characters, options)`, `data-office-usability`) surfaces dense aggregation, missing-source fallback, reduced-motion, responsive posture, and Korean copy checks.
12. Stage 10-G: completed — density/readability modes (`OfficeMapDensityMode`, `OfficeMapDensityPlan`, `buildOfficeMapDensityPlan(mode, characters)`, `data-office-density-controls`) cap visible generated characters and fold rails safely.
13. Stage 10-H: completed — keyboard jump targets (`OfficeMapJumpTarget`, `buildOfficeMapJumpTargets(densityPlan)`, `data-office-jump-targets`) link 지도/사용성/최근 변화/안전 정보 to focusable safe sections and adapt the recent target for 요약 mode.
14. Stage 11 next: decide whether the CSS/SVG RPG office is sufficient before opening the separate renderer gate.

## Non-goals for now

- No mutation controls.
- No remote mode.
- No raw transcript/log/task/script display.
- No Phaser/Pixi/canvas renderer.
- No persistent browser storage for sensitive records.
