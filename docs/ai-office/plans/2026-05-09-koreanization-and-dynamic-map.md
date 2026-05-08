# Stage 9-E/9-F proposal — Korean-first dashboard and dynamic office map

Last updated: 2026-05-09 08:45 KST

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

## Suggested implementation order

1. Stage 9-E: Korean-first labels and helper text.
2. Stage 9-F1: pure helper tests for `buildOfficeStateDelta(previous, next)`.
3. Stage 9-F2: UI badges and recent-change rail from safe deltas.
4. Stage 9-F3: optional local live polling toggle with reduced-motion-safe highlights.

## Non-goals for now

- No mutation controls.
- No remote mode.
- No raw transcript/log/task/script display.
- No Phaser/Pixi/canvas renderer.
- No persistent browser storage for sensitive records.
