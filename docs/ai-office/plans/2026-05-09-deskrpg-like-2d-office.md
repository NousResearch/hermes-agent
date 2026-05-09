# DeskRPG-like 2D Hermes AI Office Planning Note

> **For Hermes:** Use `writing-plans` for the execution plan, and use `subagent-driven-development` only after implementation is explicitly approved. This note is product/UX planning; it does not approve code changes, dependencies, service restarts, mutation controls, or asset reuse.

**Goal:** Move Hermes AI Office toward a lightweight DeskRPG-inspired 2D office map while preserving the current read-only, local-first, safe-DTO dashboard boundary. Stage 9-C implemented the first dependency-free CSS/SVG prototype of this plan, and Stage 9-D polished its contrast, marker hierarchy, and legend spacing.

**Architecture:** First create a dependency-free CSS/SVG “2D office prototype” on top of the existing `OfficeState -> officeView.ts -> OfficePage.tsx` path. Only after that prototype proves useful should the project revisit a true renderer layer such as Phaser/PixiJS via a separate dependency, security, license, and accessibility review.

**Tech Stack:** Current Hermes web dashboard, React/TypeScript, Vitest, CSS/Tailwind-style utility classes, SVG/CSS only for the next prototype; no Phaser/PixiJS/canvas dependency in the next implementation slice.

**Last updated:** 2026-05-09 12:01 KST

---

## 1. Material sufficiency check

### Verdict

The project has enough material to plan and implement a first Hermes-native, DeskRPG-like 2D office prototype without adding dependencies or copying assets.

It does not yet have enough cleared material to implement a production Phaser/PixiJS pixel renderer or reuse DeskRPG assets/code. That requires a separate license/security/dependency review.

### What is sufficient now

| Area | Current material | Sufficiency | Notes |
|---|---|---:|---|
| Product direction | User preference for lightweight 2D over 3D; existing AI Office product docs | High | Direction is clear: readable 2D office dashboard, not immersive 3D. |
| DeskRPG reference | Homepage inspected; GitHub metadata/README/package/README assets inspected; `deskrpg-reference.md` added | High for inspiration | Use visual/product grammar only, not code/assets. |
| Existing Hermes dashboard | Stage 8 read-only `/office`; Stage 9-A/B CSS/SVG map; Safe inspector | High | The current map is the natural prototype surface. |
| Safe data boundary | `OfficeState` DTO, redaction tests, metadata-only frontend helpers | High | Existing DTO can drive room/object counts and health. |
| Test harness | `OfficePage.test.ts`, Vitest, ESLint, build, focused backend office tests | High | Enough to TDD helper projections and privacy regressions. |
| Visual semantics | Sessions/Work/Automation/Routing rooms + safe flow hints | High | Already maps to DeskRPG-like lobby/workbench/machine/routing zones. |
| Accessibility/fallback | Existing dashboard list/panel UI and Safe inspector | Medium-high | Need explicit reduced-motion and keyboard/ARIA acceptance in next slice. |
| Original visual assets | None in `web/` | Low | Use CSS/SVG placeholders first; original sprite assets can come later. |
| External code/license reuse | DeskRPG license is not permissive MIT; GitHub reports NOASSERTION/Other and `LICENSE.md` uses Sustainable Use terms | Low | Do not copy DeskRPG code/assets into Hermes without legal review. |
| Renderer dependency readiness | Current branch intentionally has no Phaser/PixiJS/pixel assets | Low by design | Good for next prototype; insufficient for engine adoption. |

### Key source facts checked

- DeskRPG repository description: “2D pixel art multiplayer virtual office game.”
- DeskRPG package uses `phaser`, `socket.io`, Next/React, map editor, NPC/task/meeting structures.
- DeskRPG README/README.ko describe LPC character customization, shared office channels, AI NPC coworkers, task assignment, reports, AI meetings, and browser map editor.
- DeskRPG license is not a simple MIT license for project reuse; it uses Sustainable Use terms with commercial/internal-use limitations and third-party component caveats.
- Hermes current `web/` has no image assets to reuse for a sprite implementation.
- Hermes current office map already has safe room nodes, zones, flows, and fixture tests.

### Material gaps before production pixel renderer

1. Original Hermes sprite/tile asset set.
2. Dependency review for Phaser/PixiJS if engine adoption is desired.
3. License review for any external pixel assets or DeskRPG-derived visual materials.
4. Accessibility spec for keyboard navigation, screen-reader names, reduced motion, and non-pixel fallback.
5. Performance budget: target FPS, bundle-size cap, map object cap, acceptable animation count.
6. More data semantics if we want more than the existing four rooms: meeting/library/review/unknown buckets.

---

## 2. Product direction

The target should be:

> A lightweight 2D pixel-office observability layer for Hermes AI Office: DeskRPG-like in readability and warmth, but Hermes-native, read-only, privacy-safe, and dashboard-first.

Avoid:

- 3D camera/navigation.
- Full “metaverse” friction.
- Simulated agent society.
- Fabricated speech bubbles.
- Copying DeskRPG assets or code.
- Adding mutation controls to the visual layer.

Use:

- 2D top-down or RPG Maker-like office floor.
- Tile-like zones, not necessarily real tile engine yet.
- Small CSS/SVG avatar placeholders.
- Furniture metaphors for safe state.
- Serious Safe inspector for details.
- Existing list/dashboard fallback for actual reading and verification.

---

## 3. Proposed Hermes visual grammar

### Rooms / zones

| Hermes concept | 2D office metaphor | Current DTO source | First prototype display |
|---|---|---|---|
| Sessions / agents | Lobby / entry desks | `agents`, `sessions` source health | Pixel-avatar dots or small operator seats |
| Work items | Workbench / desk row | `work_items`, `kanban` source health | Desks/cards/monitors with counts and state lights |
| Automations | Machine room | `automations`, `cron` source health | Machines with status lights |
| Routing / provenance | Mailroom / dispatch board | `topics`, `provenance`, `topics` source health | Mail board / route arrows / unknown bucket |
| Attention rail | Alert board | `buildOfficeAttentionItems()` | Red/amber bulletin board or warning strip |
| Safe inspector | Operations console | Clicked object safe metadata | Keep existing inspector style; do not turn into fake dialogue |

### Object states

| State | Visual treatment |
|---|---|
| ok | soft green/teal light, normal border |
| missing | muted gray/outline, “unknown/unrouted” visible bucket |
| partial | amber light, dashed path, warning badge |
| error | red light, stronger outline, attention priority |
| active | subtle pulse or monitor glow, disabled by reduced motion |
| blocked | red blocker sign/card, never hidden by cute animation |

### Layout style

- Use a single office floor canvas-like section inside the current overview mode.
- Keep four major zones for the first implementation.
- Use floor material differences with CSS/SVG:
  - entry/lobby: darker tile
  - workbench: warm wood-like grid
  - machine: cool utility tile
  - routing: mailroom/dispatch board texture
- Use simple geometry, not asset-heavy art.
- Keep typography readable; pixel style should not make status labels tiny.

---

## 4. Implementation plan: Stage 9-C dependency-free 2D office prototype

### Task 1: Add scene-model helper types

**Objective:** Introduce a browser-local scene model that makes the office-map semantics more like a 2D office while still deriving only from safe DTO fields.

**Files:**
- Modify: `web/src/pages/officeView.ts`
- Modify tests: `web/src/pages/OfficePage.test.ts`

**Implementation shape:**

Add types similar to:

```ts
export type OfficeSceneObject = {
  id: string;
  roomId: OfficeMapNode["id"];
  kind: "avatar" | "desk" | "machine" | "mail" | "alert";
  label: string;
  health: OfficeMapNode["health"];
  x: number;
  y: number;
};
```

Add helper:

```ts
export function buildOfficeSceneObjects(state: OfficeState, nodes: OfficeMapNode[]): OfficeSceneObject[]
```

Rules:

- Produce bounded, deterministic placeholders.
- Use counts from `agents`, `work_items`, `automations`, `topics`, `provenance` only.
- Cap rendered object count per room, e.g. 6 visible placeholders plus a `+N` safe count object.
- Do not read `body`, `prompt`, `transcript`, `script`, `log`, `auth`, or secret-like fields.

**Test first:**

Add a Vitest case that inserts raw-looking fields into fixture rows and asserts scene object labels/details do not include them.

**Verification:**

Run:

```bash
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts
```

Expected after implementation: all tests pass.

### Task 2: Render tile-like room zones in current OfficeMap

**Objective:** Replace the current abstract node-only map with a more DeskRPG-like office floor, still CSS/SVG only.

**Files:**
- Modify: `web/src/pages/OfficePage.tsx`

**Implementation shape:**

- Keep the existing `OfficeMap` component.
- Add per-zone floor panels using CSS gradients/repeating-linear-gradient.
- Keep room buttons as accessible click targets.
- Add furniture-like minimal geometry:
  - desk rows for Work
  - machine block for Automation
  - mail board for Routing
  - lobby seats/avatar slots for Sessions
- Keep existing room labels/counts/health visible.

**Constraints:**

- No image assets.
- No canvas.
- No dependency.
- No absolute raw data display.
- No mutation controls.

**Verification:**

- Browser smoke `/office`.
- Check room click still updates Safe inspector.
- Check console has no JS errors.

### Task 3: Add lightweight CSS/SVG avatars and objects

**Objective:** Make the office feel alive with small placeholders without pretending to show real conversations or actions.

**Files:**
- Modify: `web/src/pages/OfficePage.tsx`
- Modify: `web/src/pages/officeView.ts`
- Modify tests: `web/src/pages/OfficePage.test.ts`

**Implementation shape:**

Render `OfficeSceneObject` items as small buttons or inert markers:

- `avatar`: circular/pixel-person marker in Sessions.
- `desk`: small table/monitor marker in Work.
- `machine`: small machine/status-light marker in Automation.
- `mail`: small board/mail marker in Routing.
- `alert`: warning marker for attention items, if included.

Click behavior:

- Either clicks inspect safe metadata only, or objects are non-click decorative and rooms remain the click target.
- If clickable, inspector rows should be only:
  - kind
  - room
  - health
  - safe count/index
  - redaction note

**Verification:**

Add tests for object cap and raw-field avoidance.

### Task 4: Add reduced-motion and responsive acceptance

**Objective:** Prevent game styling from harming daily dashboard usability.

**Files:**
- Modify: `web/src/pages/OfficePage.tsx`
- Possibly modify: `web/src/index.css` only if reusable CSS is necessary.

**Implementation shape:**

- Any pulse/glow animation must be subtle and disabled under `prefers-reduced-motion`.
- Small screens should stack/scale without hiding Safe inspector access.
- Maintain readable labels and `aria-label`s.

**Verification:**

- Browser visual check desktop width.
- Browser visual check narrower viewport if available.
- Console check.

### Task 5: Documentation and review

**Objective:** Update durable handoff docs and preserve the safety boundary.

**Files:**
- Modify: `docs/ai-office/STATUS.md`
- Modify: `docs/ai-office/NEXT.md`
- Optionally modify: `docs/ai-office/research/deskrpg-reference.md`

**Verification:**

Run:

```bash
cd /Users/lidises/dev/hermes-agent
source .venv/bin/activate
scripts/run_tests.sh tests/hermes_cli/test_office_redaction.py tests/hermes_cli/test_office_state_adapters.py tests/hermes_cli/test_office_api.py -q --tb=short
cd web
npm test -- --run OfficePage.test.ts
./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts
npm run build
cd ..
git diff --check
```

Then request independent code review focused on:

- privacy boundary,
- read-only boundary,
- accessibility,
- no new dependency,
- no raw field display,
- no accidental mutation affordance.

---

## 5. Later implementation plan: Stage 10+ character RPG office

The user clarified the real north star after Stage 9-I: the map should show each model/agent as its own RPG-like character performing its role, DeskRPG-inspired but Hermes-native and privacy-safe.

Detailed plan:

- `docs/ai-office/plans/2026-05-09-stage-10-character-rpg-office.md`

Updated stage sequence:

1. Stage 10-A: character role projection model — completed
   - Added `OfficeCharacter` and `buildOfficeCharacters(state, nodes)`.
   - Derive only safe role/status/count/source-health metadata from `OfficeState`.
   - Start with Korean generic labels such as `모델 캐릭터 1`, `작업자 1`, `자동화 관리인 1`, `전달자 1` when safe identity is unavailable.

2. Stage 10-B: original CSS/SVG character style — completed
   - Replaced generic markers with recognizable role characters.
   - No external sprite assets, DeskRPG assets, Phaser, PixiJS, or canvas.

3. Stage 10-C: role action loops — completed
   - Show safe activities such as `생각 중`, `작업 중`, `검토 중`, `전달 중`, `예약 대기`, `확인 필요`.
   - Do not invent speech bubbles or display private work contents.

4. Stage 10-D: room-to-room RPG flow choreography — completed
   - Added route hints only from safe `changedFlows` / room-level deltas.
   - Provided static text equivalents under reduced-motion.

5. Stage 10-E: safe character inspector — completed
   - Added character click/keyboard inspect affordances after ARIA/helper tests.
   - Inspector shows generated role, room, status, action, safe delta, and redaction note only.
   - No raw record fields, individual task identity, speech bubbles, or mutation target.

6. Stage 10-F: usability hardening
   - Dense-state caps, missing-source sentinels, small-screen smoke, Korean-first copy, and visual regression checklist.

7. Stage 11: renderer decision gate
   - Decide whether CSS/SVG remains enough or whether PixiJS, Phaser, or custom canvas is justified.
   - Any renderer dependency requires explicit user approval, bundle-size/security/accessibility/license review, and a CSS/SVG fallback path.

---

## 6. Acceptance criteria for Stage 9-C

Stage 9-C is complete only if:

- `/office` shows a visibly more DeskRPG-like 2D office floor while remaining Hermes-native.
- It uses only CSS/SVG/React; no Phaser/PixiJS/canvas/image assets.
- It uses only redacted `OfficeState` DTO fields and pure frontend helper projections.
- Safe inspector still exposes only metadata.
- Existing list/dashboard fallback remains intact.
- Tests prove raw-looking fields are not projected into scene labels/details.
- Tests cover object caps and room health states.
- Browser smoke confirms visible 2D room zones, object placeholders, click/inspect behavior, and no console JS errors.
- Focused backend Office tests still pass.
- Documentation records that this is a prototype, not a dependency adoption.

---

## 7. Open questions before implementation

1. Should object placeholders be clickable, or should only rooms remain clickable for now?
   - Recommendation: room click first; object click only after labels/ARIA stay clean.
2. Should avatars represent sessions only, or future specialized AI agent roles?
   - Recommendation: sessions only for Stage 9-C; roles later when safe labels exist.
3. Should the map include Meeting/Library rooms now?
   - Recommendation: not yet; keep four rooms until DTO semantics are stronger.
4. Should the visual tone be warmer DeskRPG wood-floor style or darker Hermes control-room style?
   - Recommendation: hybrid: warm floor inside dark Hermes panel.
5. Should this be committed as docs-only now?
   - Recommendation: yes; implementation should be a separate approved stage.


---

## 6. Stage 9-D follow-up status

Stage 9-D completed the first visual-polish pass after the dependency-free DeskRPG-like prototype:

- Room cards now use stronger backgrounds, borders, focus rings, and text contrast.
- Decorative scene markers now have a testable presentation helper and remain `aria-hidden`/non-interactive.
- SVG flow, marker, room-card, projection-label, and legend layers use explicit hierarchy to avoid accidental interaction or visual blocking.
- Lower rooms/markers were moved upward and the floor section height was increased so the bottom safety/flow legend no longer blocks lower room labels/cards/markers.
- No data boundary, dependency, renderer, backend route, or mutation-control expansion was introduced.

Next recommended work remains hardening rather than engine adoption: empty/error/loading/source-health fixtures, small-screen/responsive smoke, and only then a separate renderer review if still desired.
