---
name: pixel-pal-web-subagent-recovery
description: PixelPal subagent failure modes and manual recovery workflows after max_iterations exhaustion
version: 1.0
tags: [pixel-pal-web, subagent, recovery]
created: 2025-05-07
---

# PixelPal Subagent Recovery Patterns

## Context
pixel-pal-web V21-V36+ development uses `claude` subagent delegation with max_iterations=30.
Recurring failure modes require manual intervention.

## Known Failure Modes

### 1. Subagent reaches max_iterations mid-implementation
**Symptom**: Subagent completes SOME features but not ALL. Files may be committed (agent commits incrementally), but requirements are incomplete.

**Recovery workflow**:
1. `git status --short` — see what files modified/staged
2. `git log --oneline -5` — see what agent committed
3. **Critical**: `git checkout -b vXX-feature` may fail with "already exists". If so, `git checkout vXX-feature` to use existing branch (agent may have created it). Confirm with `git branch | grep '*'`
4. Check which requirements from PRD are MISSING (grep for key strings)
5. **Always check PersonaDetail.tsx** — V25 (Settings), V35 (PersonaProfile), V36 (Memo send dialog) all added to this file. Agent may have accidentally deleted existing components when adding new ones. Check for: `PersonaProfile` JSX, `MemoPanel` imports, `Badge` import
6. **Always check PersonaSelector.tsx** — new features (V36 unread badge) often added here. Verify `Badge` imported and count display present
7. **Always check ChatPanel.tsx** — new features (V36 memo banner) often added here. Verify `MemoPanel` import and notification banner JSX present
8. Manually patch missing pieces using patch tool
9. `git add` → `git commit` → `GIT_TERMINAL_PROMPT=0 git push origin vXX-feature --quiet`

**V36 detailed case study**:
- Agent: committed 4 things (Memo interface, MemoPanel, PersonaDetail memo dialog, store actions)
- Missing/Broken: (a) PersonaProfile deleted from PersonaDetail (import + JSX gone), (b) PersonaSelector unread badge not implemented, (c) ChatPanel memo banner not implemented
- Fix PersonaDetail: re-add `import { PersonaProfile }` and `<PersonaProfile .../>` JSX before closing `</Dialog>`
- Fix PersonaSelector: add `Badge` to MUI imports, add `getUnreadMemosCount` to store selectors, add badge rendering in menu item Box
- Fix ChatPanel: add `import { MemoPanel }`, add `memoOpen` state, add memoNotification state + setMemoNotification, add memo banner JSX between Header and Messages, add `MemoPanel` JSX at bottom, add TWO useEffects — one to set notification on persona switch, one to clear notification on subsequent switches
- Committed as 2 logical commits (selector+chatpanel, detail+store) then pushed

**V30-V38 stat**: All 9 iterations (V30-V38) chose option A. All 9 required some manual intervention — ranging from minor patches to multi-file fixes.

### 2. Subagent goes off-track (wrong feature implementation)
**Symptom**: Agent implements something completely different from PRD scope.

**V21 example**: Agent built "multi-agent group chat collaboration" instead of "persona switching isolation".

**Recovery**: Requires full rebase + reimplementation. Not salvageable.

**Prevention**: More explicit PRD scope language.

### 3. Subagent completes but doesn't push
**Symptom**: All requirements met, code committed, but `git push` never happens.

**Recovery**:
1. `GIT_TERMINAL_PROMPT=0 git push origin vXX-feature --quiet`
2. If timeout, retry once
3. If still fails, push via GitHub REST API (see git/github-api-push-when-network-blocks-git skill)

### 4. Agent completed but never committed (code exists but git clean)
**Symptom**: `git status --short` shows modified files but `git log` shows no new commits. Agent hit max_iterations before calling `git commit`.

**Recovery**:
1. Verify branch exists: `git branch | grep '*'`
2. `git add <files>` — stage the modified files
3. `git commit -m "feat(VXX): description"`
4. Push

### 5. Two-effect pattern for ChatPanel notifications
When implementing cross-component notification features (e.g., memo notification on persona switch):

**Effect 1 — Set notification**:
```typescript
useEffect(() => {
  if (unreadMemos > 0) {
    setNotification(`📬 [发送者] 给你留了便条`);
  }
}, [activePersonaId]);
```

**Effect 2 — Clear notification** (prevents stale notification on re-render):
```typescript
useEffect(() => {
  return () => setNotification(null); // cleanup on unmount
}, []);
```

**Why two effects**: The notification should appear on persona switch, but shouldn't re-trigger on every render. Cleanup function alone isn't enough — need explicit dependency on activePersonaId.

### 6. PersonaDetail cross-component contamination
PersonaDetail.tsx accumulates components from V25 (Settings tab), V35 (PersonaProfile), V36 (Memo send dialog), V37 (Voice tab), V38 (Appearance tab). When agent adds V39 component here:
- Always check existing imports haven't been removed
- Always verify PersonaProfile/MemoPanel/etc. still present after agent's changes
- Common pattern: agent adds NEW import + JSX but accidentally drops existing ones — always grep before committing

**Verification**: `grep -c "PersonaProfile\|MemoPanel\|memoOpen\|profileOpen" src/components/Persona/PersonaDetail.tsx` — should be ≥ 4

### 6b. Sidebar/Layout integration gaps
Sidebar.tsx (and other layout components) may have full implementations in child component files that are never rendered:
- **PersonaSelector.tsx**: full 367-line component, never imported in Sidebar (V23 case)
- **TaskPanel.tsx**: exists but may not be in PANEL_COMPONENTS map
- Always check `NAV_ITEMS` array IDs match `PANEL_COMPONENTS` map keys, and each key's component is actually imported

**Verification**: `grep -c "PersonaSelector" src/components/Layout/Sidebar.tsx` (should be 2 = 1 import + 1 usage). `grep -n "id:" src/components/Layout/Sidebar.tsx` to list all nav item IDs, then confirm each ID has a corresponding entry in `PANEL_COMPONENTS` map in MainPage.tsx.

### 7. Integration Gap — new service files created but not wired into existing call sites
**Symptom**: Subagent creates new service/engine files (e.g., `emotionEngine.ts`, `emotionResponse.ts`) but doesn't integrate them into existing call sites (`ChatPanel.tsx`, `companionService.ts`, etc.). Files exist, tests pass, but the feature does nothing.

**Discovery**: 
- `git status` shows new service files created
- But `grep -r "emotionResponseEngine\|emotionEngine" src/components/ChatPanel.tsx` returns nothing
- Or the engine is imported but its methods are never called

**V21 example**: Subagent created `emotionResponse.ts` (373 lines) and `emotionEngine.ts` (422 lines) but never called `emotionResponseEngine.shouldRespond()` in `ChatPanel.tsx`.

**Recovery workflow**:
1. Identify the new service file and its main exported singleton (e.g., `emotionResponseEngine`)
2. Find the existing call site where the feature should be triggered (usually `ChatPanel.tsx` or `companionService.ts`)
3. Search for existing pattern: where are similar services imported/called in that file?
   ```bash
   grep -n "import.*from.*emotion\|import.*from.*service" src/components/ChatPanel.tsx
   ```
4. Identify the correct injection point (after emotion detection, before API call, etc.)
5. Patch the call site to import and invoke the new engine
6. For emotionResponse: also patch the `emotionContext` variable to use the detected emotion

**Common integration points**:
- `ChatPanel.tsx` handleSendMessage() — for text-based features (emotion detection, keyword triggers)
- `companionService.ts` buildCompanionSystemPrompt() — for personality/mood features
- `Sidebar.tsx` — for UI toggle components

**Prevention**: In delegation prompt, explicitly state: "You MUST also integrate the new service into existing call sites. Do not create files and leave them unused. Show the integration code in your delivery report."

### 8. Local branch already exists from previous attempt
**Symptom**: `git checkout -b vXX-feature` fails with "a branch named X already exists".

**Recovery**: `git checkout vXX-feature` (branch already has the intended commits).

## 9. Component exists but never rendered in parent (Sidebar/Layout)
**Symptom**: A component file has full implementation (PersonaSelector.tsx 367 lines) but is never visible in the app. Parent (Sidebar.tsx) doesn't import it.

**Root cause**: Subagent simplified/refactored Sidebar and accidentally dropped the import + rendering of a feature component. Unlike Mode 6 (PersonaDetail contamination where components get DELETED), here the component file was never wired into its parent at all.

**V23 example**: `PersonaSelector.tsx` had complete persona switching UI, but `Sidebar.tsx` never imported it. User reported "人格切换看不到" — the component existed and worked, just never rendered.

**Recovery workflow**:
1. Confirm component file has full implementation (read it — if it's skeletal or empty, different problem)
2. Check parent file (Sidebar.tsx) for the import: `grep -n "PersonaSelector" src/components/Layout/Sidebar.tsx`
3. If import missing: add `import { PersonaSelector } from '../Persona/PersonaSelector';`
4. Find where in parent it should render (check v38 branch: `git show v38-sidebar:src/components/Layout/Sidebar.tsx | grep -A5 -B5 PersonaSelector`)
5. Add JSX rendering at correct location
6. Verify no duplicate: `grep -c "PersonaSelector" src/components/Layout/Sidebar.tsx` should be exactly 2 (1 import + 1 usage)

**Prevention**: In delegation prompt, explicitly list all components that MUST be rendered in each panel/layout file. Don't assume agent knows to wire up cross-component UI elements.

## Delegation Prompt Template

Always include in context:
```
V21-V[N-1] context summary (key features + file locations)
Stay on scope — DO NOT touch [list of stable features]

IMPORTANT: PersonaDetail.tsx already contains: Settings tab (V25), PersonaProfile (V35), memo send dialog (V36), Voice tab (V37), Appearance tab (V38). DO NOT remove existing imports or components when adding new ones. Check with grep before committing.
```

## Verification Checklist After Subagent

1. `git log --oneline -3` — did agent commit?
2. `git status --short` — any uncommitted changes?
3. Grep for key feature strings from PRD — are they present?
4. **PersonaDetail verification**: `grep -c "PersonaProfile\|MemoPanel\|memoOpen\|profileOpen" src/components/Persona/PersonaDetail.tsx` (should be ≥ 4)
5. **PersonaSelector verification**: check if new badge/feature is present in the selector's persona list rendering
6. **ChatPanel verification**: check if new notification/component has BOTH the banner JSX and the useEffect hooks
7. **Sidebar verification**: `grep -c "PersonaSelector\|TaskPanel\|SettingsPanel" src/components/Layout/Sidebar.tsx` — every panel in NAV_ITEMS should be imported and rendered
8. `git push --quiet` — did it succeed?
