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

**V29 case study** (memory system v2, V27-V29 cycle):
- Subagent (`claude` acp_command): created branch `v29-memory-v2`, committed 3 files (memoryTypes.ts, memoryStorage.ts, MemoryPanel.tsx UI), but `claude push` got BLOCKED by terminal prompt
- Recovery: used `execute_code` subprocess with `GIT_TERMINAL_PROMPT=0 git push origin v29-memory-v2 --quiet` to bypass prompt
- Network EOF: retry with backoff (4 attempts, 2s delay), exit_code=0 = success
- Pattern: dev agent uses `claude commit` (works) but `claude push` (blocked by git credential prompt). The subprocess push bypass is the key fix.
- V27-V29 all used squash merge (`gh pr merge --squash --delete-branch`), all had network retries

**V36 detailed case study**:
- Agent: committed 4 things (Memo interface, MemoPanel, PersonaDetail memo dialog, store actions)
- Missing/Broken: (a) PersonaProfile deleted from PersonaDetail (import + JSX gone), (b) PersonaSelector unread badge not implemented, (c) ChatPanel memo banner not implemented
- Fix PersonaDetail: re-add `import { PersonaProfile }` and `<PersonaProfile .../>` JSX before closing `</Dialog>`
- Fix PersonaSelector: add `Badge` to MUI imports, add `getUnreadMemosCount` to store selectors, add badge rendering in menu item Box
- Fix ChatPanel: add `import { MemoPanel }`, add `memoOpen` state, add memoNotification state + setMemoNotification, add memo banner JSX between Header and Messages, add `MemoPanel` JSX at bottom, add TWO useEffects — one to set notification on persona switch, one to clear notification on subsequent switches
- Committed as 2 logical commits (selector+chatpanel, detail+store) then pushed

**V29 case study** (floating toggle, V27-V29 cycle):
- Subagent reached max_iterations before completing integration
- Removed `sidebarCollapsed` from store correctly
- Committed Sidebar.tsx changes (remove collapsed prop)
- Committed MainPage.tsx changes (add floating button) BUT: still had `useStore((s) => s.sidebarCollapsed)` even though property deleted
- Also: `DocumentUpload` import accidentally removed
- Recovery: manually patched MainPage.tsx (fix state destructuring + restore import), amend commit, push
- Pattern: multi-commit branch but integration incomplete — had to patch before merge

**V30-V38 stat**: All 9 iterations (V30-V38) chose option A. All 9 required some manual intervention — ranging from minor patches to multi-file fixes.

### 2. Subagent goes off-track (wrong feature implementation)
**Symptom**: Agent implements something completely different from PRD scope.

**V21 example**: Agent built "multi-agent group chat collaboration" instead of "persona switching isolation".

**Recovery**: Requires full rebase + reimplementation. Not salvageable.

**Prevention**: More explicit PRD scope language.

### 3. Subagent completes but doesn't push
**Symptom**: All requirements met, code committed, but `git push` never happens.

**Recovery**:
1. `GIT_TERMINAL_PROMPT=0 git push origin vXX-feature --quiet` (foreground, use retry loop)
2. If timeout/EOF, retry once
3. If still fails, push via GitHub REST API (see git/github-api-push-when-network-blocks-git skill)

**V29 pattern** (subagent uses `claude push` which gets BLOCKED by terminal prompt):
```python
import subprocess
for attempt in range(4):
    result = subprocess.run(
        ['git', 'push', 'origin', branch_name, '--quiet'],
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GITHUB_TOKEN': token},
        capture_output=True, text=True
    )
    if result.returncode == 0:
        break
    time.sleep(2)
```

**Why it happens**: The `claude` subagent calls `git push` which invokes git credential helper. The hermes terminal tool blocks the credential prompt, causing it to hang indefinitely. The subprocess `GIT_TERMINAL_PROMPT=0` bypasses this entirely.

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

### 10. Store state removed but dangling references remain
**Symptom**: Subagent removes `sidebarCollapsed` and `toggleSidebar` from store AND updates MainPage.tsx to use them, but MainPage.tsx still has `useStore((s) => s.sidebarCollapsed)` even though the store property no longer exists. TypeScript errors or silent failures.

**Root cause**: Subagent patches the store definition out correctly, but doesn't fully migrate ALL call sites to use local state instead. Commits appear to be made but are incomplete.

**Recovery workflow**:
1. After `git checkout vXX-feature`, immediately check `git diff --stat` — what files were modified?
2. Check each modified file for dangling references: if a store property was removed, grep for its name in the modified files
3. In V29 case: store had `sidebarCollapsed` removed, but MainPage.tsx still called `useStore((s) => s.sidebarCollapsed)`. Fix: replace with `useState(false)` and local setter
4. Also check for accidentally removed imports (V29: `DocumentUpload` import was dropped)

**Verification after store changes**:
```bash
# Check if removed store property still referenced
git diff HEAD~1 src/store/index.ts | grep "^-" | grep "sidebarCollapsed"
grep "sidebarCollapsed" src/pages/MainPage.tsx  # should use local state now

# Check for dropped imports
git show HEAD:src/pages/MainPage.tsx | grep "DocumentUpload"  # should exist
```

**Prevention**: When delegating store removal tasks, explicitly say: "Migrate ALL call sites to local useState. Do NOT leave any dangling useStore((s) => s.OLD_PROPERTY) references."

### 11. Branch exists but commits incomplete — always inspect before creating new branch
**Symptom**: When recovering from incomplete subagent work, `git checkout -b vXX-feature` fails with "branch already exists". Subagent created the branch and committed some things, but integration is incomplete.

**V29 pattern**: Agent created `v29-floating-toggle`, committed 2 of 3 planned steps, but didn't complete the MainPage integration. On recovery, found dangling `useStore((s) => s.sidebarCollapsed)` reference.

**Critical**: Do NOT force-create a new branch. Always `git checkout vXX-feature` first and INSPECT what the agent actually committed (`git log --oneline -5`, `git show COMMIT --stat`). The branch may have partial commits you can build upon. Creating a new branch from HEAD loses those commits.

### 9. Component exists but never rendered in parent (Sidebar/Layout)
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
