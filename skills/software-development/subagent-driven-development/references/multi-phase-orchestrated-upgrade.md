# Multi-Phase Orchestrated Upgrade

A workflow pattern for sequential multi-phase upgrades (frontend migrations, dependency upgrades, API version bumps) where each phase has hard verification gates before the next begins.

## When to Use

- User provides a numbered phase plan with explicit verification criteria per phase
- Phases are sequential (Phase 2 depends on Phase 1 being committed)
- Each phase has binary pass/fail checks (file exists, build clean, tsc clean)
- User wants checkpoint stops between phases (e.g., "ask me before Phase 6")

## The Pattern

```
For each phase:
  1. Run verification checks YOURSELF (don't delegate verification)
  2. If all pass → log "Phase N already done" → move to next phase
  3. If any fail → execute the phase → re-verify → commit
  4. If checkpoint requested → STOP and ask user before continuing
```

### Verification Checks (run in orchestrator, not subagent)

```bash
# File existence
test -f src/lib/query-client.ts && echo "EXISTS" || echo "MISSING"

# Package installed
npm list @nivo/radar --depth=0

# Import present
grep -c "ResponsiveRadar" src/components/ProgressDashboard.tsx

# Build + type check
npx tsc --noEmit && echo "TSC: PASS"
npm run build 2>&1 | tail -3
```

### Execution Strategy: Direct vs Delegated

| Task Type | Do Directly | Delegate to Subagent |
|-----------|-------------|---------------------|
| Install npm packages | ✓ | |
| Add shadcn components | ✓ (interactive CLI) | |
| Create simple config files | ✓ | |
| Rebuild large components (300+ lines) | | ✓ |
| Multi-file wiring + imports | | ✓ |
| Fix tsc errors | ✓ (targeted fixes) | |
| tsconfig changes | ✓ | |

**Rule of thumb:** If the task involves writing ONE file >200 lines or touching 3+ files with coordinated changes, delegate. Everything else, do directly.

### Commit Strategy

One commit per phase. Message format: `feat(scope): description` matching the phase's purpose.

```bash
git add -A && git commit -m "feat(ui): install shadcn/ui primitive layer"
```

## Pitfall: Verification Must Be Binary

Verification checks must produce a clear YES/NO. Avoid:
- "Does the dashboard look good?" (subjective)
- "Are the components wired correctly?" (requires code review)

Use instead:
- "Does button.tsx exist in src/components/ui/?" (file existence)
- "Does `npm run build` exit 0?" (build success)
- "Does ProgressDashboard import from @nivo/radar?" (grep check)

## Pitfall: Don't Redo Already-Completed Phases

Always verify BEFORE executing. If Phase 1 files already exist and build passes, log it and move on. Re-doing completed work wastes time and risks introducing regressions.

## Pitfall: CRLF on Windows Projects

When writing files from WSL to Windows-mounted paths (`/mnt/c/...`), use PowerShell or Python binary mode to preserve CRLF. See `ai-agent-delegation` skill for full CRLF preservation techniques.

## Pitfall: Subagent Timeout Recovery

When a delegated subagent times out (600s), check what it completed before redoing work:

```bash
# Check if packages were installed
npm list @vidstack/react --depth=0

# Check if files were created
test -f src/components/VideoPlayer.tsx && wc -l src/components/VideoPlayer.tsx

# Check if wiring was done
grep -c "VideoPlayer" src/components/LearnSection.tsx

# Check if it compiles
npx tsc --noEmit 2>&1 | head -5
```

A subagent that timed out after installing deps and creating 3 of 4 files saved you 80% of the work. Finish the remaining 20% directly.

## Merge Conflict Resolution After Upgrade

When merging main into the upgrade branch after all phases are committed, conflicts are inevitable because both branches touched the same files.

### Strategy Per File Type

| File | Resolution Strategy |
|------|-------------------|
| Files you didn't modify (PracticeSection, QuizSection, auth/) | `git checkout --theirs` — take main's version |
| Files you modified AND they modified (App.tsx, PvPBattleSection) | Read both sides, keep both sets of changes |
| package-lock.json | `git checkout --theirs`, then `npm install` to regenerate |
| package.json | Merge dependency objects — keep ALL packages from both sides |
| Config files (vite.config.ts, tsconfig.json) | Keep yours if it has features theirs lacks (VitePWA, etc.) |
| Files deleted in main (admin.ts) | `git rm` — respect the deletion |
| Test files from main that fail | Fix type errors (functions may have changed signatures) |

### Step-by-Step Workflow

```bash
# 1. Start the merge
git merge main

# 2. Identify all conflicts
git status --short | grep "^UU\|^DU\|^UD"

# 3. Resolve easy files first (checkout --theirs for files you didn't touch)
git checkout --theirs src/components/PracticeSection.tsx
git add src/components/PracticeSection.tsx

# 4. Resolve deleted files
git rm src/utils/admin.ts

# 5. Resolve complex merges (App.tsx, store, types) — read conflict markers
grep -n "<<<<<<\|======\|>>>>>>" src/App.tsx

# 6. For package.json — merge deps with Python JSON parsing
python3 -c "
import json
# Read both sides, merge dependencies, write back
"

# 7. Stage resolved files incrementally
git add src/App.tsx

# 8. After all conflicts resolved — regenerate package-lock
npm install

# 9. Fix any tsc errors from the merge (type mismatches, removed functions)
npx tsc --noEmit 2>&1 | head -20

# 10. Build to verify
npm run build 2>&1 | tail -5

# 11. Commit and push
git add -A
git commit -m "chore: merge main into feat/branch, resolve conflicts"
git push origin feat/branch
```

### Pitfall: sfx/audio Wiring Lost During Merge

When resolving conflicts by taking "their version" of a component, check if your branch added sfx/celebrate imports and wiring. If so, you need to:
1. Take theirs as the base
2. Re-add your imports at the top
3. Re-wire the function calls at the correct locations (answer handlers, result screens)

Use `git show HEAD:src/components/PvPBattleSection.tsx | grep "sfx\|celebrate"` to check what your version had before the merge.

### Pitfall: Post-Merge tsc Errors

Common tsc errors after merge:
- `TS2345: string | null not assignable to string` — their code passes nullable to non-nullable. Fix: `value ?? "default"`
- `TS2554: Expected N arguments but got M` — function signatures changed. Fix: update call sites
- `TS6133: unused variable` — they removed the usage but left the declaration. Fix: remove or prefix with `_`

## Example: 6-Phase Frontend Upgrade

```
Phase 1: shadcn/ui foundation
  Verify: button.tsx, card.tsx exist + build + tsc
  Execute: npx shadcn@latest init + add components

Phase 2: TanStack Query
  Verify: query-client.ts exists + hooks exist + build + tsc
  Execute: npm install + create hooks + wrap QueryClientProvider

Phase 3: Dashboard rebuild
  Verify: @nivo/radar installed + ProgressDashboard imports it + build + tsc
  Execute: delegate to subagent (300+ line component rebuild)

Phase 4: KaTeX math rendering
  Verify: katex installed + math.tsx exists + build + tsc
  Execute: npm install + create components + wire in App.tsx

Phase 5: Sonner + cmdk
  Verify: sonner installed + cmdk installed + CommandPalette exists + build + tsc
  Execute: delegate to subagent (App.tsx changes + new component)

Phase 6: STOP — ask user before continuing
```
