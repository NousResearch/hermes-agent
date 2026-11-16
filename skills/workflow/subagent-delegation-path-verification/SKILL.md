---
name: subagent-delegation-path-verification
description: 委托 subagent 前必须验证项目路径和技术栈，防止 subagent 在错误目录工作
---

# Subagent Delegation Path Verification

## Problem
When delegating to subagents for web UI work, subagents may search for projects in the wrong directory, resulting in wasted iterations. Subagents use file search which finds files by content/pattern, not verifying the actual intended project root.

## Symptoms
- Subagent completes task in a different directory than intended
- Subagent reports "Note: path X doesn't exist" or creates work in unexpected locations
- Files modified are not in the expected project tree

## Root Cause
Subagent searches for files like `package.json`, `src/`, `vite.config.*` across the entire filesystem and picks the first match, without verifying it's the right project context.

## Prevention Checklist (mandatory before delegation)
Before creating subagent tasks, always verify:
1. Confirm the exact absolute path of the project root
2. Confirm the technology stack (React vs vanilla JS vs Vue, etc.)
3. Confirm where the web entry point actually is (is it a single HTML file or a framework?)
4. Confirm the backend API path and its actual endpoints
5. Provide the subagent with the exact path, not a pattern to search for

## CRITICAL: Silent Path Failure Pattern
Subagents may silently fail to write to the specified path and write to `~/.hermes/proposals/workspace-dev/proposals/<project>/` instead, WITHOUT any error message. Tool trace may show success even though files landed in the wrong location.

**Detection after delegation:**
```bash
# Files NOT where expected = subagent wrote to workspace-dev
ls /home/hermes/<project>/src/stores/  # If empty, check workspace-dev
ls ~/.hermes/proposals/workspace-dev/proposals/<project>/src/stores/  # May have the files
```

**Recovery:**
```bash
# Copy from workspace-dev to actual code directory
cp -r ~/.hermes/proposals/workspace-dev/proposals/<project>/src/stores/* /home/hermes/<project>/src/stores/
cp -r ~/.hermes/proposals/workspace-dev/proposals/<project>/src/pages/<feature>/* /home/hermes/<project>/src/pages/<feature>/
cp ~/.hermes/proposals/workspace-dev/proposals/<project>/src/pages.json /home/hermes/<project>/src/pages.json

# Then commit from actual code directory
cd /home/hermes/<project>
git add <files>
git commit -m "fix: sync files from subagent delegation"
```

## If Subagent Goes Wrong Path
- Do NOT wait for max_iterations to finish - abort and re-delegate with corrected path info
- Provide explicit `cd /exact/path &&` in the task goal
- If the subagent creates files in the wrong location, note which files need to be manually moved/copied

## Project Structure: Deployed Assets vs Source Code Split

Some projects have a split structure where:
- `/home/hermes/<project>/` contains **only the deployed gh-pages** (built assets from `dist/` or `build/`)
- The **actual source code** lives in `~/.hermes/proposals/workspace-dev/proposals/<project>/`

**How to identify this pattern:**
```bash
# Check if it's a gh-pages clone (built assets only)
ls /home/hermes/<project>/  # If only assets/, index.html, static/ — it's deployed content
ls /home/hermes/<project>/.git  # Has .git but only gh-pages branch

# Check if source exists in proposals workspace
ls ~/.hermes/proposals/workspace-dev/proposals/<project>/src/  # Has full source
```

**Working with split structure:**
1. Source code: `~/.hermes/proposals/workspace-dev/proposals/<project>/`
2. If it has no `.git`, initialize first: `git init && git remote add origin <url>`
3. Create branch: `git checkout -b feature/xxx`
4. Push: `git push -u origin feature/xxx`
5. Deploy assets: `dist/` or `build/` goes to `/home/hermes/<project>/`

**Recovery when you accidentally work in wrong location:**
- If you modified `/home/hermes/<project>/` (gh-pages), that's the deployment — don't push directly
- Source changes should go to `~/.hermes/proposals/workspace-dev/proposals/<project>/`
- Build then deploy to gh-pages

## Variant: Subagent Creates a Parallel Directory with Different Name

Subagent may clone the repo to a NEW path instead of finding/using the existing project path. Result: two parallel directories with similar content.

**Symptoms:**
```bash
# You expected work in:
/home/hermes/ai-subscription/

# But subagent cloned to:
/home/hermes/ai-subscription-new/   # (has .git)
# While the actual project lived at:
/home/hermes/ai-subscription/       # (no .git, has node_modules)
```

**Detection:**
```bash
ls -d /home/hermes/*/ | grep -E "project-name|ai-subscription"  # Find parallel dirs
# If two dirs exist with similar names, compare:
ls /home/hermes/ai-subscription/.git 2>/dev/null || echo "no .git"
ls /home/hermes/ai-subscription-new/.git 2>/dev/null && echo "has .git"
```

**Recovery:**
1. Subagent's new files are in the non-git directory (e.g., `/home/hermes/ai-subscription/`)
2. Copy new files from non-git dir to git dir
3. Commit from the git dir and push

```bash
# Example recovery for ai-subscription
cp /home/hermes/ai-subscription/shared/lib/ai/tools.ts /home/hermes/ai-subscription-new/shared/lib/ai/
cp /home/hermes/ai-subscription/shared/lib/utils/*.ts /home/hermes/ai-subscription-new/shared/lib/utils/
cp /home/hermes/ai-subscription/web/src/api/stream-summary.ts /home/hermes/ai-subscription-new/web/src/api/
# ... copy all new files ...

# Then work from the git directory
cd /home/hermes/ai-subscription-new
git add <files>
git commit -m "feat: feature name"
git push origin master
```

**Prevention:** Always provide the **exact clone command and target path** in the subagent goal, not just "the project":
```
Clone to EXACTLY /home/hermes/ai-subscription — do NOT create a new directory with a different name
```

## pnpm install Timeout Workaround

When `pnpm install` times out due to network issues but you need node_modules:

```bash
# Copy node_modules from a working project (same dependencies)
cp -r /home/hermes/working-project/web/node_modules /home/hermes/new-project/web/

# Then verify
ls /home/hermes/new-project/web/node_modules/zod  # should exist
```

## When tsc -b && vite build Fails but vite build Works

Some projects fail at TypeScript compilation (`tsc -b`) due to monorepo shared module resolution issues, but Vite can still build successfully:

```bash
# Bad:
cd web && pnpm build  # runs "tsc -b && vite build"

# Good:
cd web && npx vite build  # skip tsc, Vite handles transpilation directly
```

This is common when `shared/` folder has no `package.json` or proper TypeScript config — tsc can't resolve it but Vite can.

## Post-Delegation Verification (MANDATORY)
After subagent completes, ALWAYS verify the files are in the correct location:
```bash
# Check file line count or existence at expected path
wc -l /expected/path/index.html

# If subagent wrote to workspace-dev instead, copy to correct location
cp -r ~/.hermes/proposals/workspace-dev/proposals/<project>/ /correct/project/path/
```

**Common wrong path pattern**: subagent writes to `~/.hermes/proposals/workspace-dev/proposals/<project>/` instead of the actual project path like `/mnt/c/Users/<username>/Desktop/<project>/`

**Recovery steps when this happens**:
1. Identify where subagent wrote files (check workspace-dev/proposals/)
2. Copy files to correct project location
3. Create new branch from master, replace files, commit and push
4. If merge conflict occurred, be careful: "ours" in git merge refers to current branch HEAD, which may be OLD code - always verify with `git show :2:filename` (stage 2 = ours/HEAD, stage 3 = theirs) before resolving
5. In merge conflict: stage :1 = common base, :2 = ours (current HEAD), :3 = theirs (being merged). Use `git show :2:filename` to verify which version contains your feature code

## Example of Bad vs Good Delegation

**Bad**: "Modify the collaboration web UI at collaboration/web/index.html"
- Subagent searches for "index.html" and finds the wrong one

**Good**:
- Project root: `/home/hermes/.hermes/proposals/workspace-dev/proposals/hermes-agent-collab/`
- Web UI: `collaboration/web/index.html` (pure vanilla JS, NOT React)
- API backend: `collaboration/collab_api.py` (FastAPI, prefix /api/collab)
- Instructions: "First run `cd /home/hermes/.hermes/proposals/workspace-dev/proposals/hermes-agent-collab && ls collaboration/web/` to confirm the file exists"

## Technology Stack Confirmation Commands
```bash
# Check if it's a framework project or plain HTML
ls /path/to/project/package.json  # React/Vite
ls /path/to/project/*.html         # Single file or plain HTML
ls /path/to/project/src/           # Framework with src/

# Check backend API structure  
grep -n "@router\." /path/to/collab_api.py | head -20

# Check if web assets exist as compiled output or source
file /path/to/project/collaboration/web/index.html
```
