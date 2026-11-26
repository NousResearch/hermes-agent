---
name: frontend-verification-sweep
description: Run comprehensive verification sweeps on frontend projects after merges, deploys, or major changes. Bundle analysis, dependency audit, code quality, SEO, PWA checks. Structured PASS/FAIL reporting.
tags: [verification, frontend, bundle-analysis, code-quality, merge, audit, react, vite]
triggers:
  - "verification sweep"
  - "verify the merge"
  - "check everything works"
  - "audit the frontend"
  - "bundle analysis"
  - "run all checks"
---

# Frontend Verification Sweep

Run comprehensive verification sweeps on frontend projects after merges, deploys, or major feature work. Produces structured PASS/FAIL reports with actionable findings.

## When To Use

- After merge conflict resolution
- After major feature branch merges
- Pre-deploy verification
- When user asks to "check everything" or "audit the project"
- After dependency upgrades

## Execution Strategy

### PowerShell from WSL (Windows projects)
```bash
powershell.exe -Command "cd C:\Users\<user>\<project>; <commands>"
```
Batch related checks into single PowerShell calls to stay within tool limits. Each `terminal()` call should handle one logical phase.

### Native WSL projects
```bash
cd /path/to/project && <commands>
```

### Batching Rules
- Group file existence checks (`Test-Path`) into single calls
- Group import/regex searches (`Select-String`) into single calls
- Run `npm list`, `npx tsc`, `npm run build` as separate calls (they're slow)
- Use `execute_code` for 10+ sequential checks with logic between them

## Sweep Structure

Run in this order — each phase depends on the previous:

### PHASE 0: Git Status
```
git status              → clean working tree
git log --oneline -3    → confirm merge/commit present
```

### PHASE 1: File Inventory
Check all files that should exist post-merge/change. Use `Test-Path` for each.
Check files that should be deleted. Use `Test-Path` expecting False.

### PHASE 2: Feature Phases
For each feature added (shadcn/ui, TanStack Query, PWA, etc.):
1. Package installed? `npm list <pkg> --depth=0`
2. Config files present? `Test-Path`
3. Source files present? `Test-Path`
4. Imports wired correctly? `Select-String -Pattern <import>`
5. Mounted/used in app? Search App.tsx/main.tsx

### PHASE 3: TypeScript Compilation
```bash
npx tsc --noEmit 2>&1
# Must exit 0. Report any errors verbatim.
```

### PHASE 4: Production Build
```bash
npm run build 2>&1
# Record: build time, module count, any warnings
```

### PHASE 5: Bundle Analysis
From build output or filesystem:
- Total JS size (sum all `dist/assets/*.js`)
- Total CSS size (sum all `dist/assets/*.css`)
- Top 10 largest chunks
- Chunks > 500KB = WARNING
- Chunks > 1MB = CRITICAL
- Number of chunks (code splitting indicator)
- Check for `React.lazy` or dynamic imports in App.tsx

### PHASE 6: Dependency Audit
```bash
npm list --depth=0
npx depcheck 2>&1
```
Flag: unused deps, redundant packages (two libraries doing same thing), missing deps.
Note: depcheck false positives — tailwindcss via vite plugin, typescript via tsc, CLI tools.

### PHASE 7: Performance Checks
- **Images**: `Get-ChildItem public -Recurse -Include *.png,*.jpg,...` — flag >100KB
- **Fonts**: Check index.html + index.css for font loading strategy
- **PWA precache**: Check sw.js size, precache entry count from build output

### PHASE 8: Code Quality
- **Largest components**: `Get-ChildItem src -Recurse -Filter *.tsx | Sort-Object Length` — top 10
  - >500 lines = WARNING, >1000 lines = CRITICAL
- **TypeScript strict**: Check tsconfig.json for `"strict": true`
- **Suppressions**: Search `@ts-ignore`, `@ts-nocheck`
- **Hardcoded secrets**: Search for API keys, Supabase URLs (public bucket URLs are OK, anon keys in env vars are OK)

### PHASE 9: SEO & Accessibility
Check index.html for:
- `<title>` tag
- `<meta name="description">`
- `<meta name="viewport">`
- OG tags (og:title, og:description, og:image)
- Twitter cards
- `<link rel="manifest">`
- JSON-LD structured data
- Canonical URL

### PHASE 10: Final Git
```
git log --oneline -5
git status              → must be clean
```

## Report Format

```
BUNDLE HEALTH
  Total size: X MB
  Chunks: X
  Largest chunk: X KB (name)
  Code splitting: YES/NO
  Lazy loading: YES/NO
  Status: GOOD / WARNING / CRITICAL

DEPENDENCY HEALTH
  Total dependencies: X
  Unused packages: list
  Redundant packages: list
  Status: GOOD / WARNING / CRITICAL

PERFORMANCE
  Images: GOOD / WARNING / CRITICAL (list issues)
  Fonts: GOOD / WARNING / CRITICAL (list issues)
  PWA precache: X entries, X KB
  Status: GOOD / WARNING / CRITICAL

CODE QUALITY
  Largest components: list top 5 with line counts
  TypeScript strict: YES/NO
  any suppressions: X found
  ts-ignore usage: X found
  Hardcoded secrets: YES/NO
  Status: GOOD / WARNING / CRITICAL

SEO AND ACCESSIBILITY
  Title: YES/NO
  Meta description: YES/NO
  Viewport: YES/NO
  OG tags: YES/NO
  Manifest linked: YES/NO
  Status: GOOD / WARNING / CRITICAL

OVERALL VERDICT: GOOD / NEEDS WORK / CRITICAL ISSUES
```

Always list ALL warnings and criticals. Always suggest top 3 impactful fixes.

## PowerShell Gotchas (WSL → Windows)

When running PowerShell commands from WSL via `powershell.exe -Command`:

1. **`Select-String` has NO `-Recurse` flag.** Unlike bash `grep -r`, you must pipe from `Get-ChildItem`:
   ```powershell
   # WRONG: Select-String -Path src -Recurse -Pattern "foo"
   # RIGHT:
   Get-ChildItem src -Recurse -Include *.tsx,*.ts | ForEach-Object { Select-String -Path $_.FullName -Pattern "foo" }
   ```
   Alternatively, use the `search_files` tool which works natively from WSL.

2. **`| cat` breaks on git output in PowerShell.** `cat` is an alias for `Get-Content` in PowerShell and doesn't accept pipeline input the same way. Use `Out-String` or just let the output flow:
   ```powershell
   # WRONG: git commit -m "msg" 2>&1 | cat
   # RIGHT: git commit -m "msg" 2>&1
   ```

3. **PowerShell uses `$_` not `$1` for pipeline variables**, and `$LASTEXITCODE` for exit codes.

4. **`-Include` requires `-Recurse` or `-Path` with wildcard.** `Get-ChildItem src -Include *.tsx` returns nothing; use `Get-ChildItem src -Recurse -Include *.tsx`.

## PWA Removal Procedure

When removing PWA from a Vite project:

1. `npm uninstall vite-plugin-pwa`
2. Remove VitePWA import + `VitePWA({...})` from `vite.config.ts` plugins array
3. Delete: `src/components/InstallPrompt.tsx`, `public/offline.html`
4. In App.tsx: remove InstallPrompt lazy import, `<InstallPrompt />` JSX, offline event handler `useEffect`, unused `toast` import from sonner
5. Clean generated: `dist/sw.js`, `dist/workbox-*.js`, `dist/registerSW.js`, `dist/manifest.webmanifest`
6. Leave `public/manifest.json` alone if pre-existing (it's just browser metadata, not PWA)
7. Verify: `npx tsc --noEmit` exit 0, `npm run build` exit 0, `Test-Path dist/sw.js` returns False
8. Commit and push

## Key Pitfalls

1. **depcheck false positives**: tailwindcss (used via @tailwindcss/vite), typescript (used by tsc), CLI tools (shadcn, etc.) — verify before flagging as unused
2. **recharts + @nivo**: If both installed, flag as redundant — pick one
3. **PWA precache bloat**: >50 entries or >2MB is too much. Should precache only shell + offline page
4. **CSS lazy loading technique**: Moving a global CSS import (e.g. `katex/dist/katex.min.css`) from `main.tsx` into the component that needs it (e.g. `VerticallyAndCrosswise.tsx`) causes Vite to automatically split it into a separate CSS chunk. Users who never visit that component skip the download entirely. Example result: main CSS drops from 267KB to 238KB, KaTeX CSS becomes a 29KB lazy chunk.
5. **Large components**: >500 lines should be split. >1000 is critical. Report line counts, not just names
6. **Hardcoded Supabase storage URLs**: Public bucket URLs for media assets are OK (not secrets), but should be in a config constant, not inline
7. **Build warnings**: Some Vite builds emit warnings that aren't errors — capture and report them separately
8. **sw.js size**: Small sw.js (~5KB) means workbox runtime, precache list is inlined. Check build output for precache entry count, not sw.js size alone
9. **Check for existing dev server**: Before running `npm run dev`, check if Vite is already listening. In PowerShell: `Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in @(5173,5174,5175,3000) }`. If a server is already running, use it — don't spawn duplicates. Also, PowerShell buffers stdout from long-running processes, so `watch_patterns` may never fire; use port detection instead.
