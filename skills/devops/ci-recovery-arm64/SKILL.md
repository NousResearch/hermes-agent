---
name: ci-recovery-arm64
description: "Recover from failed CI on ARM64 runners, diagnose bottlenecks, and fix main after premature merge. Oracle Cloud Free Tier specific."
version: 1.0.0
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [CI/CD, ARM64, Oracle-Cloud, GitHub-Actions, Recovery]
    related_skills: [github-self-hosted-runner, github-pr-workflow]
---

# CI Recovery for ARM64 Self-Hosted Runners

Recovery patterns for CI failures on ARM64 platforms (Oracle Cloud Free Tier), including diagnosing bottlenecks and fixing main after premature merge.

## Core Principle

**Rule #2:** CI green = merge. Never merge with red CI.

If this rule was violated and main is broken, follow this recovery pattern.

---

## ARM64 CI Bottleneck Analysis

### Expected Runtimes (Oracle Cloud Free Tier)

| Job | Runtime | Bottleneck |
|-----|---------|------------|
| `quality` (typecheck, lint, Biome) | ~6 min | TypeScript compilation |
| `web-build` (next build) | ~6 min | ARM64 compilation |
| **Total** | **12-15 min** | - |

**NOT normal:** 50+ minute runs (likely runner offline or backlog)

### Why ARM64 is Slower

- Cannot use x86_64 cached artifacts
- Must compile Next.js for aarch64 target
- Oracle Free Tier resource limits (4 OCPU, 24GB RAM)

### Monitoring Runner Status

```bash
# Check runner busy status
gh api /repos/<owner>/<repo>/actions/runners | jq '.runners[] | {name, busy, status}'

# Check for queued runs
gh run list --limit 5 --json status,conclusion,createdAt,displayTitle | jq '.[] | select(.status == "queued")'
```

### Diagnosis Commands

```bash
# Get full CI log for analysis
gh run view <RUN_ID> --log | tee /tmp/ci-log.txt

# Check for specific bottlenecks
grep -A 10 "next build" /tmp/ci-log.txt
grep -A 5 "typecheck" /tmp/ci-log.txt
grep -E "error|Error|ERROR" /tmp/ci-log.txt | head -20
```

---

## Recovery: Fixing Main After Bad Merge

**Scenario:** PR merged to `main` with failing CI, introducing broken code.

### Step 1: Diagnose the Damage

```bash
# Check current CI status on main
gh run list --branch main --limit 5

# Identify the problematic PR
gh pr list --state merged --limit 10

# Get logs of the failing run
gh run view <RUN_ID> --log-failed
```

### Step 2: Create Fix Branch

**DO NOT revert the merge** — this creates merge commits and confuses history.

```bash
git checkout main && git pull origin main
git checkout -b fix/main-ci-failures
```

### Step 3: Identify and Fix Issues

**Common ARM64/TypeScript Errors:**

#### 1. Sentry Import Error

```typescript
// WRONG (for Next.js client components)
import * as Sentry from '@sentry/react';

// CORRECT
import * as Sentry from '@sentry/browser';
```

**Root Cause:** Next.js client components must use `@sentry/browser`, not `@sentry/react`.

#### 2. Unknown Types in OAuth Providers

```typescript
// WRONG ( TypeScript error: user is unknown)
const user = await response.json();

// CORRECT with Zod validation
import { z } from 'zod';

const UserInfoSchema = z.object({
  id: z.number(),
  login: z.string(),
  email: z.string().email(),
});

const user = UserInfoSchema.parse(await response.json());
```

#### 3. Express Dependency Leak

```bash
# Symptom
Module '"express"' has no exported member 'Request'

# Fix
rm src/auth/oauth/callback-server.ts  # Remove unnecessary file
```

#### 4. Next.js Build Failures on ARM64

```bash
# Check for architecture-specific issues
gh run view <RUN_ID> --log | grep -i "aarch64\|arm64\|architecture"

# Common issues:
# - Wrong node version (use nvm with aarch64 node)
# - Native modules not compiled for ARM64
# - Docker images using wrong base image
```

### Step 4: Create PR from Fix Branch

```bash
git add -A
git commit -m "fix(ci): resolve TypeScript errors introduced in bad merge

- Fix Sentry import (@sentry/react -> @sentry/browser)
- Add Zod schemas for OAuth providers to validate API responses
- Remove callback-server.ts (unnecessary express dependency)

Fixes CI failures on main caused by premature merge of feat/sentry-integration.

Run: <RUN_ID>

Recovery workflow per skill ci-recovery-arm64."

git push origin fix/main-ci-failures

gh pr create \
  --title "fix: resolve CI failures on main" \
  --body "## Summary

Fixes TypeScript errors introduced by premature merge of #8.

## Changes
- Fix Sentry import: @sentry/react -> @sentry/browser
- Add Zod schemas for GitHub/LinkedIn OAuth providers
- Remove callback-server.ts (unnecessary express dependency)

## Test Plan
- [ ] CI passes on this branch
- [ ] Merge to main
- [ ] Verify CI passes on main after merge

## CI Status
- Failed run on main: <RUN_ID>
- Fix run: <NEW_RUN_ID>"
```

### Step 5: Wait for CI Green

```bash
# Monitor CI
gh pr checks --watch

# Or use a cronjob for long-running CI
```

### Step 6: Merge Fix to Main

```bash
# ONLY merge when CI is green
gh pr merge --squash --delete-branch
```

### Step 7: Verify Main is Clean

```bash
git checkout main && git pull origin main
gh run list --branch main --limit 3

# Disparar novo CI no main para validar
gh workflow run ci.yml --ref main
```

---

## Cronjob Monitoring Pattern

When CI is queued or taking longer than expected, create a monitoring cronjob:

```bash
cronjob action=create \
  name="ci-monitor-<run-id>" \
  schedule="every 5m" \
  repeat=3 \
  deliver="discord" \
  prompt="Check CI run <run-id> on hireme-agent:

1. If green:
   - Merge PR: https://github.com/OWNER/hireme-agent/pull/<pr-number>
   - Trigger new CI on main
   - Report: 'CI verde! PR merged. Verificando main...'

2. If red:
   - Exibir erros de TypeScript
   - Pedir direção ao usuário

3. If queued/in_progress:
   - Report elapsed time since trigger
   - 'CI ainda rodando: <min> minutos'

Use gh CLI to check status. Runner: oracle-arm64-hireme-agent."
```

---

## Prevention: CI Gate Enforcement

### Before Merge

```bash
#!/bin/bash
# pre-merge-check.sh

# Check CI status before merge
STATUS=$(gh pr checks --json conclusion --jq '.[].conclusion' | grep -v null)

if [ "$STATUS" != "success" ]; then
  echo "❌ ERROR: CI not green. Aborting merge."
  echo "Current status: $STATUS"
  exit 1
fi

echo "✅ CI is green. Proceeding with merge."
# Merge command here
gh pr merge --squash --delete-branch
```

### GitHub Branch Protection

Enable these settings in repo settings:

- **Require status checks to pass before merging**
  - `quality`
  - `web-build`
- **Require branches to be up to date before merging**
- **Do not allow bypassing the above settings**

---

## ARM64-Specific Gotchas

### 1. Docker Images

```dockerfile
# WRONG (x86_64 only)
FROM node:20-alpine

# CORRECT (multi-arch or explicit ARM64)
FROM --platform=linux/arm64 node:20-alpine
# OR
FROM node:20-alpine@sha256:<aarch64-specific-sha>
```

### 2. Native Modules

```bash
# Install with correct architecture
npm rebuild --build-from-source

# Check if module supports ARM64
npm view <package> cpu | grep arm64
```

### 3. Node Version

```bash
# Use nvm with aarch64 node
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20 --aarch64
```

### 4. Oracle Free Tier Limits

- **CPU:** 4 OCPU
- **RAM:** 24GB
- **Network:** 200 GB/mo outbound

**Workarounds:**
- Use `jobs.<job_id>.concurrency: group: ${{ github.workflow }}-${{ github.ref }}` to prevent queue buildup
- Schedule heavy builds during off-hours
- Use caching for dependencies

---

## Reference: Real-World Recovery Example

**Bad Merge:** `feat/sentry-integration` merged with failing CI (Run 26987797972)

**Errors Introduced:**
1. `src/lib/sentry.ts`: Used `@sentry/react` instead of `@sentry/browser`
2. `src/auth/oauth/providers/github.ts`: Unknown type for GitHub API response
3. `src/auth/oauth/providers/linkedin.ts`: Unknown type for LinkedIn API response
4. `src/auth/oauth/callback-server.ts`: Introduced express dependency

**Recovery:**
1. Created branch: `fix/typescript-errors`
2. Fixed Sentry import
3. Added Zod schemas for OAuth providers
4. Removed `callback-server.ts`
5. Created PR: https://github.com/OWNER/hireme-agent/pull/10
6. Triggered CI: Run 26988446793
7. Created cronjob for monitoring
8. Awaiting CI green before merge

**Lessons:**
- ARM64 Next.js builds take ~6min — normal
- Total CI ~12-15min — NOT 50min
- Rule #2 enforcement is critical
- Zod schemas prevent `unknown` type errors
- Sentry imports must be environment-specific

---

## Related Skills

- `github-self-hosted-runner`: Setup and troubleshoot ARM64 runners
- `github-pr-workflow`: General PR lifecycle
- `systematic-debugging`: Root cause analysis methodology