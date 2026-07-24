---
name: github-self-hosted-runner
author: rafael.zendron22@gmail.com
description: "Setup GitHub Actions self-hosted runners on Linux ARM64."
triggers:
  - CI jobs failing with "runner offline" or "no available runners"
  - Setting up a new repo for deployment with self-hosted runners
  - Runner process dying after VM reboot
  - Need to re-register an existing runner instance
---

## Steps

1. **Diagnose Status**:
   - Check runner status via GitHub CLI:
     ```bash
     gh api repos/{owner}/{repo}/actions/runners | jq '.runners[] | {name, status, busy}'
     ```
   - Identify runners with status `offline` or remove duplicate entries.

2. **Cleanup** (if re-installing):
   - Remove zombie runner from GitHub:
     ```bash
     gh api repos/{owner}/{repo}/actions/runners/{id} --method DELETE
     ```
   - Clean up local directory:
     ```bash
     sudo rm -rf /opt/actions-runner/{runner-name}
     ```

3. **Setup Environment**:
   - Install dependencies:
     ```bash
     sudo apt-get update -qq
     sudo apt-get install -yqq curl jq ca-certificates
     ```
   - Ensure `gh` CLI is authenticated (`gh auth status`).

4. **Install & Configure**:
   - Create directory with correct ownership (CRITICAL for non-root users):
     ```bash
     sudo mkdir -p /opt/actions-runner/{runner-name}
     sudo chown -R $USER:$USER /opt/actions-runner/{runner-name}
     cd /opt/actions-runner/{runner-name}
     ```
   - Get registration token (uses authenticated `gh`, no PAT needed):
     ```bash
     REGISTRATION_TOKEN=$(gh api -X POST repos/{owner}/{repo}/actions/runners/registration-token --jq '.token')
     ```
   - Download latest runner (check arch: x64 vs ARM64):
     ```bash
     LATEST_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r '.tag_name')
     RUNNER_VERSION=${LATEST_VERSION#v}
     curl -sSL "https://github.com/actions/runner/releases/download/$LATEST_VERSION/actions-runner-linux-{arch}-$RUNNER_VERSION.tar.gz" -o actions-runner.tar.gz
     tar xzf ./actions-runner.tar.gz
     rm ./actions-runner.tar.gz
     ```
   - Configure:
     ```bash
     ./config.sh --url "https://github.com/{owner}/{repo}" \
         --token "$REGISTRATION_TOKEN" \
         --name "{runner-name}" \
         --labels "self-hosted,Linux,{arch}" \
         --work "/tmp/_work" \
         --unattended
     ```

5. **Persist (Systemd)**:
   - **Official Method**: Use the provided install script inside the runner dir:
     ```bash
     ./svc.sh install
     sudo systemctl enable actions.runner.{owner}.{repo}.{runner-name}.service
     sudo systemctl start actions.runner.{owner}.{repo}.{runner-name}.service
     ```
   - **Manual Method**: Create `/etc/systemd/system/github-runner-{name}.service` (see `scripts/setup-runner.sh` in skill files for template).

6. **Verify**:
   - Check service status:
     ```bash
     sudo systemctl status github-runner-{runner-name}
     ```
   - Check GitHub (should show `online`):
     ```bash
     gh api repos/{owner}/{repo}/actions/runners
     ```

## Pitfalls

- **Permission Denied (curl: 23)**:
  - Writing to `/opt` requires root. If running as user, create dir with `sudo` AND `chown -R $USER:$USER` immediately. Do NOT try to download as root then switch users—it breaks config context.

- **Runner Dies on Reboot**:
  - Root cause: Runner was started manually (`./run.sh`) or via a one-off command.
  - Fix: MUST install a systemd service. Use `./svc.sh install` (GitHub's script) or create a manual unit file and `systemctl enable` it.

- **GITHUB_TOKEN Hardcoding**:
  - Security risk and management burden. Do NOT hardcode PATs in scripts.
  - Fix: Use `gh api` CLI command which automatically picks up the authenticated session token.

- **Architecture Mismatch**:
  - Oracle Cloud Free Tier (Ampere) is ARM64.
  - Fix: Explicitly download `actions-runner-linux-arm64-*.tar.gz`. Do not assume x64.

- **Zombie Runners**:
  - If a VM is destroyed and runner isn't removed from GitHub settings, it stays "offline" forever.
  - Fix: Always `gh api ... --method DELETE` old runner IDs before re-installing.

- **Docker-based runner token expired**:
  - Runner token is ONE-TIME USE. If container restarts after initial registration, it fails with 404 from GitHub.
  - Symptom: Container exits immediately, logs show connection refused or 404.
  - Fix: Generate new token + recreate container:
    ```bash
    # 1. Get fresh token (requires gh auth)
    TOKEN=$(gh api repos/OWNER/REPO/actions/runners/registration-token --method POST --jq '.token')
    
    # 2. Remove old container
    sudo docker stop gh-runner && sudo docker rm gh-runner
    
    # 3. Recreate with new token
    sudo docker run -d --name gh-runner --restart unless-stopped \
      -e REPO_URL=https://github.com/OWNER/REPO \
      -e RUNNER_TOKEN="$TOKEN" \
      -e RUNNER_NAME=oracle-arm64-runner \
      -e RUNNER_LABELS=self-hosted,linux,arm64 \
      myoung34/github-runner:latest
    
    # 4. Verify "Listening for Jobs" in logs
    sleep 10 && sudo docker logs gh-runner --tail 15
    ```
  - CRITICAL: Do NOT pass the token as a truncated/masked value — it must be the full string. Use `execute_code` or a script to avoid shell history leaking it.

- **Script Syntax Errors**:
  - Editing scripts via patch/write tools can introduce truncation or syntax bugs.
  - Fix: Validate with `bash -n script.sh` before execution.

- **Attempting to Tunnel a Runner**:
  - GitHub Actions Runners connect OUTBOUND to GitHub. They do NOT expose HTTP for inbound connections.
  - **Symptom:** Cloudflare tunnel shows 502, `curl localhost:PORT` fails with "Connection refused"
  - **Reality Check:** Runners poll GitHub for jobs. No tunnel needed.
  - **Fix:** Remove any cloudflared tunnel pointing at the runner's listening port (if any). Verify runner status via `gh api .../actions/runners`, not HTTP access.
  - **Note:** If you need remote access to a runner VM, tunnel SSH or expose a dashboard (e.g., Grafana, Uptime Kuma) instead.

## Monitoring & Benchmarks

### Job Duration Benchmarks (ARM64 Oracle Free Tier)

Based on real measurements from `hireme-agent` CI:

| Job | Duration | Notes |
|-----|----------|-------|
| `quality` (lint, test, audit) | ~1.5-2min | Fast, parallel tests |
| `web-build` (Next.js build) | ~2-3min (varies 2-10min) | Bottleneck on ARM64; CPU throttling causes variance |
| **Total CI (2 jobs parallel)** | **~12-15min** | NOT 50min |

**IMPORTANT:** Observed job duration (12-15min) ≠ perceived slowness (50min). If you see 50min+, check QUEUE TIME, not execution time.

**If jobs take >20min (execution time, not queue):**
- Check CPU throttling (Oracle Free Tier shared resources) - **variance is normal**
- Check build cache misconfiguration (`cache: npm` missing)
- Check memory pressure (`free -h`, runner may be OOM-killed)

### Diagnose Slow Jobs

```bash
# Check runner resources
sudo systemctl status github-runner-RUNNER_NAME.service | grep -E "(Memory|CPU|Tasks)"

# See what's taking time in worker logs
tail -100 /opt/actions-runner/RUNNER_NAME/_diag/Worker_*.log | grep -iE "(build|compile|next|vitest)"

# Monitor job execution time
sudo journalctl -u github-runner-RUNNER_NAME --since "10 minutes ago" --no-pager | grep -E "(Running job|completed with result)"
```

### Common Session-Discovered Pitfalls

#### Old Runs Queue Up and Block New Runs

**Symptom:** New workflow runs stay `queued` even though runner is `online`

**Root Cause:** Runner is processing old runs (cancelled, stale, dependabot). New runs wait in queue.

**Fix:**
```bash
# Cancel all queued runs
for run in $(gh run list --repo OWNER/REPO --json databaseId,status | jq -r '.[] | select(.status == "queued") | .databaseId'); do
  gh run cancel $run --repo OWNER/REPO 2>/dev/null
done

# Re-trigger workflow
gh workflow run ci.yml --repo OWNER/REPO
```

#### Branch Protection Does NOT Block Jobs

**Myth:** "Jobs queued because branch protection is too strict"

**Reality:** Branch protection only blocks MERGE when CI fails. Jobs queue/run based on runner availability and label matching, NOT branch protection settings.

**Verification:**
```bash
# Check branch protection
gh api repos/OWNER/REPO/branches/main/protection | jq '.required_status_checks.contexts'

# This shows required checks, but does NOT block jobs from running
# Jobs only queue if runner is offline/busy or labels mismatch
```

#### Oracle Free Tier CPU Throttling

**Symptom:** Build times vary wildly (2min → 10min for same job)

**Root Cause:** Shared CPU resources on Oracle Free Tier Ampere A1 VMs. Other VMs on same host can consume CPU quota.

**Mitigation:**
- Accept variance as normal on free tier
- Consider paid tier for consistent performance
- Use build cache (`cache: npm`) to reduce CPU work

### Verification Checklist

After setup or troubleshooting, verify:

```bash
# 1. Runner Online on GitHub
gh api repos/OWNER/REPO/actions/runners | jq '.runners[] | select(.name == "RUNNER_NAME") | {status, busy}'
# Expected: {"status": "online", "busy": false}

# 2. Service Running Locally
sudo systemctl is-active github-runner-RUNNER_NAME.service  # Expected: active
sudo systemctl is-enabled github-runner-RUNNER_NAME.service  # Expected: enabled

# 3. Picks Up New Jobs (starts within 30s)
gh workflow run ci.yml --repo OWNER/REPO
timeout 60 bash -c 'until sudo journalctl -u github-runner-RUNNER_NAME --since "30 seconds ago" --no-pager | grep -q "Running job"; do sleep 5; done'

# 4. Job Completes Within Benchmark
sudo journalctl -u github-runner-RUNNER_NAME --since "5 minutes ago" --no-pager | grep -E "(Running job|completed with result)" | tail -4
# quality: <2min, web-build: <3min
```

## Common CI/CD Failures (Self-Hosted Context)

### Type Check Passes Locally, Fails on Runner

**Symptom:** TypeScript type errors appear in CI but not in local development

**Example from hireme-agent:**
```typescript
// This worked locally:
type SentryEvent = Parameters<NonNullable<Parameters<typeof Sentry.init>[0]["beforeSend"]>>[0];
type SentryHint = Parameters<NonNullable<Parameters<typeof Sentry.init>[0]["beforeSend"]>>[1];

Sentry.init({
  beforeSend(event: SentryEvent, hint?: SentryHint) { ... }
});

// CI error: "Property 'beforeSend' does not exist on type 'BrowserOptions | undefined'"
```

**Root Cause:** Complex type inference (`Parameters<>` + `NonNullable<>`) behaves differently across TypeScript versions/environments

**Fix:** Simplify type definitions - let TypeScript infer from actual usage:
```typescript
// Remove custom types, let inference do the work
Sentry.init({
  beforeSend(event, hint) {
    // Filter out sensitive information
    if (event.request?.headers) {
      event.request.headers = undefined;
    }
    return event;
  },
});
```

**Lesson:** When type inference is too complex, simplify to avoid environment-specific failures

### npm ci Lock File Mismatch

**Symptom:** 
```
npm error `npm ci` can only install packages when your package.json and package-lock.json or npm-shrinkwrap.json are in sync
npm error Missing: @sentry/browser@10.56.0 from lock file
```

**Root Cause:** CI uses `npm ci` (requires `package-lock.json`) but repository had `yarn.lock` or outdated lock file

**Fix:**
```bash
# Remove conflicting lock files
rm yarn.lock pnpm-lock.yaml

# Regenerate with npm
npm install

# Commit BOTH files
git add package.json package-lock.json
git commit -m "fix: sync lock file for npm ci"
```

**Lesson:** `npm ci` requires exact lock file sync - never use `npm install` in CI, never mix package managers

### Runner Queue Perception vs Reality

**Symptom:** User says "CI takes 50min" but logs show 12min execution time

**Root Cause:** Perception includes queue time (runner busy), but actual job time is within benchmark

**Verification:**
```bash
# Check execution time (not queue time)
gh run view RUN_ID --json conclusion,startedAt,completedAt | jq '{
  status,
  conclusion,
  startedAt,
  completedAt,
  duration: (.completedAt - .startedAt)
}'

# If duration is <15min but perceived >50min, issue is queue time, not performance
```

**Lesson:** Benchmark execution time separately from queue time; 50min perceived ≠ 50min execution

---

## References

- **`references/github-actions-queue-debugging.md`** - Queue debugging patterns, when jobs stay queued despite online runner, resolution steps, misconceptions about branch protection.
- **`references/benchmarks.md`** - Detailed CI job duration benchmarks (ARM64 vs x64, common stacks).
- **`references/typescript-sentry-browser-patterns.md`** - TypeScript type inference pattern: when complex types fail on CI but pass locally.
- **`references/npm-ci-lock-file-sync.md`** - npm ci lock file sync pattern: solving "out of sync" errors when adding dependencies.
- **`scripts/setup-runner.sh`** - Complete, validated installation script for Oracle ARM64.