# CI Escalation Guide

This document explains how automatic and manual CI escalation works in this repository.

## Automatic Escalation (Option A)

The `ci-escalation-detector` workflow automatically detects persistent failures:

### How It Works
1. **Continuous Monitoring**: Watches all CI workflows (Tests, Lint, Nix, etc.)
2. **Failure Detection**: Tracks consecutive failures per job type
3. **Escalation Trigger**: If 3+ failures detected → escalation activated
4. **Investigation Ready**: System is prepared for Claude Code agent to investigate

### What Gets Escalated
- **Persistent Test Failures**: Same test failing multiple times
- **Build Issues**: Nix, Docker, or compilation failures
- **Linting Blockers**: Linting rules that can't be auto-fixed
- **Security Scan Failures**: CodeQL or supply chain audit failures

### Automatic Response
When persistent failures detected:
- ✅ Failure patterns analyzed
- ✅ Escalation event logged
- ✅ Investigation triggered (Claude Code agent)
- ✅ Root cause analysis begins

---

## Manual Escalation (Option B)

You can manually trigger investigation at any time without waiting for automatic detection.

### How to Manually Escalate

1. **Go to Actions Tab**
   - Navigate to: `https://github.com/dizhaky/hermes-agent/actions`

2. **Select CI Escalation Detector**
   - Find and click "CI Escalation Detector" workflow

3. **Run Workflow**
   - Click the "Run workflow" button dropdown

4. **Choose Action**
   - Select: `manual-escalation`

5. **Provide PR Number (Optional)**
   - If investigating a specific PR, enter its number
   - Leave blank to investigate latest failures

6. **Execute**
   - Click "Run workflow"

### What Happens
1. Escalation event triggered
2. Claude Code agent invoked
3. Agent investigates:
   - Clones repository
   - Checks out failing branches
   - Runs tests locally
   - Analyzes error logs
   - Identifies root cause
   - Proposes fixes or reports findings

---

## Claude Code Agent Investigation Process

When escalated, the agent will:

### 1. **Environment Setup**
```bash
git clone <repo>
cd <repo>
git checkout <failing-branch>
```

### 2. **Run Failing Tests**
```bash
# Run tests that failed in CI
pytest tests/
# or
npm test
# or specific test suite
```

### 3. **Analyze Failures**
- Parse error messages
- Check stack traces
- Identify failure patterns
- Check dependencies/versions

### 4. **Attempt Fixes**
- Update dependencies if needed
- Fix code issues
- Adjust configuration
- Commit and push fixes

### 5. **Report Findings**
- Document root cause
- Explain solution applied
- Provide debugging logs if needed
- Recommend next steps

---

## When to Escalate Manually

**Escalate immediately if:**
- ❌ CI has failed 3+ times
- ❌ Same error repeating
- ❌ Auto-fix and auto-healer unable to resolve
- ❌ Build is completely blocked
- ❌ You're waiting on a critical fix

**Don't escalate if:**
- ✅ CI is still running (wait for completion first)
- ✅ Auto-fix is still attempting repair
- ✅ One-off transient failure (auto-healer will retry)

---

## Monitoring Dashboard

### Real-time Status
Check PR status in:
- **GitHub UI**: PR #50 checks section
- **Actions Tab**: Recent workflow runs
- **This Chat**: Agent updates

### Failure Patterns to Watch For
- Same test failing repeatedly → Flaky test
- Different tests failing → Environment issue
- Build step failures → Dependency problem
- Security scan failures → Code quality issue

---

## Escalation Workflow Diagram

```
CI Runs
  ↓
Success? → Done ✅
  ↓ No
Auto-Retry (up to 3x)
  ↓
Fixed by retry? → Done ✅
  ↓ No
Auto-Fix (linting, formatting)
  ↓
Fixed by auto-fix? → Done ✅
  ↓ No
Persistent Failure Detected
  ↓
[AUTOMATIC ESCALATION]
  OR
[MANUAL ESCALATION]
  ↓
Claude Code Agent Investigation
  ↓
Root Cause Identified
  ↓
Fix Applied / Findings Reported
  ↓
CI Re-run
  ↓
Success ✅
```

---

## Emergency Contact

If a CI failure is critical and needs immediate attention:

1. **Manually escalate** via workflow_dispatch
2. **Set escalation type**: `manual-escalation`
3. **Provide PR number** if applicable
4. **Claude Code agent** will investigate immediately

---

## Troubleshooting

### Q: How do I know if escalation was triggered?
A: Check the `CI Escalation Detector` workflow run in Actions tab. Look for green ✅ (triggered) or red ✗ (not triggered).

### Q: Can I escalate multiple PRs at once?
A: Currently escalation works on one PR at a time. For multiple, trigger separate manual escalations.

### Q: What if the agent finds a blocker?
A: Agent will report detailed findings. You can then decide on next steps (revert PR, refactor code, etc.).

### Q: How long does investigation take?
A: Typically 5-15 minutes depending on:
- Size of test suite
- Complexity of failure
- Number of dependencies to check
