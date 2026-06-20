# Harness Consistency Audit Checklist

> Based on Rahul "Harness Engineering" + Karpathy CLAUDE.md practical audit framework.
> Don't blindly apply coding-scene standards — only audit what's active in YOUR actual usage pattern.

## Audit Steps

### 1. Config File Consistency
- [ ] Does config.yaml `personalities.<name>` match SOUL.md version?
- [ ] Is SOUL.md version current? (check `> Last updated` line)
- [ ] Do AGENTS.md Key Paths point to real, existing files?
- [ ] No two config sources defining the same constraint with different content (e.g., SOUL v3.0 vs v3.3)

### 2. Harness Artifacts (applicable items only)

| Artifact | Check |
|----------|-------|
| AGENTS.md / CLAUDE.md | Exists? Has project context + paths + safety gates? |
| Session init flow | Under Telegram, does SOUL/AGENTS actually load? (Issue #5200) |
| Plan/execute separation | Cron reports: generate → push separated? Complex tasks: proposal first? |
| Feedback loops | Push audit script running? approvals.mode active? checkpoints enabled? |

### 3. Karpathy 4 Rules Assessment

| Rule | Check |
|------|-------|
| Think before write | Does SOUL/AGENTS have "judgment over response" / "mark confidence when uncertain"? |
| Simple first | Is there explicit anti-overengineering constraint? |
| Surgical changes | Does AGENTS.md Safety Gates include "don't touch adjacent config / only operate on explicitly asked targets"? |
| Goal-driven | Does Workflow have "understand goal first" / "preserve completed work on failure"? |

### 4. Harness Decay Signals

- [ ] Memory usage (>70% watch, >85% immediate clean)
- [ ] Skills have curator auto-expiry (168h interval)
- [ ] Quarterly review of SOUL Hard Constraints
- [ ] config.yaml has no stale model references
