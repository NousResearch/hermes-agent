# Orchestration Reliability Lockdown OR1

Date: 2026-06-03

## Purpose

OR1 defines a repo-local reliability protocol for bounded Travis/Jenny/Codex
orchestration work. Its purpose is to reduce wrong-worktree execution, lane
drift, ambiguous approvals, unverified "implemented" claims, and lost
carry-forward state between continuation threads.

OR1A is documentation-only. It records the protocol, templates, failure rules,
and evidence expectations for later implementation work. OR1A does not
implement enforcement, command parsing, runtime checks, UI, backend behavior,
feature flags, persistence, tests, deployment, restarts, or worker automation.

## Mandatory Task Control Envelope

Every bounded task must begin with a Task Control Envelope before any command,
file read, file edit, test, commit, deploy, or remote probe. The envelope is
the authority for the current slice; previous chat context and adjacent threads
may inform risk, but they do not authorize work outside the envelope.

Template:

```markdown
Task Control Envelope

Task id:
Active lane:
Mode:
Current repo/path:
Required worktree:
Required branch:
Expected HEAD:
Allowed actions:
Forbidden actions:
Expected systems/files:
Files explicitly allowed to change:
Files explicitly forbidden:
Other threads/chat workstreams excluded:
Approval slice:
Stop condition:
Evidence required before "implemented":
```

Minimum required fields:

- Active lane
- Mode
- Current repo/path
- Required worktree
- Required branch
- Expected HEAD
- Allowed actions
- Forbidden actions
- Other threads/chat workstreams excluded
- Stop condition

If any minimum field is missing or conflicts with the request, the worker must
stop and ask for a corrected envelope before acting.

## Start Gate Template

The Start Gate confirms locality and clean state before work begins. It must be
performed in the required worktree named by the Task Control Envelope.

Template:

```markdown
Start Gate

pwd:
git rev-parse --show-toplevel:
git branch --show-current:
git rev-parse HEAD:
git status --short --branch:

Envelope match:
- Required worktree matched: yes/no
- Required branch matched: yes/no
- Expected HEAD matched: yes/no
- Worktree clean before start: yes/no
- Forbidden paths present in dirty state: yes/no/not applicable

Decision:
- proceed
- stop: wrong worktree
- stop: wrong branch
- stop: wrong HEAD
- stop: dirty worktree
- stop: envelope conflict
```

The worker may proceed only when the required worktree, branch, expected HEAD,
and allowed starting state match the envelope.

## Pre-Commit Stop-State Template

Before any commit approval or commit-only continuation, the worker must report
the exact stop state instead of continuing silently.

Template:

```markdown
Pre-Commit Stop-State

Repo/path:
Branch:
HEAD before work:
HEAD now:
Active lane:
Allowed actions used:
Forbidden actions avoided:
Files changed:
Diff summary:
Checks run:
Checks not run and why:
Secret scan result:
Adjacent lanes touched: yes/no
Other-thread/chat workstreams touched: yes/no
Known risks:
Ready for commit-only slice: yes/no
```

If the current lane forbids commits, this report is the stop condition. A later
commit-only approval slice must be explicit before any commit occurs.

## Final Report Template

Every bounded slice must end with a final report that can be audited without
reconstructing the whole chat.

Template:

```markdown
Final Report

Repo/path:
Branch:
HEAD before:
HEAD after:
Active lane:
Explicitly allowed actions:
Explicitly forbidden actions:
File changed:
Diff summary:
Checks run:
Checks failed:
Checks not run and why:
Docs-only confirmed: yes/no/not applicable
Code/UI/backend/runtime behavior changed: yes/no
Adjacent lane touched: yes/no
Other-thread/chat workstream touched: yes/no
Forbidden actions performed: yes/no
Reviewer verdicts:
Recommended next slice:
```

## Wrong-Worktree Failure Rule

If `pwd` or `git rev-parse --show-toplevel` does not match the required
worktree in the Task Control Envelope, the worker must stop immediately.

Required behavior:

- Do not edit files.
- Do not run tests.
- Do not inspect unrelated dirty state beyond what is needed to report the
  mismatch.
- Report the actual path, actual repo root, required path, active lane, and
  stop condition.
- Ask for a corrected envelope or a new explicitly approved worktree slice.

Wrong-worktree discovery is a hard stop, not a warning.

## Dirty-Worktree Failure Rule

If the Start Gate shows dirty files and the envelope does not explicitly allow
working from that dirty state, the worker must stop before editing.

Required behavior:

- List the dirty paths from `git status --short --branch`.
- Classify whether any dirty path is inside an explicitly forbidden area.
- Do not clean, revert, stage, commit, or overwrite the dirty files.
- Do not continue with the requested slice unless the user provides a new
  approval slice that addresses the dirty state.

If dirty files appear during the slice and were not created by the worker's
current work, the worker must stop and report the new dirty state before
continuing.

## Evidence Requirements For "Implemented"

"Implemented" must mean the requested behavior or artifact exists in the
allowed scope and has been verified with appropriate evidence for the lane.

Minimum evidence:

- Start Gate output matched the Task Control Envelope.
- Changed files match the allowed file list and avoid forbidden paths.
- `git diff --check` or an equivalent whitespace check passed when files were
  changed.
- A focused secret scan was run on changed files that could contain text.
- Relevant tests, docs checks, lint, build, browser QA, or runtime checks were
  run only if allowed by the active lane.
- Failures and skipped checks were reported explicitly.
- Final `git status --short --branch` confirms the expected changed files.
- The final report names any residual risk.

For documentation-only slices, "implemented" means the requested documentation
artifact was added or updated, the diff is limited to allowed docs files, and
no code, UI, backend, config, runtime, test, deployment, restart, or feature
flag behavior changed.

## Rollover / Carry-Forward State Card Template

Use a rollover card when work stops before completion, awaits review, or needs
to continue in a later thread.

Template:

```markdown
Rollover / Carry-Forward State Card

Task id:
Current stop-state:
Repo/path:
Branch:
HEAD:
Active lane:
Mode:
Last verified Start Gate:
Files changed:
Files intentionally untouched:
Checks completed:
Checks remaining:
Blocked reason:
User approval needed:
Forbidden next actions without approval:
Recommended next slice:
Notes for continuation worker:
```

The rollover card must separate completed work from recommended future work.
It must not smuggle approval for the next feature, cleanup, deployment, or
commit slice.

## Evidence Card Template

Use an Evidence Card to attach compact proof to a claim, decision, or handoff.

Template:

```markdown
Evidence Card

Claim:
Task id:
Repo/path:
Branch:
HEAD:
Command or source:
Observed output summary:
Files inspected:
Files changed:
Verification result:
Limitations:
Timestamp:
```

Evidence Cards should be short enough to read in a final report but specific
enough to identify the command, file, or source behind the claim.

## Relationship To Existing Controls

### Task Control Envelopes

OR1 makes the Task Control Envelope mandatory at task start. The envelope
defines the active lane, locality, allowed actions, forbidden actions, excluded
workstreams, and stop condition for the current slice.

### Approval Slices

OR1 treats approval as bounded and slice-specific. A previous approval does not
authorize adjacent work unless the active Task Control Envelope names that
approval slice and the requested action is inside its allowed actions.

### Lane Lock

OR1 operationalizes Lane Lock by requiring the active lane and forbidden
adjacent lanes to be declared before action. Documentation-only remains
documentation-only; read-only inventory remains read-only; stop-state-only
permits no cleanup or forward progress; cleanup/revert may touch only the
named target.

### Evidence Cards

OR1 uses Evidence Cards to keep proof close to claims. They are especially
important for "implemented", "verified", "clean", "docs-only", and "wrong
worktree" claims.

### Dirty Worktree Gate

OR1 formalizes dirty-worktree handling as a Start Gate decision. A dirty
worktree blocks the slice unless the envelope explicitly allows that exact
state. The worker reports dirty files instead of cleaning or overwriting them.

### File Locality Resolver

OR1 depends on file locality before action. The worker must confirm whether a
requested file or system is expected in the current repo, another local
worktree, a laptop path, OneDrive/rclone, a remote host, or an unknown source.
Locality context can inform risk, but it does not authorize remote probes,
copies, moves, extraction, credential access, or unrelated file inspection
unless an approval slice explicitly allows those actions.

## OR1A Non-Enforcement Statement

OR1A is a documentation-only hardening slice. It creates this protocol document
only. It does not add runtime guards, CLI commands, gateway hooks, dashboard
panels, MCP tools, test suites, CI checks, commit hooks, config keys, feature
flags, persistence changes, deployment changes, restart behavior, or social
platform/live-count behavior.

Any enforcement, automation, UI surfacing, or runtime integration for OR1 must
be proposed and approved in a later implementation slice.
