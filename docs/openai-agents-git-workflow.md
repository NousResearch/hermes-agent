# OpenAI Agents SDK Hardening Git Workflow

Status: local-first, not pushed  
Tracker: `docs/openai-agents-project-tracking.json`  
Gate: `python scripts/check_openai_agents_quality.py`

## Purpose

This document wires the OpenAI Agents SDK hardening work into a native Git workflow before any external tracker is authorized. Git history plus the project-tracking manifest are the current source of truth.

## Current authority boundary

Allowed without further approval:

- local file edits under the SDK hardening scope;
- local commits with conventional messages and verification bodies;
- deterministic local gates;
- local status/log/diff inspection.

Requires explicit scope before execution:

- `git fetch origin --prune`;
- rebase or merge against upstream;
- push to any remote;
- create GitHub Issues;
- create PRs;
- create Kanban tasks or cron digests;
- start NAS/Helix migration.

## Native Git workflow

1. Keep roadmap state in `docs/openai-agents-project-tracking.json`.
2. Link every completed roadmap item to at least one local commit ref.
3. Link every roadmap item to a receipt group or deterministic gate proof.
4. Run the local gate before every commit:

```bash
python scripts/check_openai_agents_quality.py
```

5. Commit with conventional messages and a verification body.
6. Before any future push/PR, run the pre-push commands declared in `git_workflow.pre_push_required_commands`.
7. Bridge to GitHub Issues or Hermes Kanban only after explicit external-tracking scope is granted.

## Quality-gate enforcement

`check_openai_agents_quality.py` validates:

- tracking manifest exists and uses `project_id=OASDK-HARDENING`;
- roadmap, receipt-group, and next-action IDs are unique and stable;
- roadmap commit refs resolve locally;
- cross-links between roadmap items, receipt groups, and next actions resolve;
- next actions declare approval level plus destructive/privileged/external/recurring flags;
- `git_workflow` is `native_git_local_first`;
- current branch and origin remote match the manifest;
- push, GitHub issue creation, and PR creation require explicit scope;
- untracked files are restricted to the manifest's allowed list.

## Current local commits tracked

```text
da6b37213 governed SDK bridge baseline
3a78ac5ce SDK failure receipts
e3401f585 architecture workflow receipts
a36a79845 project tracking manifest
060bc1648 tracking manifest status update
```

## Known local state

The working copy is intentionally allowed to show these unrelated untracked paths:

```text
.install_method
manual-test/
```

They are not part of the SDK hardening scope and should remain untouched unless explicitly authorized.
