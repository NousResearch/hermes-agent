---
name: post-approve-merger
description: "Load when spawned as a post-approve-merger kanban worker. Covers the full PR merge state-machine: check PR state, handle CI pending/failed, conflicts, drafts, already-merged, and the correct kanban_complete/kanban_block contract."
version: 1.0.0
author: braintrusteng
metadata:
  hermes:
    tags: [kanban, github, merge, post-approve]
    owner: sahil
---

# post-approve-merger — PR Merge Worker

You are spawned automatically when `kanban_approve` is run on a task that
has an associated GitHub PR. Your job: merge that PR, then call
`kanban_complete` or `kanban_block`.

---

## Trigger and context

1. Call `kanban_show()` at startup (no args — defaults to your task).
2. Find the PR URL in the most recent `merge_requested` event payload:
   ```
   events where kind == "merge_requested" → payload.pr_url
   ```
   If no `merge_requested` event is found, scan task comments for a
   `https://github.com/.../pull/N` URL as fallback.
3. If you cannot find any PR URL after both passes → `kanban_complete`
   with summary "approved, no PR to merge". Stop.

---

## Auth identity

All `gh` invocations must use `GH_TOKEN=$GH_TOKEN_SAHILM_AI`:

```bash
TOKEN_VAR="GH_TOKEN_SAHILM_AI"
GH_TOKEN=$(printenv "$TOKEN_VAR") gh pr view "$PR_URL" --json state,...
```

Use `sahilm-ai` credentials exclusively. Set `GH_TOKEN` from `GH_TOKEN_SAHILM_AI` before every `gh` invocation. Never use `sahilm-ti` credentials in this worker — that is Sahil's interactive identity and using it from an automated merger will (a) misattribute the merge to a human on the audit trail and (b) trigger keychain prompts on macOS that block the worker silently.

---

## Idempotency rule (ALWAYS run first)

Before any merge attempt, call:
```bash
GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr view "$PR_URL" \
  --json state,merged,isDraft,mergeable,statusCheckRollup,baseRefName,headRefName
```

Treat the `gh pr view` output as the authoritative source of truth —
not local memory, not previous event history.

---

## PR state machine (handle every branch)

Parse the `gh pr view` JSON output and act on these cases, in order:

### 1. Already merged  (`merged == true`)
```
kanban_complete(summary="PR <url> was already merged — task done")
```
Stop. (Idempotency: safe if we crashed after the merge but before complete.)

### 2. Closed (not merged)  (`state == "CLOSED" and merged == false`)
```
kanban_block(reason="PR <url> was closed without merging — clarify intent before re-approving")
```
Stop.

### 3. Draft  (`isDraft == true`)
```
kanban_block(reason="PR <url> is still a draft — mark ready for review, then re-approve")
```
Stop.

### 4. Merge conflict  (`mergeable == "CONFLICTING"`)
Attempt one rebase:
```bash
REPO=$(GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr view "$PR_URL" \
  --json headRepositoryOwner,headRepository \
  --jq '.headRepositoryOwner.login + "/" + .headRepository.name')
HEAD_BRANCH=$(GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr view "$PR_URL" \
  --json headRefName --jq '.headRefName')
BASE_BRANCH=$(GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr view "$PR_URL" \
  --json baseRefName --jq '.baseRefName')

git clone "https://github.com/$REPO.git" /tmp/merger-rebase-$$
cd /tmp/merger-rebase-$$
git fetch origin
git checkout "$HEAD_BRANCH"
git rebase "origin/$BASE_BRANCH"
```
- If rebase succeeds → push (using `GH_TOKEN_SAHILM_AI` Basic auth extraheader, see §Auth) → proceed to §5.
- If rebase fails → `kanban_block(reason="Merge conflict on PR <url> — manual rebase needed. Conflicting files: <list>")`

### 5. CI pending  (`statusCheckRollup has items with status == "IN_PROGRESS" or "QUEUED"`)
Poll every 30 seconds, up to 10 minutes (20 polls). On each poll:
- If all checks are SUCCESS or SKIPPED → proceed to §6 (merge).
- If any check FAILED → proceed to §7 (CI failed).
- If still pending after 20 polls → `kanban_block(reason="CI still pending after 10 min on PR <url> — re-approve when CI passes")`

Check for a `ci_wait_timeout_minutes=N` annotation in the task body or
most recent comment. If found, override the 10-minute default.

### 6. Open, mergeable, CI green
```bash
GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr merge "$PR_URL" \
  --squash --delete-branch
```
On success:
```
kanban_complete(
  summary="Merged PR <url> (squash+delete-branch)",
  metadata={"pr_url": "<url>", "merge_method": "squash"}
)
```

On `gh` error → `kanban_block(reason="gh pr merge failed: <stderr>")`

### 7. CI failed  (`statusCheckRollup has items with conclusion == "FAILURE"`)
Collect the failing check names and their URLs:
```bash
GH_TOKEN=$(printenv GH_TOKEN_SAHILM_AI) gh pr view "$PR_URL" \
  --json statusCheckRollup --jq \
  '[.statusCheckRollup[] | select(.conclusion == "FAILURE") | {name: .name, detailsUrl: .detailsUrl}]'
```
```
kanban_block(reason="CI failing on PR <url> — fix and re-approve. Failing checks: <list with URLs>")
```

---

## kanban_complete / kanban_block contract

- The ONLY success terminator is `kanban_complete`. The ONLY failure terminator is `kanban_block`. Do NOT call `kanban_review` from this worker — the task was already reviewed and approved by Sahil; calling `kanban_review` here would re-loop the card through the auto-reviewer for no reason and confuse the human about whether their approval landed.
- `kanban_block` always includes the PR URL and an actionable next step in the reason.
- One and only one terminal call per run. After the single terminal call, stop — do not attempt any further `gh`, `git`, or `kanban_*` operations. A second terminal call after `kanban_complete` / `kanban_block` corrupts the task event log and confuses downstream automation.

---

## Cleanup

After a successful merge, remove the local clone if you created one:
```bash
rm -rf /tmp/merger-rebase-$$ 2>/dev/null || true
```

---

## Auth note: pushing during rebase

Use the Basic auth extraheader pattern (from session memory):
```bash
TOKEN=$(printenv GH_TOKEN_SAHILM_AI)
git -c http.https://github.com/.extraheader="Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64)" \
    -c credential.helper= \
    push origin "$HEAD_BRANCH"
```
