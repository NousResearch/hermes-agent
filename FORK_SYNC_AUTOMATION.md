# Fork sync automation setup

Cursor Cloud Agent automation for recovering from failed `sync-upstream.yml`
runs. Code lives in this repo (`ops-overlay` → daily cherry-pick to `main`).

## One-time setup (human)

### 1. Create Cursor Automation

1. Open [cursor.com/automations/new](https://cursor.com/automations/new)
2. **Trigger:** Webhook
3. **Repository:** `pebble-tech/hermes-agent`, default branch `main` (agent reads skill from checkout)
4. **Tools:** enable **Open pull request** (preferred — PRs target source branches, not `main`)
5. **Prompt** (paste and adjust):

```
You are recovering a failed pebble-tech/hermes-agent upstream sync.

1. Read the webhook JSON payload (event, upstream_sha, failed_step, failed_branch, conflict_files, workflow_run_url, issue_number).
2. Read FORK.md and follow .cursor/skills/recover-fork-sync/SKILL.md exactly.
3. Fix the failing SOURCE branch (ops-overlay or a FEATURE_BRANCHES entry). Never commit recovery fixes to integration main.
4. If upstream absorbed the patch (closed/superseded PR), drop the branch from FEATURE_BRANCHES via an ops-overlay PR.
5. Open a PR to the source branch you fixed (not to main).
6. After merge, run: gh workflow run sync-upstream.yml --repo pebble-tech/hermes-agent --ref main
7. Wait for sync success, comment on the sync-failure issue, close it.

Use gh and git. Run focused tests from the skill before opening the PR.
```

6. Save the automation → copy **Webhook URL** and **API key**

### 2. GitHub repository secrets

In `pebble-tech/hermes-agent` → Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|--------|
| `CURSOR_SYNC_RECOVERY_WEBHOOK_URL` | Webhook URL from step 1 |
| `CURSOR_SYNC_RECOVERY_WEBHOOK_KEY` | API key from step 1 |

Existing `SYNC_PUSH_TOKEN` is unchanged (workflow push token).

Optional for cloud agent sandbox (if `gh issue view` fails with permission errors):

| Secret / env | Where |
|--------------|--------|
| `GH_TOKEN` | Cursor automation environment variables (PAT with `repo` scope) |

### 3. Merge ops-overlay changes

Automation wiring is committed on **`ops-overlay`**. After merge:

```bash
gh workflow run sync-upstream.yml --repo pebble-tech/hermes-agent --ref main
```

### 4. Test the webhook (optional)

Dry-run payload:

```bash
curl -sf -X POST "$CURSOR_SYNC_RECOVERY_WEBHOOK_URL" \
  -H "Authorization: Bearer $CURSOR_SYNC_RECOVERY_WEBHOOK_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "sync-failure",
    "repository": "pebble-tech/hermes-agent",
    "upstream_sha": "test",
    "failed_step": "rebase-feature-branch",
    "failed_branch": "feature/example-upstream",
    "conflict_files": ["gateway/example.py"],
    "workflow_run_url": "https://github.com/pebble-tech/hermes-agent/actions/runs/0",
    "issue_number": null
  }'
```

Cancel the agent run if this was only a connectivity test.

## What runs automatically

On sync workflow **failure**:

1. Writes `.sync-failure.json` (step, branch, conflict files)
2. Opens/updates GitHub issue labelled `sync-failure` with a machine-readable block
3. POSTs the same payload to the Cursor webhook (if secrets are set)

On sync **success**: no webhook, no new issue.

## Security

- Webhook URL + key are repo secrets — never commit them
- Automation should open PRs to source branches for human or policy review before merge
- Do not grant the agent direct push to `main`
