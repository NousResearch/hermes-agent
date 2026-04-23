---
name: gitlab-mr-workflow
description: Full Merge Request lifecycle — create branches, commit changes, open MRs, monitor CI pipelines, auto-fix failures, and merge. Works with gitlab.com and self-hosted GitLab.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [GitLab, Merge-Requests, CI/CD, Git, Automation, Merge]
    related_skills: [gitlab-auth, gitlab-code-review]
---

# GitLab Merge Request Workflow

Complete guide for managing the MR lifecycle on GitLab. Uses the `gitlab-review` plugin tools for API interactions and `git` for local operations.

## Prerequisites

- `GITLAB_TOKEN` environment variable set (see `gitlab-auth` skill)
- `GITLAB_URL` set for self-hosted GitLab (default: https://gitlab.com)
- Inside a git repository with a GitLab remote

### Quick Auth Check

```bash
if [ -n "${GITLAB_TOKEN:-}" ]; then
  echo "GitLab: ready (${GITLAB_URL:-https://gitlab.com})"
else
  echo "GitLab: NOT CONFIGURED — set GITLAB_TOKEN"
fi
```

---

## 1. Branch Creation

```bash
# Make sure you're up to date
git fetch origin
git checkout main && git pull origin main

# Create and switch to a new branch
git checkout -b feat/add-user-authentication
```

Branch naming conventions:
- `feat/description` — new features
- `fix/description` — bug fixes
- `refactor/description` — code restructuring
- `docs/description` — documentation
- `ci/description` — CI/CD changes

## 2. Making Commits

Use the agent's file tools (`write_file`, `patch`) to make changes, then commit:

```bash
# Stage specific files
git add src/auth.py src/models/user.py tests/test_auth.py

# Commit with a conventional commit message
git commit -m "feat: add JWT-based user authentication

- Add login/register endpoints
- Add User model with password hashing
- Add auth middleware for protected routes
- Add unit tests for auth flow"
```

Commit message format (Conventional Commits):
```
type(scope): short description

Longer explanation if needed. Wrap at 72 characters.
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`

## 3. Pushing and Creating an MR

### Push the Branch

```bash
git push -u origin HEAD
```

### Create the MR via GitLab API

Use the terminal tool to create an MR:

```bash
PROJECT="group/project"
BRANCH=$(git branch --show-current)

curl -s -X POST \
  --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  --header "Content-Type: application/json" \
  "${GITLAB_URL:-https://gitlab.com}/api/v4/projects/$(python3 -c "import urllib.parse; print(urllib.parse.quote('$PROJECT', safe=''))")/merge_requests" \
  --data "{
    \"title\": \"feat: add JWT-based user authentication\",
    \"description\": \"## Summary\nAdds login and register API endpoints.\n\nCloses #42\",
    \"source_branch\": \"$BRANCH\",
    \"target_branch\": \"main\",
    \"remove_source_branch\": true
  }"
```

The response JSON includes the MR `iid` — save it for later commands.

To create as a draft, add `"draft\": true` to the JSON body.

## 4. Monitoring CI/CD Pipelines

### Check Pipeline Status

Use the plugin tool:

```
gitlab_mr_pipelines(project="group/project", mr_iid=42)
```

Or via API:

```bash
# Get pipelines for an MR
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "${GITLAB_URL:-https://gitlab.com}/api/v4/projects/$(python3 -c "import urllib.parse; print(urllib.parse.quote('group/project', safe=''))")/merge_requests/42/pipelines"
```

### Get Pipeline Job Logs

```
gitlab_pipeline_jobs(project="group/project", pipeline_id=12345)
```

### Poll Until Complete

```bash
PIPELINE_ID=12345
PROJECT="group/project"
ENCODED_PROJECT=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$PROJECT', safe=''))")

for i in $(seq 1 20); do
  STATUS=$(curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
    "${GITLAB_URL:-https://gitlab.com}/api/v4/projects/$ENCODED_PROJECT/pipelines/$PIPELINE_ID" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Check $i: $STATUS"
  if [ "$STATUS" = "success" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "canceled" ]; then
    break
  fi
  sleep 30
done
```

## 5. Auto-Fixing CI Failures

When CI fails, diagnose and fix:

### Step 1: Get Failure Details

```
gitlab_pipeline_jobs(project="group/project", pipeline_id=12345)
```

This returns job details and trace (log) excerpts for failed jobs.

### Step 2: Fix and Push

After identifying the issue, use file tools (`patch`, `write_file`) to fix it:

```bash
git add <fixed_files>
git commit -m "fix: resolve CI failure in <check_name>"
git push
```

### Step 3: Retry the Pipeline

```
gitlab_pipeline_retry(project="group/project", pipeline_id=12345)
```

### Auto-Fix Loop Pattern

1. Check pipeline status → identify failures
2. Read failure logs → understand the error
3. Use `read_file` + `patch`/`write_file` → fix the code
4. `git add . && git commit -m "fix: ..." && git push`
5. Wait for pipeline → re-check status
6. Repeat if still failing (up to 3 attempts, then ask the user)

## 6. Merging

### Via Plugin Tool

Use the `gitlab_mr_review` tool to approve, then accept the merge via API:

```bash
PROJECT="group/project"
MR_IID=42
ENCODED_PROJECT=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$PROJECT', safe=''))")

# Squash merge
curl -s -X PUT \
  --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  --header "Content-Type: application/json" \
  "${GITLAB_URL:-https://gitlab.com}/api/v4/projects/$ENCODED_PROJECT/merge_requests/$MR_IID/merge" \
  --data "{
    \"squash\": true,
    \"squash_commit_message\": \"feat: add user authentication (!42)\",
    \"should_remove_source_branch\": true
  }"
```

Merge methods (set per project in GitLab settings):
- `"squash"` — squash all commits into one (cleanest for feature branches)
- `"merge"` — create a merge commit
- `"rebase"` — rebase the branch (no merge commit)

### Delete the Local Branch After Merge

```bash
git checkout main && git pull origin main
git branch -d feat/add-user-authentication
```

## 7. Complete Workflow Example

```bash
# 1. Start from clean main
git checkout main && git pull origin main

# 2. Branch
git checkout -b fix/login-redirect-bug

# 3. (Agent makes code changes with file tools)

# 4. Commit
git add src/auth/login.py tests/test_login.py
git commit -m "fix: correct redirect URL after login

Preserves the ?next= parameter instead of always redirecting to /dashboard."

# 5. Push
git push -u origin HEAD

# 6. Create MR (see Section 3)

# 7. Monitor CI (see Section 4)

# 8. Merge when green (see Section 6)
```

---

## Useful MR Commands Reference

| Action | Plugin Tool / API |
|--------|-------------------|
| View MR | `gitlab_mr_view(project, mr_iid)` |
| Get diff | `gitlab_mr_diff(project, mr_iid)` |
| List changed files | `gitlab_mr_list_files(project, mr_iid)` |
| Post comment | `gitlab_mr_comments(project, mr_iid, body)` |
| Post inline comment | `gitlab_mr_inline_comment(project, mr_iid, file_path, line, body, head_sha)` |
| Submit review | `gitlab_mr_review(project, mr_iid, action, body)` |
| List MRs | `gitlab_mr_list(project, state)` |
| Check pipelines | `gitlab_mr_pipelines(project, mr_iid)` |
| Get job logs | `gitlab_pipeline_jobs(project, pipeline_id)` |
| Retry pipeline | `gitlab_pipeline_retry(project, pipeline_id)` |
| View discussions | `gitlab_mr_discussions(project, mr_iid)` |
| Get MR context | `gitlab_mr_context(project, mr_iid)` |
