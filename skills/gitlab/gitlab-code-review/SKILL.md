---
name: gitlab-code-review
description: Review code changes on GitLab Merge Requests by analyzing diffs, leaving inline comments, and performing thorough pre-merge review. Uses the gitlab-review plugin tools for native API integration.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [GitLab, Code-Review, Merge-Requests, Quality]
    related_skills: [gitlab-auth, gitlab-mr-workflow]
---

# GitLab Merge Request Code Review

Perform code reviews on GitLab Merge Requests using native API tools. The `gitlab-review` plugin provides first-class tools for viewing MRs, reading diffs, posting inline comments, and submitting formal reviews.

## Prerequisites

- `GITLAB_TOKEN` environment variable set (see `gitlab-auth` skill)
- `GITLAB_URL` set for self-hosted GitLab (default: https://gitlab.com)
- The `gitlab-review` plugin enabled

### Quick Auth Check

```bash
if [ -n "${GITLAB_TOKEN:-}" ]; then
  echo "GitLab: ready (${GITLAB_URL:-https://gitlab.com})"
else
  echo "GitLab: NOT CONFIGURED — set GITLAB_TOKEN"
fi
```

---

## 1. Reviewing a Merge Request

### Step 1: Get MR Metadata

Use the `gitlab_mr_view` tool to understand the MR scope:

```
gitlab_mr_view(project="group/project", mr_iid=42)
```

This returns: title, description, author, branches, state, labels, merge status.

### Step 2: Check Existing Discussions

Avoid duplicating feedback already given:

```
gitlab_mr_discussions(project="group/project", mr_iid=42)
```

### Step 3: See the Scope of Changes

```
gitlab_mr_list_files(project="group/project", mr_iid=42)
```

This lists all changed files — use it to prioritize which files need the most attention.

### Step 4: Read the Full Diff

```
gitlab_mr_diff(project="group/project", mr_iid=42)
```

For large MRs, review the diff file by file. Use the `read_file` tool on individual files for full context around the changes.

### Step 5: Check CI/CD Pipeline Status

```
gitlab_mr_pipelines(project="group/project", mr_iid=42)
```

If pipelines are failing, investigate:

```
gitlab_pipeline_jobs(project="group/project", pipeline_id=12345)
```

### Step 6: Check Related Context

```
gitlab_mr_context(project="group/project", mr_iid=42)
```

This shows issues the MR closes and commits in the branch comparison.

### Step 7: Apply the Review Checklist

Go through each category systematically (see Section 2 below).

### Step 8: Post the Review

#### General Summary Comment

```
gitlab_mr_comments(project="group/project", mr_iid=42, body="## Code Review Summary\n\n...")
```

#### Inline Comments on Specific Lines

```
gitlab_mr_inline_comment(
    project="group/project",
    mr_iid=42,
    file_path="src/auth/login.py",
    line=45,
    body="🔴 **Critical:** SQL injection — use parameterized queries.",
    head_sha="abc123..."
)
```

The `head_sha` is available from `gitlab_mr_view` output.

#### Submit Formal Review

```
# Approve
gitlab_mr_review(project="group/project", mr_iid=42, action="approve", body="LGTM!")

# Request changes
gitlab_mr_review(project="group/project", mr_iid=42, action="request_changes", body="See inline comments.")

# Comment only (no approval change)
gitlab_mr_review(project="group/project", mr_iid=42, action="comment", body="Some suggestions, nothing blocking.")
```

---

## 2. Review Checklist

When performing a code review, systematically check:

### Correctness
- Does the code do what it claims?
- Edge cases handled (empty inputs, nulls, large data, concurrent access)?
- Error paths handled gracefully?
- No off-by-one errors or logic inversions?

### Security
- No hardcoded secrets, credentials, or API keys in the diff
- Input validation on user-facing inputs
- No SQL injection, XSS, or path traversal
- Auth/authz checks where needed
- No sensitive data logged or exposed in error messages

### Code Quality
- Clear naming (variables, functions, classes)
- No unnecessary complexity or premature abstraction
- DRY — no duplicated logic that should be extracted
- Functions are focused (single responsibility)
- No dead code or commented-out blocks

### Testing
- New code paths tested?
- Happy path and error cases covered?
- Tests readable and maintainable?
- No flaky test patterns (time-dependent, race conditions)?

### Performance
- No N+1 queries or unnecessary loops
- Appropriate caching where beneficial
- No blocking operations in async code paths
- Database queries use indexes effectively

### Documentation
- Public APIs documented
- Non-obvious logic has comments explaining "why"
- README updated if behavior changed
- Migration/deployment notes if applicable

---

## 3. Review Output Format

When posting a review, use this structured format for the summary comment:

```markdown
## Code Review Summary

**Verdict: [Approved / Changes Requested / Comment]**

### 🔴 Critical
- **src/auth.py:45** — SQL injection: user input passed directly to query.
  Suggestion: Use parameterized queries.

### ⚠️ Warnings
- **src/models/user.py:23** — Password stored in plaintext. Use bcrypt or argon2.
- **src/api/routes.py:112** — No rate limiting on login endpoint.

### 💡 Suggestions
- **src/utils/helpers.py:8** — Duplicates logic in `src/core/utils.py:34`. Consolidate.
- **tests/test_auth.py** — Missing edge case: expired token test.

### ✅ Looks Good
- Clean separation of concerns in the middleware layer
- Good test coverage for the happy path
- Proper error handling in the data transformation pipeline

---
*Reviewed by Hermes Agent*
```

For the inline comment format:

- 🔴 **Critical** — Must fix before merge
- ⚠️ **Warning** — Should fix, but not a blocker
- 💡 **Suggestion** — Nice to have, optional improvement

---

## 4. Decision: Approve vs Request Changes vs Comment

- **Approve** — No critical or warning-level issues, only minor suggestions or all clear
- **Request Changes** — Any critical or warning-level issue that should be fixed before merge
- **Comment** — Observations and suggestions, but nothing blocking (use when unsure or the MR is a draft)

---

## 5. Self-Hosted GitLab

All tools work identically with self-hosted GitLab. The `GITLAB_URL` environment variable controls the target instance:

```bash
# Point to your self-hosted instance
export GITLAB_URL="https://gitlab.mycompany.com"
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"
```

No other configuration changes are needed — the tools use `GITLAB_URL` as the API base.

---

## 6. Complete Review Workflow (End-to-End)

When asked to "review MR #42" or "review the merge request":

1. `gitlab_mr_view(project, mr_iid=42)` — understand scope and context
2. `gitlab_mr_discussions(project, mr_iid=42)` — check existing feedback
3. `gitlab_mr_list_files(project, mr_iid=42)` — see changed files
4. `gitlab_mr_diff(project, mr_iid=42)` — read the full diff
5. `gitlab_mr_pipelines(project, mr_iid=42)` — check CI status
6. For each changed file, use `read_file` if you need more context
7. Apply the review checklist (Section 2)
8. Post inline comments with `gitlab_mr_inline_comment` for specific issues
9. Post summary with `gitlab_mr_comments`
10. Submit formal review with `gitlab_mr_review` (approve / request changes / comment)

### Extracting project path

The `project` parameter accepts either:
- A project path: `"group/project"` or `"group/subgroup/project"`
- A numeric project ID: `"12345"`

You can find the project path from the GitLab URL: `https://gitlab.com/group/project/-/merge_requests/42` → project is `"group/project"`.
