# PR Review via the GitHub REST API

When `gh` CLI is unavailable or unauthenticated, use `curl` + `GITHUB_TOKEN` for the full review pipeline.

## Phase 0: Fetch the Diff

### ⚠️ Truncation: always save to file first

`gh pr diff` output can be silently truncated when piped through terminal (common for diffs >1000 lines). **Always save to a file first**, then read in chunks:

```bash
gh pr diff <N> --repo owner/repo > /tmp/pr<N>.diff
wc -l /tmp/pr<N>.diff                    # verify completeness
cat /tmp/pr<N>.diff | head -800 | tail -300  # read a section
```

This avoids wasting turns re-fetching a truncated diff. The `--name-only` flag works cleanly for just the file list without truncation risk.

### Endpoint

```get
GET /repos/{owner}/{repo}/pulls/{pull_number}
```
With `Accept: application/vnd.github.v3.diff` to get the raw unified diff.

### One-liner

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3.diff" \
  "https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
```

### Getting a file from a specific commit ref

To inspect a file at the PR's HEAD commit (e.g. to find line numbers for inline comments):

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={sha}" \
  | python3 -c "import sys,json,base64; print(base64.b64decode(json.load(sys.stdin)['content']).decode())"
```

### Getting the PR metadata (title, author, branch, sha, mergeability)

**⚠️ `gh pr view --json` field names are non-obvious:** Use `headRefOid` (not `headSha`), `isDraft` (not `draft`), `baseRefName` (not `base`). Running with invalid fields prints the full available field list.

```bash
gh pr view <number> --repo owner/repo --json title,author,headRefName,baseRefName,mergeable,state,isDraft,headRefOid,url,changedFiles,additions,deletions
```

Via curl + API:

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/{owner}/{repo}/pulls/{number}" \
  | python3 -c "
import sys, json
pr = json.load(sys.stdin)
print(f\"Title: {pr['title']}\")
print(f\"Author: {pr['user']['login']}\")
print(f\"Head SHA: {pr['head']['sha']}\")
print(f\"Mergeable: {pr.get('mergeable')}\")
print(f\"Draft: {pr.get('draft')}\")
"
```

## Prerequisites

- `GITHUB_TOKEN` env var set (from `.env` or other source)
- The PR's head commit SHA (needed for comment anchoring)

### ⚠️ `execute_code` does NOT inherit shell environment variables

`execute_code` runs in an isolated Python process. Shell `export` commands from `terminal()` do **not** carry over. If you try `os.environ.get("GITHUB_TOKEN")` in `execute_code` after exporting it in `terminal()`, you get `None`.

**Always read `.env` directly with Python inside `execute_code`:**

```python
import json, subprocess

def get_token():
    with open('/opt/data/.env') as f:
        for line in f:
            line = line.strip()
            if 'GITHUB_TOKEN=' in line and not line.startswith('#'):
                return line.split('=', 1)[1].strip()
    return None

GH_TOKEN = get_token()
```

**When to use which tool for posting comments:**
- `execute_code` — preferred. Isolated Python, safe for complex JSON payloads, avoids shell quoting issues with backticks in review bodies.
- `terminal()` — works for simple `curl` one-liners, but shell-piping the token can mangle special characters. Use `subprocess.run` from Python if shell quoting breaks.

### Extracting the token reliably

GITHUB_TOKEN is often stored in a `.env` file alongside many commented-out lines. Using `grep` + `cut` in a shell pipeline can return a redacted/truncated value. Use Python to read the file directly instead:

```python
def get_token():
    with open('/opt/data/.env') as f:
        for line in f:
            line = line.strip()
            if 'GITHUB_TOKEN=' in line and not line.startswith('#'):
                return line.split('=', 1)[1].strip()
    return None
```

If `curl` to the API returns empty output but the token looks right, the issue is often shell piping — the token may be getting mangled by intermediate commands. Use Python's `subprocess.run()` to call curl directly, constructing the auth header with string concatenation (not shell interpolation).

### ⚠️ `gh` CLI from `execute_code` needs `GH_TOKEN` explicitly

The `gh` CLI looks for `GH_TOKEN` (not `GITHUB_TOKEN`) in the subprocess environment. Even if `gh auth status` works in `terminal()`, subprocess calls from `execute_code` won't inherit the auth. Always pass it explicitly:

```python
import os, subprocess

env = os.environ.copy()
env['GH_TOKEN'] = GH_TOKEN  # not GITHUB_TOKEN

result = subprocess.run(
    ['gh', 'pr', 'diff', '123', '--repo', 'owner/repo'],
    capture_output=True, text=True, env=env
)
```

Without this, `gh` returns empty stdout with `gh auth login` instructions on stderr.

### Parsing diff line numbers for inline comment anchoring

To post inline comments at the correct new-file line, parse the unified diff to find the line number of each `+` line. Use the reusable script at `scripts/parse-diff-lines.py`:

```bash
gh pr diff 123 --repo owner/repo | python3 scripts/parse-diff-lines.py
# Output: [["path/to/file.ts", 42], ["other/file.ts", 17], ...]
```

The script handles: context lines (` `), added lines (`+`), removed lines (`-`), and multi-hunk diffs. Returns `(filepath, new_line_number)` pairs for every added line — exactly what the inline comment `line` field needs.

For `gh` CLI calls from `execute_code`, combine with the `GH_TOKEN` pattern above:

```python
result = subprocess.run(
    ['gh', 'pr', 'diff', str(PR_NUMBER), '--repo', REPO],
    capture_output=True, text=True, env=env  # env has GH_TOKEN
)
from scripts.parse_diff_lines import parse_diff
changes = parse_diff(result.stdout)  # [(filepath, line_num), ...]
```

## Two-Phase Workflow

**Phase 1 — Post each finding as an individual inline comment** (reliable, always works).

**Phase 2 — Post the review summary body** on the issue thread (for scannability).

Phase 1 uses the **review comments** endpoint; Phase 2 uses the **issue comments** endpoint.

## Phase 1: Individual Inline Comments

### Endpoint

```
POST /repos/{owner}/{repo}/pulls/{pull_number}/comments
```

### Request body

```json
{
  "commit_id": "abc123def...",
  "path": "path/to/file.ts",
  "line": 123,
  "side": "RIGHT",
  "body": "⚠️ **Warning:** Your comment text here"
}
```

| Field | Required | Notes |
|---|---|---|
| `commit_id` | Yes | HEAD SHA of the PR branch |
| `path` | Yes | Relative path from repo root |
| `line` | Yes | Line number **in the new file** (not diff position) |
| `side` | Yes | `"RIGHT"` for additions/context, `"LEFT"` for deletions |
| `body` | Yes | Markdown text; prefix with severity icon |

### Critical: Can only comment on files in the PR diff

A frequently-surprising constraint: **you can only post inline comments on files that were changed in the PR diff.** GitHub's API returns:

```json
{"message": "Validation Failed", "errors": [{
  "resource": "PullRequestReviewComment",
  "code": "invalid",
  "field": "pull_request_review_thread.path",
  "message": "could not be resolved"
}]}
```

If your finding relates to a file **not** in the diff (e.g. `nx.json`'s `installation.version` is out of sync with `package.json`), you cannot comment inline on that file directly. **Workaround:** Post the inline comment on the nearest related line in a changed file (e.g. the version bump line in `package.json`), and describe the unrelated file path in the comment body:

```
⚠️ **Warning:** The `installation.version` in `nx.json` is still `"21.3.10"`
but this PR updates `package.json` to `22.7.3`. Update `nx.json` as well.
```

### Critical (sub-file level): Line must be within a diff hunk

Even when the file **is** in the diff, you can only comment on lines that fall within a diff hunk — that is, lines that were changed, added, or deleted, plus a few lines of surrounding context (the diff's "neighbourhood"). Attempting to comment on a line in a changed file that was **not** part of any diff hunk produces a different error:

```json
{"message": "Validation Failed", "errors": [{
  "resource": "PullRequestReviewComment",
  "code": "custom",
  "field": "pull_request_review_thread.line",
  "message": "could not be resolved"
}]}
```

Note the `field` is `pull_request_review_thread.line` (not `.path`) and the `code` is `"custom"` (not `"invalid"`). The file resolved fine — the API just couldn't map the line to any diff hunk.

**This trap is most common when:**
- Commenting on a ConfigMap structural line like `data:` that exists in the file but wasn't modified
- Commenting on a file header or import block that wasn't touched
- Picking a line number from a full-file fetch (Git Blob API) that corresponds to an unchanged region

**Workaround:** Choose the closest line that **was** changed in the diff. If the finding is about the PR description or an overall pattern (not tied to any particular changed line), skip the inline comment and include the finding only in the review summary body — an inline comment with no anchor is worse than no inline comment.

#### Debugging: line number not in diff hunk

When the API returns `"field": "pull_request_review_thread.line"` with code `"custom"`:

0. **Verify your line numbers with the parser.** Before manual debugging, confirm the line came from `scripts/parse-diff-lines.py` or equivalent diff parsing. Most "line not in hunk" errors are caused by incorrect line number calculation (e.g. counting from the old file instead of the new file, or not handling multi-hunk diffs).

1. **Determine the valid line range from the diff header.** Each hunk starts with `@@ -old,count +new,count @@`. The `+new` offset is the first line in the **new file** that belongs to the hunk. The hunk covers `new` through `new + count - 1`. Only those lines (plus ~3 lines of context on each side that were unchanged) accept inline comments. A line number outside this range will fail.

2. **Re-count from the full file.** Fetch the file at the PR commit via the Git Blob API and count lines carefully. Off-by-one errors are common — count from line 1, not from `cat -n` or a misaligned copy.

3. **Surgical test-post (if not sure).** Write a small Python script that posts a minimal test comment (`body: "test"`) to the suspected line. If it fails with the line field error, try line+1, line-1, and line+2 until one succeeds. Then **delete the test comment** via `DELETE /repos/{owner}/{repo}/pulls/comments/{id}` before posting the real one. This is not the cleanest approach but beats blind retries.

4. **Cleanup checkpoint.** After posting the real comment, verify the repo doesn't have orphaned test comments. The `GET /repos/{owner}/{repo}/pulls/{number}/comments` endpoint returns all live inline comments — scan for ones with `body: "test"` and delete them.

### Critical: Avoid the bundled `reviews` API for inline comments

Do **NOT** use this pattern for inline comments:

```
POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews
{
  "comments": [{ ... }, { ... }]  // <-- may silently drop comments
}
```

The bundled `reviews` API with `comments` array can silently drop inline comments without error. Always post each inline comment individually via the `/pulls/{number}/comments` endpoint instead.

### Prefix every inline comment with the severity icon

The diff view needs the severity front-and-centre for scanability:

```
🔴 **Critical:** ...
⚠️ **Warning:** ...
💡 **Suggestion:** ...
✅ **Nice:** ...
```

### How to get the commit SHA

From the PR API:
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/{owner}/{repo}/pulls/{number}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])"
```

Or from the commits list:
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/{owner}/{repo}/pulls/{number}/commits" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)[-1]['sha'])"
```

## Phase 2: Review Summary Body

Post the full review body (Scope + Summary + Findings listing + Verdict) as an issue comment:

```
POST /repos/{owner}/{repo}/issues/{pull_number}/comments
```

The `body` field contains the full markdown review. The inline comments (Phase 1) are the authoritative record; this summary is for scannability and the verdict.

## Handling Token Scope Issues

A token that works for **reading** (`GET` endpoints returning 200) may fail for **writing** (`POST` returning 401 "Bad credentials"). In that case:

- Check the token's scopes: does it have `repo` (full control) or at minimum `public_repo` + write access?
- For private repos, a fine-grained PAT needs `pull_requests: write` permission.
- If writing fails, post the full review as a PR **issue comment** only (loses inline comments, but the review isn't lost).

## Practical Tips (Real-World Reliability)

### Avoid inline `python3 -c` for complex payloads

Inline `python3 -c "..."` in bash interprets backticks as command substitution, even inside quoted strings. This breaks review bodies that contain backtick-escaped code snippets or JSON with escaped quotes.

**Do this:** write a `.py` script file and run it with `python3 /tmp/script.py`.
**Don't do this:** embed Python code in a `python3 -c "..."` shell argument with complex strings.

When using `subprocess.run` from a Python script, construct tokens and URLs with string concatenation (`'Authorization: token ' + TOKEN`) rather than f-string interpolation to avoid shell injection issues.

### Verify the token before investing in analysis

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/user" \
  | python3 -c "import sys,json; u=json.load(sys.stdin); print(u.get('login','FAIL'))"
```

If this returns `FAIL`, the token is invalid, expired, or has no scopes.

### Empty curl response from GitHub API

If `curl -H "Authorization: token $TOKEN"` returns blank/empty to API endpoints:
1. The token might contain special characters that shell quoting mangled. Use Python `subprocess.run` instead of shell pipes.
2. Test read access with the `/user` endpoint first (above).
3. Ensure the token isn't being redacted in shell output (`.env` files with `grep | cut` can lose characters). Read the `.env` file with Python instead.

## Implementing the Loop in Python

```python
import json, subprocess

GH_TOKEN = "..."
COMMIT_SHA = "abc..."
BASE_URL = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"

# Phase 1: Post inline comments
inline_comments = [
    {
        "commit_id": COMMIT_SHA,
        "path": "path/to/file.ts",
        "line": 123,
        "side": "RIGHT",
        "body": "⚠️ **Warning:** Issue description"
    },
]

for c in inline_comments:
    result = subprocess.run([
        'curl', '-s', '-X', 'POST',
        f"{BASE_URL}/comments",
        '-H', f'Authorization: token {GH_TOKEN}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(c)
    ], capture_output=True, text=True)
    resp = json.loads(result.stdout)
    if 'id' in resp:
        print(f"Posted comment #{resp['id']}")
    else:
        print(f"Failed: {resp.get('message', 'unknown')}")

# Phase 2: Post review summary as issue comment
summary = "## Code Review – Scope\n\n..."
subprocess.run([
    'curl', '-s', '-X', 'POST',
    f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
    '-H', f'Authorization: token {GH_TOKEN}',
    '-H', 'Content-Type: application/json',
    '-d', json.dumps({"body": summary})
])
```
