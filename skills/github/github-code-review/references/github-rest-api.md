# GitHub Posting Payloads — github-code-review

Full `gh` and REST/`curl` forms for every PR interaction in the skill. The `curl`
variants assume the Setup block from SKILL.md has run, so `$GITHUB_TOKEN`, `$OWNER`,
`$REPO` are set and `PR_NUMBER=N`.

## Posting with gh

```bash
# Single inline comment — head SHA required as commit_id
HEAD_SHA=$(gh pr view N --json headRefOid --jq '.headRefOid')

gh api repos/$OWNER/$REPO/pulls/N/comments \
  --method POST \
  -f body="This could be simplified with a list comprehension." \
  -f path="src/auth/login.py" \
  -f commit_id="$HEAD_SHA" \
  -f line=45 \
  -f side="RIGHT"

# Formal review
gh pr review N --approve --body "LGTM!"
gh pr review N --request-changes --body "See inline comments."
gh pr review N --comment --body "Some suggestions, nothing blocking."

# Top-level comment
gh pr comment N --body "Overall looks good, a few suggestions below."
```

## PR metadata and changed files (curl)

```bash
# PR details (title, author, description, branch, state)
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "
import sys, json
pr = json.load(sys.stdin)
print(f\"Title: {pr['title']}\")
print(f\"Author: {pr['user']['login']}\")
print(f\"Branch: {pr['head']['ref']} -> {pr['base']['ref']}\")
print(f\"State: {pr['state']}\")
print(f\"Body:\n{pr['body']}\")"

# Changed files with line counts
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/files \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    print(f\"{f['status']:10} +{f['additions']:-4} -{f['deletions']:-4}  {f['filename']}\")"
```

## Single inline comment (curl)

```bash
# Get the head commit SHA (required as commit_id)
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments \
  -d "{
    \"body\": \"This could be simplified with a list comprehension.\",
    \"path\": \"src/auth/login.py\",
    \"commit_id\": \"$HEAD_SHA\",
    \"line\": 45,
    \"side\": \"RIGHT\"
  }"
```

The `line` field is the line number in the *new* version of the file. For deleted lines,
use `"side": "LEFT"` with the line number in the *old* version. Multi-line ranges use
`start_line`/`start_side` + `line`/`side`.

## Formal review — atomic multi-comment payload (curl)

```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

# Build the review JSON — event is APPROVE, REQUEST_CHANGES, or COMMENT
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"REQUEST_CHANGES\",
    \"body\": \"Found 2 issues, 1 suggestion. See inline comments.\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"🔴 **Critical:** User input passed directly to SQL query — use parameterized queries.\"},
      {\"path\": \"src/models.py\", \"line\": 23, \"body\": \"⚠️ **Warning:** Password stored without hashing.\"},
      {\"path\": \"src/utils.py\", \"line\": 8, \"body\": \"💡 **Suggestion:** This duplicates logic in core/utils.py:34.\"}
    ]
  }"
```

## Top-level comment

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues/$PR_NUMBER/comments \
  -d '{"body": "Overall looks good, a few suggestions below."}'
```

(Top-level PR comments use the **issues** comments endpoint.)

## Failure triage

- **401/403** — token missing or lacking write scope (`repo` on a classic token, or
  Pull requests: write on a fine-grained token).
- **404** — wrong owner/repo, or a private repo without auth.
- **422** — inline anchor outside a diff hunk. Re-check the line/side against
  `gh pr diff N -- path/to/file` (old-file line number for `side: "LEFT"`), or fall back
  to a file-level comment (`"subject_type": "file"`, no line/side), or fold the point
  into the review body.
