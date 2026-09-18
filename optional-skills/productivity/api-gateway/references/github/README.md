# GitHub Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `github`
**Base URL proxied:** `api.github.com`

## API Path Pattern

```
/github/{resource}
```

GitHub API does not use a version prefix in paths. Versioning is handled via the `X-GitHub-Api-Version` header.

## Common Endpoints

### Get Authenticated User
```bash
GET /github/user
```

Example:

```bash
maton github whoami
```

### Get User by Username
```bash
GET /github/users/{username}
```

### List User Repositories
```bash
GET /github/user/repos?per_page=30&sort=updated
```

Example:

```bash
maton github repo list --sort updated
```

### List Organization Repositories
```bash
GET /github/orgs/{org}/repos?per_page=30
```

Example:

```bash
maton github repo list {org}
```

### Get Repository
```bash
GET /github/repos/{owner}/{repo}
```

Example:

```bash
maton github repo view --repo {owner}/{repo}
```

### Create Repository (User)
```bash
POST /github/user/repos
Content-Type: application/json

{
  "name": "my-new-repo",
  "description": "A new repository",
  "private": true,
  "auto_init": true
}
```

Example:

```bash
maton github repo create my-new-repo --description "A new repository" --visibility private
```

### Create Repository (Organization)
```bash
POST /github/orgs/{org}/repos
Content-Type: application/json

{
  "name": "my-new-repo",
  "private": true
}
```

Example:

```bash
maton github repo create {org}/my-new-repo --visibility private
```

### Update Repository
```bash
PATCH /github/repos/{owner}/{repo}
Content-Type: application/json

{
  "description": "Updated description",
  "has_issues": true
}
```

Example:

```bash
maton github repo edit --repo {owner}/{repo} --description "Updated description" --enable-issues
```

### List Repository Contents
```bash
GET /github/repos/{owner}/{repo}/contents/{path}
```

### Get File Contents
```bash
GET /github/repos/{owner}/{repo}/contents/{path}?ref={branch}
```

### Create or Update File
```bash
PUT /github/repos/{owner}/{repo}/contents/{path}
Content-Type: application/json

{
  "message": "Create new file",
  "content": "SGVsbG8gV29ybGQh",
  "branch": "main"
}
```

Note: `content` must be Base64 encoded.

### List Branches
```bash
GET /github/repos/{owner}/{repo}/branches?per_page=30
```

### Get Branch
```bash
GET /github/repos/{owner}/{repo}/branches/{branch}
```

### Merge Branches
```bash
POST /github/repos/{owner}/{repo}/merges
Content-Type: application/json

{
  "base": "main",
  "head": "feature-branch",
  "commit_message": "Merge feature branch"
}
```

### List Commits
```bash
GET /github/repos/{owner}/{repo}/commits?per_page=30
```

Query parameters: `sha`, `path`, `author`, `committer`, `since`, `until`, `per_page`, `page`

### Get Commit
```bash
GET /github/repos/{owner}/{repo}/commits/{ref}
```

### Compare Two Commits
```bash
GET /github/repos/{owner}/{repo}/compare/{base}...{head}
```

### List Repository Issues
```bash
GET /github/repos/{owner}/{repo}/issues?state=open&per_page=30
```

Query parameters: `state` (open, closed, all), `labels`, `assignee`, `creator`, `mentioned`, `sort`, `direction`, `since`, `per_page`, `page`

Example:

```bash
maton github issue list --repo {owner}/{repo} --state open
```

### Get Issue
```bash
GET /github/repos/{owner}/{repo}/issues/{issue_number}
```

Example:

```bash
maton github issue view {issue_number} --repo {owner}/{repo}
```

### Create Issue
```bash
POST /github/repos/{owner}/{repo}/issues
Content-Type: application/json

{
  "title": "Found a bug",
  "body": "Bug description here",
  "labels": ["bug"],
  "assignees": ["username"]
}
```

Example:

```bash
maton github issue create --repo {owner}/{repo} --title "Found a bug" --body "Bug description here" --label bug --assignee username
```

### Update / Close Issue
```bash
PATCH /github/repos/{owner}/{repo}/issues/{issue_number}
Content-Type: application/json

{
  "state": "closed",
  "state_reason": "completed"
}
```

Example:

```bash
maton github issue close {issue_number} --repo {owner}/{repo} --reason completed
```

### List Issue Comments
```bash
GET /github/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=30
```

Example:

```bash
maton github issue view {issue_number} --repo {owner}/{repo} --comments
```

### Create Issue Comment
```bash
POST /github/repos/{owner}/{repo}/issues/{issue_number}/comments
Content-Type: application/json

{
  "body": "This is a comment"
}
```

Example:

```bash
maton github issue comment {issue_number} --repo {owner}/{repo} --body "This is a comment"
```

### List Labels
```bash
GET /github/repos/{owner}/{repo}/labels?per_page=30
```

Example:

```bash
maton github label list --repo {owner}/{repo}
```

### Create Label
```bash
POST /github/repos/{owner}/{repo}/labels
Content-Type: application/json

{
  "name": "priority:high",
  "color": "ff0000",
  "description": "High priority issues"
}
```

Example:

```bash
maton github label create "priority:high" --repo {owner}/{repo} --color ff0000 --description "High priority issues"
```

### List Pull Requests
```bash
GET /github/repos/{owner}/{repo}/pulls?state=open&per_page=30
```

Query parameters: `state` (open, closed, all), `head`, `base`, `sort`, `direction`, `per_page`, `page`

Example:

```bash
maton github pr list --repo {owner}/{repo} --state open
```

### Get Pull Request
```bash
GET /github/repos/{owner}/{repo}/pulls/{pull_number}
```

Example:

```bash
maton github pr view {pull_number} --repo {owner}/{repo}
```

### Create Pull Request
```bash
POST /github/repos/{owner}/{repo}/pulls
Content-Type: application/json

{
  "title": "New feature",
  "body": "Description of changes",
  "head": "feature-branch",
  "base": "main",
  "draft": false
}
```

Example:

```bash
maton github pr create --repo {owner}/{repo} --base main --head feature-branch --title "New feature" --body "Description of changes"
```

### List Pull Request Files
```bash
GET /github/repos/{owner}/{repo}/pulls/{pull_number}/files?per_page=30
```

Example:

```bash
maton github pr diff {pull_number} --repo {owner}/{repo}
```

### Merge Pull Request
```bash
PUT /github/repos/{owner}/{repo}/pulls/{pull_number}/merge
Content-Type: application/json

{
  "commit_title": "Merge pull request",
  "merge_method": "squash"
}
```

Merge methods: `merge`, `squash`, `rebase`.

Example:

```bash
maton github pr merge {pull_number} --repo {owner}/{repo} --squash --delete-branch
```

### Create Pull Request Review
```bash
POST /github/repos/{owner}/{repo}/pulls/{pull_number}/reviews
Content-Type: application/json

{
  "body": "Looks good!",
  "event": "APPROVE"
}
```

Events: `APPROVE`, `REQUEST_CHANGES`, `COMMENT`.

Example:

```bash
maton github pr review {pull_number} --repo {owner}/{repo} --approve --body "Looks good!"
```

Note: GitHub does not allow approving your own pull requests; `--approve` returns `422 Can not approve your own pull request` in that case. Use `--comment` or `--request-changes` instead.

### Search Repositories
```bash
GET /github/search/repositories?q={query}&per_page=30
```

Example:

```bash
maton github repo search tetris --language python
```

### Search Issues
```bash
GET /github/search/issues?q={query}&per_page=30
```

Example:

```bash
maton github issue search "bug" --state open
```

### Search Code
```bash
GET /github/search/code?q={query}&per_page=30
```

Note: Code search may timeout (`408`) on broad queries. Always scope with `repo:`, `org:`, `user:`, or `extension:`.

### List User Organizations
```bash
GET /github/user/orgs?per_page=30
```

Note: Requires `read:org` scope.

### Get Rate Limit
```bash
GET /github/rate_limit
```

## Pagination

GitHub uses page-based pagination via the `Link` response header. The CLI handles this automatically with `--paginate`:

```bash
maton github repo list --paginate
```

For raw HTTP requests, use `per_page` (max 100, default 30) and `page` query parameters, or follow the `rel="next"` URL in the `Link` response header.

## Notes

- Repository names are case-insensitive but the API preserves case
- Issue numbers and PR numbers share the same sequence per repository — a PR is also an issue
- File `content` must be Base64 encoded when creating/updating files via the contents API
- File update/delete requires the current `sha` of the file (fetch via GET first)
- Rate limits: 5,000 requests/hour for authenticated users; search is 30 requests/minute
- Some endpoints require specific OAuth scopes (e.g., `read:org` for organization operations). If you receive a scope error, contact Maton support at support@maton.ai
- Search queries may timeout (`408`) on very broad patterns — always scope code search to a repo or org
- Cannot approve your own pull requests; use `COMMENT` or `REQUEST_CHANGES` events instead

## Resources

- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [Repositories API](https://docs.github.com/en/rest/repos/repos)
- [Issues API](https://docs.github.com/en/rest/issues/issues)
- [Pull Requests API](https://docs.github.com/en/rest/pulls/pulls)
- [Search API](https://docs.github.com/en/rest/search/search)
- [Rate Limits](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting)
- [Maton CLI Manual](https://cli.maton.ai/manual)