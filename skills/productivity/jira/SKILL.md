---
name: jira
description: "Jira Cloud: search, create, update, transition issues, manage projects and comments."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  tools: [jira_issue, jira_search, jira_project, jira_comment]
metadata:
  hermes:
    tags: [Jira, Project Management, Issues, Tickets, Bug Tracking, Productivity]
    related_skills: [linear]
---

# Jira Cloud

Manage Jira Cloud issues, projects, and comments via 4 native tools. Auth required: run `hermes auth jira` once, then the tools work without any credentials in the conversation.

## When to use this skill

The user says something like "create a Jira ticket", "what's the status of PROJ-123", "search Jira for login bugs", "transition PROJ-42 to In Progress", "add a comment to PROJ-7", "list all open issues in my project", etc.

## The 4 tools

- `jira_issue` — actions: `get`, `create`, `update`, `transitions`, `transition`
- `jira_search` — JQL search, returns compact issue list
- `jira_project` — actions: `list`, `get`
- `jira_comment` — actions: `list`, `add`

## Canonical patterns

### "Show me issue PROJ-123"
```
jira_issue({"action": "get", "issue_key": "PROJ-123"})
```
Returns: key, summary, status, issuetype, priority, assignee, labels, description (truncated at 500 chars), browse URL.

### "Search for open login bugs"
```
jira_search({"jql": "text ~ \"login\" AND status != Done ORDER BY created DESC", "max_results": 10})
```
JQL tips:
- `text ~ "word"` — full-text search across summary, description, comments
- `status = "In Progress"` — exact status name (quote multi-word names)
- `assignee = currentUser()` — current authenticated user
- `project = PROJ` — filter by project key
- `sprint in openSprints()` — issues in active sprints
- `ORDER BY created DESC` — newest first

### "Create a bug in project PROJ"
```
jira_issue({
  "action": "create",
  "project_key": "PROJ",
  "issuetype": "Bug",
  "summary": "Login fails with SSO on Safari",
  "description": "Steps to reproduce:\n\n1. Go to login page\n2. Click SSO\n3. Error appears",
  "priority": "High"
})
```
Returns: key (e.g. PROJ-47), ID, and self link. Common issue types: Bug, Task, Story, Epic, Sub-task.

### "Move PROJ-42 to In Progress"
Two steps — first check available transitions, then apply:
```
jira_issue({"action": "transitions", "issue_key": "PROJ-42"})
→ [{id: "21", name: "In Progress", to_status: "In Progress"}, ...]

jira_issue({"action": "transition", "issue_key": "PROJ-42", "transition_id": "21"})
```
Always fetch transitions first — IDs differ between projects and Jira instances.

### "Assign PROJ-10 to user X"
```
jira_issue({"action": "update", "issue_key": "PROJ-10", "assignee_id": "<accountId>"})
```
To find an account ID: use `jira_search` with `assignee = "displayName"` and look at the `assignee_account_id` field in results.

### "List all projects"
```
jira_project({"action": "list"})
```
Returns project keys, names, and types. Use the `key` (e.g., `PROJ`) for issue creation and JQL filters.

### "Get details about project MYAPP"
```
jira_project({"action": "get", "project_key": "MYAPP"})
```

### "Show comments on PROJ-5"
```
jira_comment({"action": "list", "issue_key": "PROJ-5"})
```

### "Add a comment to PROJ-5"
```
jira_comment({"action": "add", "issue_key": "PROJ-5", "body": "Investigated root cause — race condition in session refresh. Fix in PROJ-6."})
```
Plain text is accepted; the tool converts it to Atlassian Document Format automatically.

## JQL quick reference

| Goal | JQL |
|------|-----|
| My open issues | `assignee = currentUser() AND status != Done` |
| Issues in active sprint | `sprint in openSprints()` |
| High-priority bugs | `issuetype = Bug AND priority in (High, Highest)` |
| Created last 7 days | `created >= -7d` |
| Updated today | `updated >= startOfDay()` |
| Issues with label | `labels = "backend"` |
| Issues without assignee | `assignee is EMPTY` |
| Sub-tasks of PROJ-10 | `parent = PROJ-10` |

## Failure modes

**`401`** — token expired or wrong. Ask the user to run `hermes auth jira` again.

**`400` on create** — usually a missing required field (project key, issuetype, summary) or an invalid issuetype name for that project. Fetch `jira_project get` first to confirm the project exists, then try again with a common issuetype like `Task`.

**`404` on issue_key** — wrong project key or issue number. Confirm with the user.

**`transition_id` not found** — the transition name the user mentioned doesn't exist for that issue's current status. Always call `transitions` first; don't guess IDs.

**Auth not configured** — tools return an auth-required error. Instruct the user: "Run `hermes auth jira` to connect your Jira account."

## What NOT to do

- Do NOT use `curl` or `web_extract` for Jira API calls — the native tools handle auth and ADF conversion automatically.
- Do NOT guess transition IDs — always call `action: "transitions"` first.
- Do NOT pass account emails as `assignee_id` — Jira v3 requires the Atlassian `accountId` (a UUID-like string). Find it from `assignee_account_id` in search results.
- Do NOT describe every field returned by `get` — summarize key fields (status, assignee, priority) unless the user asked for full details.
