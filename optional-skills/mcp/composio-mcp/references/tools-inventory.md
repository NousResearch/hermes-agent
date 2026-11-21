# Composio Tool Inventory

Inventory of available toolkits and tools. Data from `composio dev toolkits list` (live snapshot).

## Overview

| Metric | Count |
|--------|-------|
| Toolkits | 30 |
| Total tools | ~3325 |
| Total triggers | ~172 |

## Toolkits by category

### Communication & Email
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Gmail | `gmail` | 61 | 2 | OAuth |
| Slack | `slack` | 158 | 9 | OAuth |
| Slack Bot | `slackbot` | 98 | 9 | OAuth |
| Discord | `discord` | 23 | 1 | OAuth |
| Outlook | `outlook` | 286 | 5 | OAuth |

### Development & Code
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| GitHub | `github` | 871 | 46 | OAuth |
| Bitbucket | `bitbucket` | 108 | 0 | OAuth |
| Jira | `jira` | 97 | 17 | OAuth |
| Linear | `linear` | 46 | 12 | OAuth |

### Productivity & Docs
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Google Calendar | `googlecalendar` | 45 | 7 | OAuth |
| Google Sheets | `googlesheets` | 45 | 16 | OAuth |
| Google Docs | `googledocs` | 41 | 10 | OAuth |
| Google Tasks | `googletasks` | 16 | 5 | OAuth |
| Notion | `notion` | 53 | 8 | OAuth |
| Airtable | `airtable` | 24 | 6 | OAuth |

### Search & AI
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Perplexity AI | `perplexityai` | 9 | 0 | API key |
| SerpAPI | `serpapi` | 48 | 0 | API key |
| Tavily | `tavily` | 5 | 0 | API key |
| Code Interpreter | `codeinterpreter` | 5 | 0 | — |
| Composio Search | `composio_search` | 21 | 0 | — |

### Social & Content
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Twitter/X | `twitter` | 79 | 0 | OAuth |
| YouTube | `youtube` | 48 | 4 | OAuth |
| Reddit | `reddit` | 21 | 0 | OAuth |

### Data & Infrastructure
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Google Drive | `googledrive` | 77 | 7 | OAuth |
| Supabase | `supabase` | 124 | 0 | API key |
| Canvas (LMS) | `canvas` | 573 | 6 | OAuth |
| Figma | `figma` | 52 | 0 | OAuth |
| Firecrawl | `firecrawl` | 29 | 0 | API key |
| HubSpot | `hubspot` | 244 | 2 | OAuth |

### Internal
| Toolkit | Slug | Tools | Triggers | Auth |
|---------|------|-------|----------|------|
| Composio | `composio` | 18 | 0 | — |

## Tool selection strategy

1. **Search by intent** — `composio search "describe the task"` is the primary discovery mechanism. It uses semantic search across all toolkits.
2. **Narrow by toolkit** — when you know the app, filter with `--toolkits <slug>`.
3. **Inspect before executing** — `composio execute <slug> --get-schema` shows required inputs.
4. **Dry-run** — `composio execute <slug> --dry-run -d '{...}'` validates without side effects.

## Common tool patterns

### Email (Gmail)
- `GMAIL_SEND_EMAIL` — send email
- `GMAIL_FETCH_EMAILS` — list/read emails
- `GMAIL_CREATE_EMAIL_DRAFT` — create draft
- `GMAIL_REPLY_TO_THREAD` — reply in thread
- `GMAIL_CREATE_LABEL` / `GMAIL_ADD_LABEL_TO_EMAIL` — label management
- `GMAIL_CREATE_FILTER` — auto-filter rules

### Code (GitHub)
- `GITHUB_CREATE_ISSUE` / `GITHUB_LIST_REPOSITORY_ISSUES` — issues
- `GITHUB_CREATE_A_PULL_REQUEST` / `GITHUB_MERGE_A_PULL_REQUEST` — PRs
- `GITHUB_GET_THE_AUTHENTICATED_USER` — user info
- `GITHUB_LIST_ORGANIZATION_REPOSITORIES` — repo listing
- `GITHUB_CREATE_A_COMMENT_ON_AN_ISSUE` — comments

### Messaging (Slack)
- `SLACK_SEND_A_MESSAGE_TO_A_SLACK_CHANNEL` — post message
- `SLACK_CREATE_A_CHANNEL` — create channel
- `SLACK_INVITE_USERS_TO_A_SLACK_CHANNEL` — invite users
- `SLACK_RETRIEVE_DETAILED_CHANNEL_INFORMATION` — channel info

### Calendar (Google Calendar)
- `GOOGLECALENDAR_CREATE_EVENT` — create event
- `GOOGLECALENDAR_LIST_EVENTS` — list events
- `GOOGLECALENDAR_UPDATE_EVENT` — update event
- `GOOGLECALENDAR_DELETE_EVENT` — delete event

### Docs (Google Docs/Sheets)
- `GOOGLEDOCS_CREATE_DOCUMENT` — create doc
- `GOOGLESHEETS_BATCH_UPDATE` — batch cell updates
- `GOOGLESHEETS_CREATE_SPREADSHEET` — create sheet

## Tag taxonomy

Tools are tagged for filtering. Common tags:

| Tag | Meaning |
|-----|---------|
| `important` | Core/high-value tools |
| `destructiveHint` | Irreversible or data-destroying |
| `idempotentHint` | Safe to retry |
| `createHint` | Creates a resource |
| `updateHint` | Modifies a resource |
| `deleteHint` | Deletes a resource |
| `readOnlyHint` | No side effects |
| `openWorldHint` | Results depend on external state |
| `batch` | Bulk operation |

## Querying tools

```bash
# Search across all toolkits
composio search "manage my calendar"

# List tools in a specific toolkit
composio tools list gmail --limit 20

# Get tool schema
composio tools info GMAIL_SEND_EMAIL

# List triggers in a toolkit
composio triggers list github
```
