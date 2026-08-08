---
name: tally-forms
description: Tally.so API for creating and managing forms, submissions, workspaces, webhooks, and organizations via curl or Python. Build surveys, contact forms, and registration flows programmatically.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [Tally, Forms, Surveys, Productivity, API]
    homepage: https://developers.tally.so/api-reference/introduction
prerequisites:
  env_vars: [TALLY_API_KEY]
---

# Tally Forms API

Create and manage Tally.so forms, submissions, workspaces, webhooks, and organizations via their REST API. No OAuth required — just an API key.

## Prerequisites

1. Sign up at [tally.so](https://tally.so)
2. Get your API key at [tally.so/settings/api-keys](https://tally.so/settings/api-keys) (starts with `tly-`)
3. Store it in `~/.hermes/.env`:
   ```
   TALLY_API_KEY=tly-your-key-here
   ```

## API Basics

- **Base URL:** `https://api.tally.so`
- **Auth:** `Authorization: Bearer $TALLY_API_KEY`
- **Rate limit:** 100 requests/minute
- **Content-Type:** `application/json`

## How to Use This Skill

### Creating or modifying a form:

1. Read `reference/blocks.md` — understand the 44+ block types and the 3-block question pattern (QUESTION + TITLE + input block sharing a `groupUuid`)
2. Read `reference/forms.md` — POST to create, PATCH to update
3. Read `docs/guides.md` — worked examples for contact forms, dropdowns, mentions, settings, styling
4. **Key rule:** PATCH `/forms/:id` replaces the entire blocks array. Always GET the form first, append new blocks, then PATCH.

### Fetching or managing submissions:

1. Read `reference/submissions.md` — LIST with filters (date range, field values, cursor pagination), GET, DELETE
2. See `docs/guides.md` > "Fetching Form Submissions" for the response structure with `questions` and `submissions` arrays

### Setting up webhooks:

1. Read `reference/webhooks.md` — CRUD + event listing and retry
2. Event type: `FORM_RESPONSE` (fires on each new submission)
3. Endpoints must be HTTPS and respond 2xx within 30 seconds

### Managing workspaces or organizations:

1. Read `reference/workspaces.md` — workspace CRUD
2. Read `reference/organizations.md` — list/remove users, list/create/cancel invites
3. Use `GET /users/me` (`reference/users.md`) to get the current user's organization ID

## Quick Examples

### Create a form (curl):

```bash
curl -X POST 'https://api.tally.so/forms' \
  -H "Authorization: Bearer $TALLY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "status": "PUBLISHED",
    "blocks": [
      {
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "type": "FORM_TITLE",
        "groupUuid": "660e8400-e29b-41d4-a716-446655440000",
        "groupType": "TEXT",
        "payload": { "html": "My Form" }
      }
    ]
  }'
```

### List submissions (curl):

```bash
curl 'https://api.tally.so/forms/{formId}/submissions?limit=10' \
  -H "Authorization: Bearer $TALLY_API_KEY"
```

### Verify API key:

```bash
curl 'https://api.tally.so/users/me' \
  -H "Authorization: Bearer $TALLY_API_KEY"
```

## Key Concepts

- **Blocks** — building units of forms. Every form is an ordered array of blocks.
- **Question pattern** — 3 blocks with shared `groupUuid`: QUESTION (container with `isRequired`), TITLE (label HTML), input block (INPUT_TEXT, DROPDOWN, RATING, etc.)
- **Choice nesting** — DROPDOWN_OPTION, MULTIPLE_CHOICE_OPTION, CHECKBOX use the container's `uuid` as their `groupUuid`
- **Mentions** — dynamic field refs in HTML: `<span class="mention" data-uuid="...">@field</span>` with a `mentions` array
- **Form statuses** — BLANK, DRAFT, PUBLISHED, DELETED
- **Hidden fields** — populated via URL query params (`?name=John`)

## Scripts

Standalone Python scripts in `scripts/` (requires `pip install requests`):

```bash
python scripts/users.py                    # Verify API key
python scripts/forms.py                    # List, create, update, delete forms
python scripts/submissions.py <form_id>    # List submissions
python scripts/workspaces.py               # List workspaces
python scripts/webhooks.py <form_id>       # Manage webhooks
python scripts/organizations.py <org_id>   # Org users and invites
```

## File Reference

| Path | Contents |
|------|----------|
| `reference/forms.md` | Forms LIST, GET, POST, PATCH, DELETE + settings |
| `reference/submissions.md` | Submissions LIST (filters, pagination), GET, DELETE + field value types |
| `reference/workspaces.md` | Workspaces CRUD |
| `reference/organizations.md` | Org users (list, remove) and invites (list, create, cancel) |
| `reference/webhooks.md` | Webhooks CRUD + event listing, retry, payload format |
| `reference/users.md` | GET /users/me |
| `reference/blocks.md` | All 44+ block types, payload fields, nesting rules |
| `scripts/*.py` | Standalone runnable scripts per endpoint group |
| `docs/guides.md` | How-to guides from official Tally docs |

## Links

- [Tally API Docs](https://developers.tally.so/api-reference/introduction)
- [Tally Dashboard](https://tally.so/dashboard)
- [API Keys](https://tally.so/settings/api-keys)
