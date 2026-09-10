---
name: url-manager
description: Save, organize, search, and share URLs as structured cards.
license: MIT
compatibility: Requires Python 3.10+, network access to https://ai.ocean94.com
metadata:
  author: Piccolo Yao (Piccolo123), Hermes Agent
  version: "2.7.0"
  requires:
    bins:
      - python
    network: true
---

# URL Manager Skill

URL Manager lets users save any link or text note into a structured, searchable library of card-based collections. Save from any platform — YouTube, Amazon, GitHub, Twitter. Organize with categories and tags. Share curated collections via invite links, and deliver organized results to users with a single magic link.

## When to Use

| Scenario | Action |
|----------|--------|
| User says "save/bookmark/collect/remember this" | `add` the URL with a title |
| User asks "what did I save about X" | `search` by keyword |
| User wants to organize their links | `categories` + `update` with category_ids |
| User wants to share a collection | `create-shared-category` + `create-invite-link` |
| User asks to bulk-reorganize | `batch-update` (max 50 per call) |
| Done organizing — deliver to user | `agent_magic_link` → send link |

**Do not** paste saved URLs into chat. Always write them into URL Manager and deliver via magic link.

## Prerequisites

- Python 3.10+
- Network access to `https://ai.ocean94.com`
- A registered account and auth token

The helper script at `scripts/footprints.py` handles token management and all API calls. All commands use the form:

```
python {baseDir}/scripts/footprints.py <subcommand> [--json]
```

Add `--json` for machine-parseable output.

## How to Run

### First-time setup (account registration)

Before using this skill for the first time, the user needs an account. **Never auto-register without explicit consent.**

1. Tell the user: "This skill connects to ai.ocean94.com to store your saved links and data. It requires creating an account. Review the [Terms](https://ai.ocean94.com/terms.html) and [Privacy Policy](https://ai.ocean94.com/privacy.html). Shall I create an account for you?"
2. Wait for explicit "yes/ok/go ahead" from the user.
3. Run: `python {baseDir}/scripts/footprints.py agent_register`
4. The token is saved to your Hermes profile directory (`~/.hermes/profiles/<profile>/.footprints_token` with `chmod 600`). Subsequent commands use it automatically.

If the user already has an account, ask them to get their token from https://ai.ocean94.com (Profile → Agent Access → Access Token) and run `python {baseDir}/scripts/footprints.py me` to verify it works.

### Returning user

Run `python {baseDir}/scripts/footprints.py me` to confirm identity. Then use any command below.

## Quick Reference

### Core commands

| Command | Purpose |
|---------|---------|
| `add "<url>" --title "<t>" [--description "<d>"] [--content-type "<ct>"] [--category-ids <ids>] [--tags "<t1>,<t2>"]` | Save a link or text note (url can be empty) |
| `get <id>` | View full details of a footprint |
| `search <query>` | Full-text search across title, description, AI summary |
| `list [--category-id <id>] [--limit <n>] [--offset <n>]` | List footprints (limit max 100) |
| `update <id> [--title "<t>"] [--description "<d>"] [--category-ids <ids>] [--tags "<t1>,<t2>"]` | Modify a footprint |
| `batch-update '<json-array>'` | Bulk update up to 50 footprints |

### Organization

| Command | Purpose |
|---------|---------|
| `categories` | List all categories grouped by category set |
| `create-category "<name>" [--category-set-id <id>]` | Create a new category |
| `category-sets` | List all category sets (workspaces) |
| `create-category-set "<name>"` | Create a new category set |
| `tags` | List all used tags |
| `content-types` | List content types in your library |

### Sharing

| Command | Purpose |
|---------|---------|
| `create-shared-category "<name>" --mode cocreate\|subscribe` | Create a shared category |
| `create-invite-link <sc_id> [--duration-hours 24]` | Generate invite code |
| `join-shared-category <code>` | Join via invite code |
| `add-to-shared <sc_id> --collection-id <id>` | Add footprint to shared category |
| `remove-from-shared <sc_id> --collection-id <id>` | Remove from shared category |
| `copy <id> --category-ids <ids>` | Copy shared footprint to personal |

### Utilities

| Command | Purpose |
|---------|---------|
| `me` | Confirm current identity |
| `agent_magic_link` | Generate magic link for user delivery (valid 30 days, reusable) |
| `agent_register` | Create new account ⚠️ requires explicit user consent |

## Procedure

### New user workflow

```
1. Get explicit consent → agent_register
2. add "<url>" → save the user's links
3. categories → discover existing structure
4. create-category "<name>" → create organizational buckets
5. update <id> --category-ids <ids> → categorize items
6. agent_magic_link → send link: "Done! View your collection → [link]"
```

### Daily use workflow

```
1. me → confirm identity
2. categories + tags → understand current structure
3. search <query> → find specific items
4. add / update / batch-update → operate
5. agent_magic_link → deliver results
```

### Team sharing workflow

```
1. create-shared-category "Team KB" --mode cocreate
2. create-invite-link <sc_id> → share code with teammates
3. Teammates run: join-shared-category <code>
4. Everyone: add-to-shared <sc_id> --collection-id <id> → build together
```

## Pitfalls

### category_ids is REPLACEMENT, not append

When updating, `--category-ids` sets the complete list. It does NOT add to existing categories. Always fetch current state first:

```bash
python {baseDir}/scripts/footprints.py get 42   # → categories: [{id: 3}, {id: 5}]
python {baseDir}/scripts/footprints.py update 42 --category-ids 3,5,7   # keep 3,5; add 7
```

### Subscribe mode is read-only

Writing to a subscribe-mode shared category returns HTTP 403. Tell the user the owner must switch it to cocreate mode.

### NEVER call agent_register without consent

Each call creates a fresh empty account. Always check for existing token with `me` first. Always get explicit user approval before registering.

### Rate limiting

Frequent API calls trigger HTTP 429. Use `batch-update` for bulk operations. Add delays between rapid calls.

### No member management via API

Inviting or removing members from shared categories requires the web UI. You cannot do this programmatically.

### Always

- **Get explicit consent before `agent_register`** — never auto-register
- Inform the user on first use: this skill connects to ai.ocean94.com, data is stored there
- Search before listing — use `search` for targeted queries
- Discover before creating — check existing categories and tags to avoid duplicates
- Deliver with magic link — after organizing, always generate and share a link

### Confirm before

- Removing footprint-category associations (irreversible)
- Clearing tags
- Modifying cocreate shared categories (affects others)
- Removing footprints from shared categories (other members lose access)

## Verification

After setting up or making changes, verify the skill works:

```bash
# 1. Confirm identity
python {baseDir}/scripts/footprints.py me --json

# 2. Create a test footprint
python {baseDir}/scripts/footprints.py add "https://example.com" --title "Test" --json

# 3. Search for it
python {baseDir}/scripts/footprints.py search "Test" --json

# 4. Cleanup — delete via the web UI at https://ai.ocean94.com
```
