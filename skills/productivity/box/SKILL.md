---
name: box
description: Box CLI and API for content, search, and SDK apps.
version: 1.0.0
author: Community
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  env_vars: [BOX_CLIENT_ID, BOX_CLIENT_SECRET, BOX_ENTERPRISE_ID]
  commands: [box]
metadata:
  hermes:
    tags: [Box, Productivity, CLI, API, Content, Files, SDK]
    homepage: https://developer.box.com/guides/cli
---

# Box Skill

Hermes operates Box as a **service account** via Client Credentials Grant (CCG). Use the `terminal` tool to run Box CLI commands for agent workflows; read `references/sdk-development.md` when the user is building a shipped Box app.

## When to Use

- Upload, download, edit(rename, new version), move, or organize files and folders
- Search content, run metadata queries, or use Box AI
- Bulk reorganize folders or batch-tag metadata
- Create webhooks or poll events for automation
- Add Box SDK auth, API calls, or webhook handlers to application code

## Prerequisites

1. **Box CLI** — Node.js 18+ and `npm install -g @box/cli`.
2. **CCG Platform App** — Developer Console → Platform App → Client Credentials Grant (auth method is locked at creation). See `references/auth-and-setup.md`.
3. **Secrets in `~/.hermes/.env`** — `BOX_CLIENT_ID`, `BOX_CLIENT_SECRET`, `BOX_ENTERPRISE_ID`.
4. **CLI environment** — copy `templates/ccg-config.json.example`, fill values, then:
   ```bash
   box configure:environments:add /path/to/ccg-config.json --ccg-auth --name hermes --set-as-current
   ```

Free developer accounts auto-authorize CCG apps. Enterprise plans may require admin approval.

5. **Folder access** — the service account only sees its own empty root until you share content with it. After the app is **authorized**, invite the service account to every folder Hermes should use (see Step 0 below).

## How to Run

### Step 0 — Auth and content gate

Run via `terminal`:

```bash
box users:get me --json --fields id,name,login
```

Record the service account **id**, **name**, and **login** (`login` is the service account email). If auth fails, walk through `references/auth-and-setup.md` before any other Box work.

#### Grant Hermes access to existing folders

The CCG service account is a separate Box user. It **cannot** see files in your personal or team folders until you collaborate it in.

**Find the service account email** (pick one):

| Source | Where |
| --- | --- |
| Developer Console | Open your app → **General Settings** tab → **Service Account** section (shown after the app is authorized). Email looks like `AutomationUser_<app-id>_…@boxdevedition.com`. |
| Box CLI | `login` field from `box users:get me` above |

**Invite the service account** (tell the user to do this in the Box web app — not via Hermes chat secrets):

1. In [Box](https://app.box.com), open the folder Hermes should access (or the parent folder).
2. Click **Invite People** (or **Share** → **Invite Collaborators**).
3. Paste the **service account email** from the table above.
4. Choose a role: **Viewer** (read/search), **Editor** (upload/move/edit), or **Co-owner** (manage collaborators — only if needed).
5. Repeat for each top-level folder Hermes needs; subfolders inherit access.

Alternatively, an Editor on the folder can add the collaborator via CLI:

```bash
box collaborations:create <FOLDER_ID> <SERVICE_ACCOUNT_EMAIL> editor --json
```

**Before any search, move, or download on user-owned content:** confirm the target folder is shared with the service account. If operations return 404 or empty search, ask the user to complete the invite step first.

Confirm where target content lives:

- **Shared folder** — user invited the service account (most common for existing content)
- **Hermes workspace** — content under the service account root (`folder id 0`) — upload/create here when Hermes owns the files
- **User impersonation (advanced)** — `--ccg-user` (see `references/auth-and-setup.md`)

### Route the request

| User need | Read first |
| --- | --- |
| Agent task (search, move, upload, AI) | This skill + domain reference below |
| Application code (SDK, OAuth in app) | `references/sdk-development.md` |
| Setup, CCG, actors, workspace | `references/auth-and-setup.md` |
| CLI patterns, `box request`, `--fields` | `references/cli-guide.md` |
| Files, folders, links, collaborations, edit/version | `references/content-workflows.md` |
| Search, metadata-query, Box AI | `references/search-and-ai.md` |
| Batch moves, folder trees | `references/bulk-operations.md` |
| Webhooks, events | `references/webhooks-and-events.md` |
| CLI missing / no command | `references/rest-api.md` |
| 401, 403, 404, 409, 429 | `references/troubleshooting.md` |

### Tool ladder

1. **Box CLI** via `terminal` — default for all agent operations
2. **`box request`** — escape hatch for any REST endpoint when no dedicated CLI subcommand exists
3. **Direct REST** via `terminal` — only when CLI is unavailable and user confirms; see `references/rest-api.md`

Run **one** `box` command at a time. Parallel CLI calls break auth.

## Quick Reference

Folder `0` is the service account root. Always append `--json`; add `--fields id,name,...` to limit output.

```bash
box folders:get 0 --json --fields id,name,item_collection
box folders:items <FOLDER_ID> --json --max-items 50 --fields id,name,type
box search "invoice" --json --limit 10
box files:upload ./file.pdf --parent-id <FOLDER_ID> --json
box files:update <FILE_ID> --name "renamed.pdf" --json
box files:versions:upload <FILE_ID> ./updated.pdf --json
box files:upload ./updated.pdf --parent-id <FOLDER_ID> --overwrite --json
box folders:create <PARENT_ID> "Hermes-Inbox" --json
box shared-links:create <FILE_ID> file --access company --json
box request GET /files/<ID> --json
```

## Procedure

1. Run Step 0 auth gate; confirm content access model (workspace vs shared folder).
2. Pick the domain reference from the route table; read it before acting.
3. Resolve folder/file IDs — prefer IDs over paths; persist IDs from responses.
4. Execute with `--json --fields`; paginate large lists with `--max-items` / offset.
5. For bulk work: inventory → plan → execute serially → verify counts (`references/bulk-operations.md`).
6. Summarize results with actor context, Box IDs, and the verification command used.

## Pitfalls

- **Serial CLI only** — never run concurrent `box` processes against the same environment.
- **No folder invite** — Hermes returns 404 or empty search for content you can see in the Box UI; invite the service account email from **General Settings** (Developer Console) or `box users:get me` → `login`.
- **Wrong actor** — many 404s are permission/actor mismatches, not missing objects.
- **No secrets in chat** — guide users to `~/.hermes/.env`; never echo client secrets or tokens.
- **Box AI pacing** — space AI calls 1–2 seconds apart; prefer AI over downloading file bodies when possible.
- **Editor role required** — rename, upload new versions, and move need **Editor** (or higher) collaboration on the folder, not Viewer.
- **Office/Google in-browser edit** — `.docx` / Google Docs editing in Box Preview (Open With) is a **human UI** flow; Hermes uses download → local edit → upload new version for file content changes.

## Verification

After setup or any write operation:

```bash
box users:get me --json --fields id,name
box folders:items 0 --json --max-items 5 --fields id,name,type
```

For writes: create a smoke folder, confirm it in the parent listing, then delete if disposable.
