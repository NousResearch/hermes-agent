---
triggers: ["act on another draas.com account", "domain-wide delegation", "read another employee mailbox", "admin access to someone else workspace account"]
name: gws-domain-admin
version: 0.2.0
description: Domain-wide-delegation ops on any Workspace account.
metadata:
  hermes:
    category: productivity
    tags: [gws, google, workspace, dwd, gmail, drive]
---

# GWS Domain Admin (domain-wide delegation)

> **GATE (NDR, 2026-09-21): use this skill ONLY when the user has explicitly asked, in this conversation, to read or act on ANOTHER @draas.com person's account by name.** For the user's own account — including NDR's own `ndr@draas.com` — use the own-token `gws_*` tools / `build_service(service_name=...)`, never DWD. DWD is never a fallback for an own-account token or scope error, and never a shortcut to a shared file. Do not import `tools.gws_dwd_tools` from scripts; call the native `gws_dwd_*` tools.


Act on ANY Google Workspace account in the DWD-allowed domain (`@draas.com`)
as the domain admin via the system service-account key — including accounts
that have NO vault identity (e.g. ex-employees like piyush@draas.com /
bhagya@draas.com). Use the native `gws_dwd_*` tools; they run in the trusted
process, resolve the acting admin from the session, and return DATA ONLY.

## When to Use

- "check <name>'s account" / "look at <name>'s mail/drive/contacts"
- "search <name>'s mail" / "what's in <name>'s inbox"
- "prepare a draft email as <name>" (never a real send)
- "list <name>'s drive files" / "transfer ownership of <file> to <name>"
- anything referencing acting on another person's Google account
- any request about an ex-employee's old mail/docs (use their exact
  `@draas.com` email as the target)

## Safety (non-negotiable)

1. **Never expose the credential.** The `gws_dwd_*` tools fetch the DWD
   service-account key into the trusted process, use it, and never return it.
   Never `print()`, log, or paste `token_json`, the SA `private_key`, or any
   impersonated access token. Never call `get_gws_credentials` yourself and
   never read the vault token files.
2. **Admin-only.** The vault enforces this (session must be
   `role==admin` or `permissions.vault_admin`). Never attempt to bypass it.
3. **Never send.** Sending is permanently blocked for the whole stack.
   "Send" always means create a draft (`gws_dwd_gmail_draft_create`,
   `gws_dwd_gmail_reply_draft`). `gws_dwd_gmail_send` does NOT exist.

## Procedure

1. **Resolve the target** with `gws_dwd_resolve`:
   - pass a name (`"piyush"`) or a full email (`"piyush@draas.com"`).
   - If `ambiguous`, ask the admin to be more specific (full email).
   - For ex-employees with no vault identity, use their exact email — the
     resolve returns `direct: true`.
2. **Pick the operation** — the native `gws_dwd_*` tools:

   | Ask | Tool |
   |---|---|
   | search mail | `gws_dwd_gmail_search` (query, max_results) |
   | read a message | `gws_dwd_gmail_get` (message_id) |
   | a conversation | `gws_dwd_gmail_thread_get` (thread_id) |
   | labels | `gws_dwd_gmail_list_labels` / `gws_dwd_gmail_labels_modify` |
   | compose (draft only) | `gws_dwd_gmail_draft_create` (to/subject/body) |
   | reply (draft only) | `gws_dwd_gmail_reply_draft` (message_id, body) |
   | drafts | `gws_dwd_gmail_draft_list` / `gws_dwd_gmail_draft_delete` |
   | trash / restore | `gws_dwd_gmail_trash` / `gws_dwd_gmail_untrash` |
   | list drive files | `gws_dwd_drive_list` (query/folder) |
   | file metadata | `gws_dwd_drive_get` (file_id) |
   | create folder/file | `gws_dwd_drive_create_folder` / `gws_dwd_drive_create_file` |
   | move | `gws_dwd_drive_move` (add/remove parents) |
   | rename/update | `gws_dwd_drive_update` |
   | delete | `gws_dwd_drive_delete` |
   | ownership | `gws_dwd_drive_transfer_ownership` |
   | sharing | `gws_dwd_drive_permissions_list` / `_add` / `_remove` |

   Each tool takes `target` (the resolved email/name) plus the operation
   params. The tool returns `{target, mode, operation, result}` — data only.

3. **Report back** concisely: who was acted on, the mode (`dwd` vs `user`),
   and the outcome. Never include credential material.

## Troubleshooting

- "No identity matches target" → the name/email can't be resolved; use the
  exact `@draas.com` email, or `gws_dwd_resolve` to see why.
- "Unauthorized: admin only" → the acting session is not the admin; do not
  bypass this.
- `needs_auth` → the target has no personal token; for DWD targets this
  should not occur (the service account impersonates directly).