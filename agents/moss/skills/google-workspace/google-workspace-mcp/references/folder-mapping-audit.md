# Reference: Multi-Inbox Folder Mapping Audit

Produced during mailbox-cleaner Phase 2, April 2026.

## Context

Before a cron-based email cleaner creates any new kensei/ labels (Gmail) or KENSEI/ folders (Outlook), it should ALWAYS run a folder-mapping audit. This prevents creating duplicates alongside existing user folders.

## Gmail Label Audit

### Tools
- `mcp_google_workspace_list_gmail_labels(user_google_email=...)`
- Sort into system vs user labels. User labels are the ones that matter.

### What to look for
- Existing labels that serve a category in the spec (e.g. Receipts, Job-Apps)
- IMAP artifact labels like [Imap]/Trash (Thunderbird/Apple Mail residue)
- Gmail's CATEGORY_* labels (CATEGORY_PROMOTIONS, CATEGORY_SOCIAL, CATEGORY_UPDATES). These are auto-tabs, not user organisation. They coexist with kensei/ labels but do not replace them.
- Empty labels that might be stale or accidentally created

### Decision matrix
| Situation | Action |
|---|---|
| Exact name match for proposed category | USE AS-IS, do not create kensei/ duplicate |
| Similar name, unclear scope | PARTIAL OVERLAP — ask user |
| Empty label, unused | MIGRATE CANDIDATE — ask to delete or repurpose |
| No match | Create new kensei/<Category> |

## Outlook Folder Audit

### Tools
- `mcp_outlook_list_mail_folders(account=...)` — root folders
- `mcp_outlook_list_mail_child_folders(account=..., mailFolderId=...)` — children
- Count items per folder from totalItemCount / unreadItemCount

### What to look for
- displayName of folders
- Page past first 10: skip=10 if `@odata.nextLink` present
- Total item counts — empty folders are candidates for migration/deletion

### Critical check for job-hunt inboxes
The spec requires a Job Applications folder in sahil_ss@outlook.com. This is known to exist. Any future build must confirm it still exists before proceeding.

### Decision matrix
Same as Gmail. Key principle: Respect existing organisation. If Sahil already curates a folder, use it. Do not create a KENSEI/ parallel.

## Multi-Account Workflow

When auditing N inboxes in parallel:
1. Fire all list_gmail_labels / list_mail_folders calls in one batch (these are lightweight reads)
2. Collect results, normalise folder/label names
3. Compare each inbox individually against the spec's proposed categories
4. Produce a per-inbox mapping decision table
5. Flag ANY question for user before creating folders

## Sample output format

| Inbox | Existing overlaps | New folders needed | Questions / notes |
|---|---|---|---|
| saghir.sahil@gmail.com | None | 8 new kensei/ labels | None |
| sahil_ss@outlook.com | Job Applications (USE AS-IS) | 3 KENSEI/ | None |

## Common pitfalls

1. Gmail labels scope vs messages scope — tokens for gmail.readonly work for search_gmail_messages but list_gmail_labels needs gmail.labels scope. Re-auth if needed.
2. Outlook pagination — list_mail_folders returns 10 by default. Use skip=10 for the next page. Folders beyond root 10 will be silently missed otherwise.
3. Unicode folder names — Outlook users sometimes have non-ASCII duplicates. Treat as equivalents where they serve the same purpose.
4. Empty user folders — A folder with 0 items might be abandoned. Ask before deleting or repurposing.

## Related
- add-gmail-account — if an account needs re-authorisation
- gmail-inbox-audit — for content analysis after folder mapping
