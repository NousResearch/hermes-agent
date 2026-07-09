---
name: tencent-docs
description: Read and write Tencent Docs through the deployed multi-user Tencent Docs tools on this Hermes host.
version: 1.0.0
metadata:
  hermes:
    tags: [tencent-docs, docs, spreadsheet, dingtalk, multi-user]
    related_skills: [mcporter, productivity]
---

# Tencent Docs Multi-User Skill

Use this skill when the user wants to inspect, read, or operate Tencent Docs from a DingTalk-routed Hermes conversation.

## Runtime

This Hermes host already has the Tencent Docs tool deployed at:

- `/root/tencent_docs_tools`

It is configured for DingTalk multi-user mode:

- identify the caller primarily by `dingtalk_username`
- treat `dingtalk_user_id`, `sender_staff_id`, and `staff_id` as auxiliary context when available
- resolve that user's own Tencent Docs MCP token from the token store
- avoid sharing one Tencent Docs identity across multiple DingTalk users

## When To Use

Use this skill when the user asks to:

- read a Tencent Docs document, sheet, or file
- inspect a Tencent Docs file's title, type, owner, URL, or metadata
- verify whether they have Tencent Docs access
- operate Tencent Docs from a DingTalk conversation routed through Hermes

Do not use this skill for Alibaba DingTalk spreadsheet APIs unless the request is specifically about DingTalk spreadsheet APIs rather than Tencent Docs.

## Routing Priority

When the request comes from a DingTalk-routed Hermes conversation and the goal is to access Tencent Docs with a per-user binding, prefer this `tencent-docs` skill over other Tencent-related skills.

Do not switch to `tencent-docs-api` or `tencent-doc-reader` for the default multi-user Hermes path unless the user explicitly asks for:

- Tencent Docs OpenAPI OAuth app setup
- `exchange-code`
- browser-style share-link diagnosis without the Hermes multi-user token flow

For normal Hermes multi-user Tencent Docs access, stay on this skill.

## Authentication Mode

Default authentication mode for this skill is:

- DingTalk username + that user's own Tencent Docs MCP token

This skill should normally use:

- `bind-dingtalk-token`
- `show-dingtalk-binding`
- `hermes-inspect-access`
- `hermes-self-check`
- `mcp-tools`
- `mcp-call`

Do not ask for or require these values unless the user explicitly wants the OAuth code-exchange flow:

- `TENCENT_DOCS_CLIENT_ID`
- `TENCENT_DOCS_CLIENT_SECRET`
- `TENCENT_DOCS_REDIRECT_URI`

Do not recommend `exchange-code` for routine DingTalk multi-user Tencent Docs access.

## Runtime Paths

- tool root: `/root/tencent_docs_tools`
- environment file: `/root/.hermes/.env`
- DingTalk directory: `/root/tencent_docs_tools/.dingtalk_users.json`
- token store: `/root/tencent_docs_tools/.tencent_docs_tokens.json`

Preferred MCP transport:

- `TENCENT_DOCS_MCP_TRANSPORT=curl`

Directory enforcement:

- local/macOS development: usually keep `TENCENT_DOCS_ENFORCE_DINGTALK_DIRECTORY=true`
- Hermes production: usually set `TENCENT_DOCS_ENFORCE_DINGTALK_DIRECTORY=false`

In Hermes production, the DingTalk allowlist is typically enforced by Hermes itself, so this skill should mainly rely on the per-user Tencent Docs token binding.

## Core Commands

Always run from:

```bash
cd /root/tencent_docs_tools
```

Inspect one DingTalk user's Tencent Docs access:

```bash
python3 main.py hermes-inspect-access \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --dingtalk-username "<dingtalk_username>"
```

Run an end-to-end self-check for one DingTalk user:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py hermes-self-check \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --dingtalk-username "<dingtalk_username>" \
  --file-id "<file_id>"
```

List Tencent Docs MCP tools as one DingTalk user:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py mcp-tools \
  --dingtalk-user-id "<dingtalk_user_id>"
```

Call one Tencent Docs MCP tool as one DingTalk user:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py mcp-call \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --tool "manage.query_file_info" \
  --arg file_id=<file_id>
```

Bind one DingTalk user to that user's own Tencent Docs MCP token:

```bash
python3 main.py bind-dingtalk-token \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --dingtalk-username "<dingtalk_username>" \
  --mcp-token "<mcp_token>"
```

Diagnose whether one online sheet should be read through MCP or exported for offline analysis:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py sheet-read-strategy \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --file-id "<file_id>" \
  --sheet-id "<sheet_id>"
```

Automatically read one bounded sheet block with strategy + fallback built in:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py sheet-read-auto \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --file-id "<file_id>" \
  --sheet-id "<sheet_id>" \
  --start-row 0 \
  --start-col 0 \
  --row-count 10 \
  --col-count 8
```

`sheet-read-auto` now returns:

- `available_sheets`: full sheet/tab list for the current file
- `selected_sheet`: the sheet actually used for reading
- `selection_reason`: why that sheet was selected, especially when no `sheet_id` was provided
- `strategy`: read-strategy probe result
- `read`: normalized rows and summary

Read one bounded sheet block through `sheet.operation_sheet` when `sheet.get_cell_data` behaves abnormally:

```bash
TENCENT_DOCS_MCP_TRANSPORT=curl \
python3 main.py sheet-read-js \
  --dingtalk-user-id "<dingtalk_user_id>" \
  --file-id "<file_id>" \
  --sheet-id "<sheet_id>" \
  --start-row 0 \
  --start-col 0 \
  --row-count 10 \
  --col-count 8
```

## DingTalk Identity Rule

When the current conversation comes from DingTalk, always prefer the DingTalk identity from the session or message context.

Use:

- DingTalk display name as `dingtalk_username`
- Use that username as the primary key for Tencent Docs token binding lookup
- Treat `dingtalk_user_id`, `sender_staff_id`, or `staff_id` as helpful metadata when Hermes exposes them

Do not use a shared Tencent Docs token when a DingTalk user-specific binding is expected.

## Hard Rules

These rules override examples and default habits.

1. Always use the current DingTalk caller identity from the runtime payload.
- Prefer the current payload's `dingtalk_username` for token binding lookup.
- Treat `sender_staff_id`, `staff_id`, and `dingtalk_user_id` as supporting metadata only.
- Do not reuse a username from earlier examples, older turns, README snippets, or memory.
- Do not assume `黄须阔` or any other username unless the current payload explicitly says so.

2. For normal sheet-reading requests, use `sheet-read-auto` first.
- Do not manually compose `sheet.get_sheet_info` + `sheet.get_cell_data` + `sheet.operation_sheet` unless you are diagnosing a failure.
- Do not switch to browser or export flows as the first choice when `sheet-read-auto` is available.

3. Never guess `sheet_id` from the URL.
- The URL path value such as `DRUp5RUx1SnRIa01L` is the file ID, not the sheet ID.
- Let `sheet-read-auto` or `sheet-read-strategy` resolve the correct worksheet tab.
- Do not claim a value is an invalid `sheet_id` before calling the actual tool that resolves sheets.

4. If the current caller identity is unclear, stop and say that the runtime did not provide a reliable DingTalk username.
- Do not silently fall back to another known colleague's binding.
- Do not continue with a guessed user.

## Recommended Workflow

1. Resolve the current caller identity from the current DingTalk runtime payload only, using `dingtalk_username` as the binding lookup key.
2. If access is uncertain, first run `hermes-inspect-access` with that current caller identity.
3. If the user asks whether a specific document is reachable, run `hermes-self-check` or `mcp-call` with `manage.query_file_info`.
4. For normal table-reading requests, run `sheet-read-auto` first.
5. If the user is missing a token binding, explain that their DingTalk user has not yet bound a Tencent Docs MCP token.
6. If Hermes production has `TENCENT_DOCS_ENFORCE_DINGTALK_DIRECTORY=false`, treat Hermes' own DingTalk allowlist as the source of truth for who is allowed to reach this skill.
7. When `sheet-read-auto` returns, first inspect `available_sheets`, `selected_sheet`, and `selection_reason` before interpreting the actual rows.
8. If no `sheet_id` was provided, explicitly tell the user which sheet/tab was selected automatically.
9. If the user asks to interpret a larger online sheet and results look suspicious, inspect `sheet-read-auto` output first, then use `sheet-read-strategy` for diagnosis.
10. If `sheet-read-strategy` returns `prefer_small_block_reads_or_export` or `prefer_export_or_offline_analysis`, prefer export/download plus local analysis over repeated online MCP probing.
11. If you need low-level manual control, switch to `sheet-read-js` and read the sheet in bounded windows such as `10x8` or `20x8`.

## Command Style

Prefer simple single-purpose shell commands.

Use:

- one `python3 main.py ...` command at a time
- direct `mcp-call` or `hermes-self-check` invocations
- `sheet-read-auto` for default sheet interpretation
- `sheet-read-strategy` when you need to diagnose why a sheet is hard to read
- `sheet-read-js` as the fallback for bounded block reads
- the current runtime caller's DingTalk username only

Avoid:

- `python3 -c`
- pipes such as `|`
- shell chaining such as `&&`
- redirections such as `2>&1`
- inline output post-processing in shell
- repeated exploratory `sheet.get_cell_data` retries across many guessed ranges
- single-cell or tiny-range loops to reconstruct a large sheet online
- reusing a DingTalk username from a previous example or previous user
- guessing `sheet_id` from the URL
- saying "I saw user X's DingTalk ID is ..." or "I saw user X's username is ..." unless that exact value exists in the current runtime payload
- browser/export fallback before trying `sheet-read-auto`

If output needs interpretation, read the command result and explain it in the response instead of building a more complex shell pipeline.

## Sheet Response Order

When explaining a `sheet-read-auto` result, prefer this order:

1. State the file title.
2. List the available sheets/tabs briefly from `available_sheets`.
3. State the chosen sheet from `selected_sheet`.
4. If the selection was automatic, quote or paraphrase `selection_reason`.
5. Then explain the header, sample rows, and what the table appears to be about.

Do not jump straight into row interpretation without first telling the user which sheet/tab is being read.

## Export Limitations (Pitfalls)

### COS Download URL Requires Browser Auth

`export_file` returns COS temporary download URLs that expire quickly and require browser-level authentication. Server-side tools (curl, wget) will receive an error page, not the actual file. The exported file is only reliably usable when:

- The user opens the COS URL in their own browser, or
- You pivot to sending an HTML email with the document's online links instead of attachments

**Workaround**: When the goal is to share document contents via email, generate an HTML link list instead of trying to attach exported files.

### Flowchart/Flowsheet Docs Cannot Be Exported as XLSX

`export_file` only supports spreadsheet formats. Documents of type `Flowchart`, `Flowsheet`, or similar non-spreadsheet types will fail with an unsupported-format error. Check `query_file_info` to confirm document type before attempting export.

## Safety Rules

- Never print full Tencent Docs tokens back to the user.
- Prefer `show-dingtalk-binding` or `hermes-inspect-access` output, which masks tokens.
- Treat `.tencent_docs_tokens.json` as secret operational data.
- If a user is not in the DingTalk allowlist or their binding is `DISABLED`, do not attempt Tencent Docs calls for them.

## Response Guidance

When reporting Tencent Docs results, summarize the useful fields clearly:

- title
- type
- URL
- owner
- permission or access status when relevant

If a command fails, explain whether the issue is one of:

- user not allowed
- token not bound
- binding disabled
- document not accessible with that user's Tencent Docs account
- transient network or MCP transport error
