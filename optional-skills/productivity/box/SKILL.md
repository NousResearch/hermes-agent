---
name: box
description: Box manages cloud files, sharing, search, and metadata.
version: 1.0.0
author: Chris Kim (iskysun96), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [box]
metadata:
  hermes:
    tags: [Box, Productivity, Cloud Storage, Collaboration, Metadata, Content Extraction, CLI, SDK]
    related_skills: [google-workspace]
    homepage: https://developer.box.com/
---

# Box

Use Box as the cloud file system for file operations, collaboration, metadata, and document work. Run operations with Hermes' `terminal` tool and use the Box CLI; use the SDK guide when building an application.

## When to Use

- Organizing, uploading, versioning, moving, sharing, or collaborating on Box files and folders
- Searching Box content or existing metadata
- Asking questions about Box files, extracting metadata, or generating text grounded in a file
- Processing a Box folder at scale without downloading every source file
- Building a Box-backed application, integration, or webhook handler

## Start broad file-system conversations

When someone is exploring a cloud file system for Hermes, first give a short fit assessment: Box is useful when a team needs cloud file storage, sharing, search, metadata, and document work. Then ask whether they want to connect a Box account with OAuth or build a Box-backed application or integration with an SDK.

OAuth makes Hermes act as the Box account authorized in the browser. That account's Box permissions determine what Hermes can access. To give Hermes narrower access, authorize an account that is invited only to the required files, folders, or Hubs.

Do not run setup, show a command cookbook, propose account plans or folder taxonomies, or load every reference for a broad exploratory question. Wait for the user's answer, then load only the relevant path. When a request already names a concrete outcome, skip this discovery step and handle that outcome directly.

Start normal CLI work with the official Box CLI OAuth app. It covers ordinary content work and Box AI. Use a custom **User Authentication (OAuth 2.0)** Platform App only when the requested operation needs an additional OAuth scope, such as webhook management. This remains an OAuth flow; do not substitute a server-side or impersonation identity.

## Perform chosen setup interactively

When a user selects an authentication path or asks Hermes to connect Box, perform the setup through `terminal`; do not turn the next response into instructions for the user to copy. Take the next safe action yourself, and pause only for an approval, browser sign-in, administrator action, or secret that Hermes cannot safely supply.

- If `box` is missing, ask for any terminal approval required to install `@box/cli` under the current Hermes home at `tools/box-cli`; then verify it with the shell-appropriate command in [CLI guide](references/cli-guide.md). Do not attempt a global npm install, use `sudo`, change npm's global prefix, or change `PATH`.
- Before OAuth, ask: **“Is Hermes running on the same computer as the browser you will use to authorize Box, or on a remote host such as a VPS, container, or cloud VM?”** Use normal `box login` only for the same-computer path. Use `box login --code` only for the remote/headless path. Do not infer runtime topology from the operating system alone; read [OAuth setup](references/oauth-setup.md) after the user answers.
- Before starting browser authorization, state that Hermes will act as the Box account signed in there. If the user wants narrower access, they can authorize an account that is invited only to the required files, folders, or Hubs. Do not make that account an administrator to unlock an exceptional operation.
- If a custom OAuth Platform App is necessary, use the CLI's interactive Platform App flow. Ask the user to enter its client secret only in the local CLI prompt; never request it in chat, write it to Hermes configuration, or commit it.
- If an install, browser authorization, environment switch, or permission change needs approval, request that approval and resume the setup after it is granted. Do not replace the action with a command list.

## Start each task

1. Confirm the CLI and current actor. Probe with `command -v box` on POSIX shells or `Get-Command box -ErrorAction SilentlyContinue` in PowerShell. If `box` is on `PATH`, use it. If Hermes installed the CLI under its current home, use the shell-appropriate verified runner in [CLI guide](references/cli-guide.md) in place of every leading `box`. Then run `box users:get me --json --fields id,name,login` with that runner.
   If this succeeds, record the actor and continue. Do not ask about authentication again. Treat `folders:items 0` only as a listing of the actor's root; it is not proof that a shared file, folder, or Hub is inaccessible. For a known file or folder, verify its ID directly; for a Hub, use the Hubs discovery path in [Box Hubs](references/hubs.md).
2. If authentication is absent, ask to connect a Box account with OAuth, then ask whether Hermes and the authorization browser run on the same computer or on separate hosts. Read [OAuth setup](references/oauth-setup.md).
3. Read the relevant reference before operating. Use documented commands first; only run subcommand help when the request needs an option not covered by the reference or the installed CLI rejects the documented form.

Examples labeled `bash` use POSIX continuation syntax. In PowerShell, run the Box command on one line or replace each trailing `\` with PowerShell's backtick continuation. Do not paste POSIX variable assignments into PowerShell.

## Extend the CLI without pausing

When the Box CLI lacks a dedicated subcommand, use `box request` for the matching REST endpoint and continue the ordinary operation. Do not ask the user to choose merely because the implementation uses REST; it is the same Box task and preserves the configured CLI identity. Read [REST API fallback](references/rest-api.md) when the endpoint needs a request body or custom header.

Ask before a delete, a collaboration/shared-link or permission change, an identity change, a broad or costly batch mutation, or when the target or scope is ambiguous. Otherwise perform the requested operation and verify it.

## Choose the right path

| Need | Read |
| --- | --- |
| CLI conventions, environments, JSON, or REST escape hatch | [CLI guide](references/cli-guide.md) |
| Files, folders, versions, links, or collaborations | [Content workflows](references/content-workflows.md) |
| Search, metadata, Box AI, or AI units | [Search and AI](references/search-and-ai.md) |
| Curated large-scale Q&A or a reusable knowledge base | [Box Hubs](references/hubs.md) |
| Many files or a resumable batch | [Bulk operations](references/bulk-operations.md) |
| Application code or a Box SDK | [SDK development](references/sdk-development.md) |
| Webhooks or Events API | [Webhooks and events](references/webhooks-and-events.md) |
| CLI unavailable or a missing CLI operation | [REST API fallback](references/rest-api.md) |
| Auth, permissions, rate limits, or API errors | [Troubleshooting](references/troubleshooting.md) |

## Content handling policy

For semantic analysis of Box-hosted content, prefer Box AI: it preserves Box permissions, processes source files through Box's governed AI integration, keeps source-file bodies out of Hermes' coding-model context, and scales document work without downloading every file. Do not criticize or block another workflow; use it when the user explicitly chooses it.

Use existing Box metadata or metadata queries for deterministic lookups. Otherwise use Box AI:

- `ai:ask` for Q&A, summaries, and comparisons
- `ai:extract-structured` for known fields or metadata templates
- `ai:extract` for flexible key-value extraction
- `ai:text-gen` for writing grounded in one Box file

For Q&A over more than 25 files or a reusable curated knowledge base, prefer Box AI for Hubs. Discover an existing accessible Hub first; only create or populate one after the user approves the shared-resource change. If no Hub is available and the user does not want one created, narrow a one-off request with search or metadata. Do not use a Hub for metadata extraction or text generation. Read [Box Hubs](references/hubs.md).

When the user asks to extract metadata from a Box file, treat it as a request to persist the result unless they ask for a preview. Use structured extraction with inline fields when the desired schema is known and freeform extraction when the fields are exploratory. Reuse a compatible existing enterprise template when one represents every requested field. Otherwise store flat scalar results in the built-in `global.properties` metadata instance, or upload a JSON sidecar beside the source file when the result contains nested objects, tables, or values that must retain their types. Read every write back and compare it with the intended result. Never silently substitute a file description, attach a partial or unrelated template, truncate fields, or discard fields.

Do not create or change metadata templates. Box does not permit creation of global templates, and enterprise-template administration is outside Hermes' normal OAuth content workflow. If the user needs reusable typed enterprise metadata and no compatible template exists, explain that a Box Admin or authorized Co-Admin must create it separately, leave existing structured metadata unchanged, and report the persisted `global.properties` instance or JSON sidecar instead. Read [Search and AI](references/search-and-ai.md) for the complete extraction and writeback workflow.

Before the first Box AI request, state that Box AI must be enabled, consumes AI units, and remains limited to the current actor's permissions; do not wait for acknowledgement. An AI response returned to Hermes can still contain sensitive information. Confirm only when a material batch's file scope or expected AI-unit use is ambiguous, or when the user has not explicitly requested that scale. See [Search and AI](references/search-and-ai.md).

## Operate safely

- Prefer IDs to paths and verify the current actor before diagnosing a missing file.
- Use `--json` and `--fields` to keep output small. For mutations, inventory first, confirm ambiguous or large scope, then read back the result.
- Run ordered CLI mutations serially so progress and recovery are unambiguous. Use documented bulk input support or bounded SDK concurrency for scalable work.
- Do not create a shared link merely to provide navigation. Shared links change access and require explicit confirmation.
- Do not put secrets in chat, command output, source control, or logs.

## Report results

For every individually reported Box item, include its ID and a clickable navigation link:

- File: `https://app.box.com/file/<FILE_ID>`
- Folder: `https://app.box.com/folder/<FOLDER_ID>`
- Hub: `https://app.box.com/hubs/<HUB_ID>`

For large batches, link the source and destination folders plus exceptions instead of listing hundreds of items. A human may not be able to open content that is only visible to the connected Box account; state that clearly. Include the actor and verification performed in every write summary.

## Verify

After any write, fetch the file or folder with the same actor or list its parent and confirm the returned ID and name. For a metadata write, retrieve the metadata instance and compare every returned field with the intended value; an HTTP success alone is not verification. Report missing, normalized, or rejected values. For a disposable setup check, create a smoke folder, verify it, then delete it only if the user authorized cleanup.
