---
name: here.now
description: Publish sites to live URLs, team workspaces, and Drives.
version: 1.35.0
author: here.now
license: MIT
prerequisites:
  commands: [curl, file, jq]
platforms: [macos, linux]
metadata:
  hermes:
    tags: [here.now, herenow, publish, deploy, hosting, static-site, web, share, URL, drive, storage, access-control, password]
    homepage: https://here.now
    requires_toolsets: [terminal]
---

# here.now

**Skill version: 1.35.0**

here.now lets agents publish websites and files to live URLs in seconds.

The core primitive is a **Site**: publish a file or folder and get a live URL at `{slug}.here.now` or a custom domain. Every Site has access control: public link (default), password, or restricted invite-only access.

here.now also includes **workspaces** — shared team accounts where Sites belong to the team and serve at `{label}.{workspace}.here.now` (see "Publish to a workspace" below).

A personal account on a paid plan can turn on a **vanity URL**: the user's name at `{name}.here.now`, and from then on every Site they publish (including through you) also serves at a readable `{site-name}.{name}.here.now` address, named from its title. When a user wants every Site under their own name, `PUT /api/v1/vanity-urls/subdomain` with `{"subdomain": "name"}` turns it on; for one Site at one hostname they choose, use a custom domain; for Sites owned by a team, use a workspace. Finalize responses carry `primaryUrl` and `urls[]` (best first); mention `primaryUrl` when it differs from `siteUrl`. See https://here.now/docs#vanity-urls.

## When to use

Use this skill when asked to "publish this", "host this", "deploy this", "share this on the web", "make a website", "put this online", "create a webpage", "generate a URL", "build a chatbot", "password protect this site", "make this site private", or "share this site with only certain people". For team workspaces, use it when asked to "publish this to our team workspace", "share this with my team", or "put this in our company workspace". For Drives, use it when asked to "save this to my Drive", "store this for later", "write this to cloud storage", or "share a folder with another agent". For domains, use it when asked to "buy a domain", "get me a .com for this", or "register a domain".

## Current docs

**Before answering questions about here.now capabilities, features, or workflows, read the current docs:**

→ **https://here.now/docs**

Read the docs:

- at the first here.now-related interaction in a conversation
- any time the user asks how to do something
- any time the user asks what is possible, supported, or recommended
- before telling the user a feature is unsupported

Topics that require current docs (do not rely on local skill text alone):

- Site access control (passwords and restricted access)
- workspaces (team accounts, membership, label URLs)
- folders (organizing the user's dashboard; see https://here.now/docs#folders)
- Drives and Drive sharing
- custom domains
- vanity URLs (`{site-name}.{name}.here.now`; see https://here.now/docs#vanity-urls)
- buying a domain (search and quote first; state the price and the renewal price and get the user's explicit yes before calling purchase — purchases are final; see https://here.now/docs#buy-domain)
- Site Data
- public profiles
- proxy routes and service variables
- limits and quotas
- SPA routing
- owner Site search
- Site analytics
- Site version history, previews, and rollback
- error handling and remediation
- feature availability

**If docs and live API behavior disagree, trust the live API behavior.**

Command-line fetches of https://here.now/docs (curl, WebFetch, etc.) receive a markdown summary, not the full HTML docs: it lists every stable public endpoint with a one-line description, but section anchors in this skill (like `/docs#access-control`) resolve only in the HTML version, and worked examples live there too. For complete request/response schemas and parameters, fetch **https://here.now/openapi.json**. Do not conclude an operation is unsupported from the markdown summary alone — check the OpenAPI spec first.

If the docs fetch fails or times out, continue with the local skill and live API/script output. Prefer live API behavior for active operations.

## Requirements

- Required binaries: `curl`, `file`, `jq`
- Required network access: `https://here.now` (API) and `https://*.r2.cloudflarestorage.com` (file uploads PUT directly to storage; the exact host is in each upload URL). In a sandbox or behind a proxy with an egress allowlist, allow BOTH hosts. With only `here.now` allowed, the create call succeeds, every upload fails, and finalize reports missing files.
- If `https://here.now` cannot be reached from where you run (connections time out while other sites load; some hosting providers block the address range it resolves to), the same API is at `https://api.here.now`. Pass `--base-url https://api.here.now` to `publish.sh` and `drive.sh`; both accept it without any override flag. Nothing else changes: Sites still serve at `{slug}.here.now`, and the key, state file and output are the same.
- Optional environment variable: `$HERENOW_API_KEY`
- Optional Drive token variable: `$HERENOW_DRIVE_TOKEN`
- Optional credentials file: `~/.herenow/credentials`
- Skill helper paths:
  - `${HERMES_SKILL_DIR}/scripts/publish.sh` for publishing sites
  - `${HERMES_SKILL_DIR}/scripts/drive.sh` for private Drive storage

## If the helper scripts aren't installed

Some environments receive this document without the bundled `scripts/` directory (for example, hosted platform integrations that provide only `$HERENOW_API_KEY`). In that case, either install the full bundle first:

```bash
npx skills add heredotnow/skill --skill here-now -g
```

or call the API directly — every script workflow in this document is a wrapper over the public API. Publishing is a three-step flow: `POST /api/v1/publish` with a `files` array (`[{path, size}]`) returns presigned upload targets, `PUT` each file's bytes to its returned URL, then `POST` the returned `finalizeUrl`. The site is not live until finalize succeeds. Full walkthrough with request/response examples: https://here.now/docs#create (then #upload and #finalize), machine-readable schemas: https://here.now/openapi.json.

## Create a site

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

Outputs the live URL (e.g. `https://bright-canvas-a7k2.here.now/`).

Under the hood this is a three-step flow: create/update -> upload files -> finalize. A site is not live until finalize succeeds. Finalize for a large Site can take minutes; if a proxy cuts the connection first or the server reports the finalize still running, the script repeats the same finalize call (it is idempotent) until it gets a real answer, for up to 15 minutes (`HERENOW_FINALIZE_WAIT_SECONDS`). It exits 1 with `publish_result.finalize=timeout` only when that budget runs out, and the Site may still go live afterwards; the message shows the call that checks.

Without an API key this creates an **anonymous site** that expires in 24 hours.
With a saved API key, the site is permanent.

**File structure:** For HTML sites, place `index.html` at the root of the directory you publish, not inside a subdirectory. The directory's contents become the site root. For example, publish `my-site/` where `my-site/index.html` exists — don't publish a parent folder that contains `my-site/`.

You can also publish raw files without any HTML. Single files get a rich auto-viewer (images, PDF, video, audio). Multiple files get an auto-generated directory listing with folder navigation and an image gallery.

**WebAssembly builds** (Emscripten, Rust, Godot, Unity, Pyodide, WebLLM, ...) publish like any folder: point the script at the build output directory and it runs, with `.wasm` served and compressed correctly. If the build uses threads (SharedArrayBuffer: Emscripten `-pthread`, Godot with Thread Support on, Unity multithreading), pass `--cross-origin-isolation`; single-threaded builds need nothing. A `publish_result.warning=` line mentioning WebAssembly means the Site has `.wasm` but the setting is off. Details and trade-offs: https://here.now/docs#wasm

## Update an existing site

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --slug {slug} --client hermes
```

The script auto-loads the `claimToken` from `.herenow/state.json` when updating anonymous sites. Pass `--claim-token {token}` to override; a token passed this way is saved to the state file, so later updates of that slug from the same directory need no flag.

Authenticated updates require a saved API key.

**Sites the user created at here.new.** Visiting https://here.new creates an empty here.now Site on the spot and shows the user a prompt to paste into you. That prompt names the Site's address (`https://{slug}.here.now/`) and, for an anonymous Site, its claim token; a signed-in user's Site is already in their account and the prompt says so. The Site exists, so **do not create a new one**: build what the user asks for and publish it as an update to that slug.

- Anonymous: `bash "${HERMES_SKILL_DIR}/scripts/publish.sh" {dir} --slug {slug} --claim-token {token} --client hermes`. No API key is needed or used. The Site expires 24 hours after it was created unless claimed; when you are done, give the user the claim link from their prompt (`https://here.now/c/{token}`), copied byte-for-byte.
- In the user's account: `bash "${HERMES_SKILL_DIR}/scripts/publish.sh" {dir} --slug {slug} --client hermes` with their API key (run the sign-in flow below first if none is saved).

Until you publish, the address serves a placeholder page that says it is waiting for an agent; the user's browser tab refreshes into the real site on its own after your first publish.

**Stale-base protection.** The live Site may have changed since your local files were published — the owner can edit it from other tools (another agent, the here.now Editor, a teammate). The script records the live `versionId` in `.herenow/state.json` after each publish and sends it as `baseVersionId` on the next update of the same slug from the same directory; if the live Site moved past it, the update is rejected with `code: "version_conflict"` naming the live version and what created it. When that happens, relay the message to the user and offer to (a) read the live files with `GET /api/v1/publish/{slug}/files` (lists them with a `url` each) and `GET /api/v1/publish/{slug}/files/{path}` (the bytes; owner API key, works for password-protected and restricted Sites without the visitor password), reconcile them into the local files, and republish, or (b) re-run with `--overwrite` to replace the live version anyway. Before editing local files for an authenticated Site you haven't touched recently, check for drift first: `GET /api/v1/publish/{slug}` returns `currentVersionId` plus `currentVersionSource` and `currentVersionCreatedAt` (what changed it and when, e.g. `editor`) — if the id differs from your state file's `versionId`, read the live files before editing. The published version is the shared truth; never fetch the public URL to read an owned Site (it is gated for protected Sites) and never ask the user for a visitor password to read their own Site. Anonymous Sites can't call these endpoints; they rely on the saved state and server enforcement. Omitting `baseVersionId` (or using `--overwrite`) is an unchecked full replacement.

Every publish records an immutable version. If the user asks to see earlier versions of a Site, undo a publish, or roll back: list history with `GET /api/v1/publish/{slug}/versions` and restore instantly with `POST /api/v1/publish/{slug}/versions/{versionId}/restore` (restoring keeps the current access mode, password, and domains). Version access is included on every plan, personal and workspace alike. A byte-identical republish returns `unchanged: true` from finalize instead of creating a new version. See https://here.now/docs#versions.

Signed-in users also have public profiles. Agents can help users show or hide Sites on their profile and manage profile settings through the API documented at https://here.now/docs#profile.

## Publish to a workspace

Workspaces are shared team accounts: Sites published into one belong to the team, not the publishing member, and get a memorable URL at `{label}.{workspace}.here.now`.

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --workspace {subdomain} --client hermes
```

Requires a saved API key and membership in the workspace. List the user's workspaces (and valid subdomains) with `GET /api/v1/accounts`. Workspace Sites default to member-only access; the script reports the team URL as `publish_result.account_url`.

For everything else — creating workspaces, invites and auto-join, workspace domains and variables, label renames — read the current docs:

→ **https://here.now/docs#workspaces**

## Site access control

A Site uses one access mode at a time:

- **anyone_with_link** (default): anyone with the URL can view.
- **password**: visitors must enter a shared password.
- **restricted**: invite-only; only verified email addresses or email domains the owner allows can view.

Workspace-owned Sites default to **account_members** (visitors sign in and must be workspace members) and also support public, public with a password, and **restricted**. On a workspace Site, `restricted` means workspace members plus a per-Site guest allowlist: members always have access, and allowlisted emails/domains are outside guests who can view only that Site — they never become workspace members, though the Site appears in the guest's own dashboard as a shared Site. Workspace restricted requires at least one guest email or domain — an empty allowlist is rejected with a 400 (use `account_members` for members-only). See https://here.now/docs#workspace-access.

Manage access with `GET`/`PATCH /api/v1/publish/{slug}/access` (passwords via the metadata endpoint). Restricted access requires a claimed Site. The PATCH replaces the full allowlists — read, merge, then write. Before working with access control, read the current docs:

→ **https://here.now/docs#access-control**

## Folders (organizing the user's dashboard)

Signed-in users can group their Sites into **folders** in the here.now dashboard. A folder is a fact about the account, not the Site's address: flat (no nesting), one folder per Site, shared by every member of a workspace, invisible to visitors. Filing a Site changes nothing about its URL or access.

When the user names a folder, project, or client for a Site ("publish this to my Reports folder", "file it under Acme"), file it as part of the publish:

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --folder "Reports" --client hermes
```

`--folder` takes a folder name or id. A name is matched case-insensitively and **created if it does not exist yet**, so this works the first time; the script reports the result as `publish_result.folder`. It requires an API key (anonymous Sites have no account to file in) and works with `--slug` (moves an existing Site) and `--workspace` (the workspace's folders).

Without the script, the same field is `folder` on `POST /api/v1/publish`, `PUT /api/v1/publish/{slug}`, and `PATCH /api/v1/publish/{slug}/metadata` (the way to file a Site without publishing a new version; `"folder": null` moves it back to the root). `GET /api/v1/folders` lists the account's folders; `GET /api/v1/publishes?folder={name-or-id}` lists one folder's Sites. Rename and delete are `PATCH`/`DELETE /api/v1/folders/{id}` (deleting a folder unfiles its Sites and deletes nothing else). Up to 50 folders per account, names up to 60 characters.

**Never invent folders or file Sites the user did not ask to file.** The folder structure is theirs; every folder appears as a tile in their dashboard. See https://here.now/docs#folders.

## Use a Drive

Use a Drive when the user wants private cloud storage for agent files: documents, context, memory, plans, assets, media, research, code, and anything else that should persist without being published as a website.

Every signed-in account has a default Drive named `My Drive`.

```bash
DRIVE="${HERMES_SKILL_DIR}/scripts/drive.sh"
bash "$DRIVE" default
bash "$DRIVE" ls "My Drive"
bash "$DRIVE" put "My Drive" notes/today.md --from ./notes/today.md
bash "$DRIVE" cat "My Drive" notes/today.md
bash "$DRIVE" share "My Drive" --perms write --prefix notes/ --ttl 7d
```

Use scoped Drive tokens for agent-to-agent handoff. If you receive a `herenow_drive` share block, use its `token` as `Authorization: Bearer <token>` against `api_base`, respect `pathPrefix` when present, and preserve ETags on writes. A `pathPrefix` of `null` means full-Drive access. If the skill is available, prefer `drive.sh`; otherwise call the listed API operations directly.

## Client attribution

Pass `--client` with the name of the **agent product or harness you are running in** — for Hermes, that is always `hermes`:

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

This sends `X-HereNow-Client: hermes/publish-sh` on publish API calls. If omitted, the script sends a fallback value.

Use the platform's name, **not** a bot, persona, sub-agent, project, or thread name inside it. To record your instance name too, append it after a slash: `--client hermes/research-bot`.

## API key storage

The publish script reads the API key from these sources (first match wins):

1. `--api-key {key}` flag (CI/scripting only — avoid in interactive use)
2. `$HERENOW_API_KEY` environment variable
3. `~/.herenow/credentials` file (recommended for agents)

To store a key, write it to the credentials file:

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

**IMPORTANT**: After receiving an API key, save it immediately — run the command above yourself. Do not ask the user to run it manually. Avoid passing the key via CLI flags (e.g. `--api-key`) in interactive sessions; the credentials file is the preferred storage method.

Never commit credentials or local state files (`~/.herenow/credentials`, `.herenow/state.json`) to source control.

## Getting an API key

To upgrade from anonymous (24h) to permanent sites:

1. Ask the user for their email address.
2. Request a one-time sign-in code. Send `X-HereNow-Client` with the same platform name you pass as `--client` (see "Client attribution"): the code email names your platform, and the account is attributed to it.

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H "content-type: application/json" \
  -H "X-HereNow-Client: hermes" \
  -d '{"email": "user@example.com"}'
```

3. Tell the user: "Check your inbox for a sign-in code from here.now and paste it here."
4. Verify the code and get the API key, with the same header:

```bash
curl -sS https://here.now/api/auth/agent/verify-code \
  -H "content-type: application/json" \
  -H "X-HereNow-Client: hermes" \
  -d '{"email":"user@example.com","code":"ABCD-2345"}'
```

If you cannot set headers, pass the same value as `"client"` in the JSON body of either call.

5. Save the returned `apiKey` yourself (do not ask the user to do this):

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

## State file

After every site create/update, the script writes to `.herenow/state.json` in the working directory:

```json
{
  "publishes": {
    "bright-canvas-a7k2": {
      "siteUrl": "https://bright-canvas-a7k2.here.now/",
      "claimToken": "4fQ9tK2mXb7cW1pZ",
      "claimUrl": "https://here.now/c/4fQ9tK2mXb7cW1pZ",
      "expiresAt": "2026-02-18T01:00:00.000Z"
    }
  }
}
```

Before creating or updating sites, you may check this file to find prior slugs.
Treat `.herenow/state.json` as internal cache only.
Never present this local file path as a URL, and never use it as source of truth for auth mode, expiry, or claim URL.

## What to tell the user

For published sites:

- Always share the `siteUrl` from the current script run.
- Put the site URL on its own line with nothing else on that line — no punctuation, dashes, or status text after it (chat clients autolink everything up to whitespace, gluing your words into the URL). Status details like "permanent, saved to your account" go on the following line.
- Read and follow `publish_result.*` lines from script stderr to determine auth mode.
- `publish_result.warning=` lines are non-fatal: the Site is live, but something needs attention (an invalid `.herenow/proxy.json`, or a `.wasm` build published without `--cross-origin-isolation`). Act on them or tell the user.
- When `publish_result.account_url` is non-empty (workspace publishes), share it as the primary team URL alongside `siteUrl`.
- When `publish_result.primary_url` is non-empty, share it first; `siteUrl` stays valid.
- When `publish_result.folder` is non-empty, mention that the Site is filed in that folder in their dashboard.
- When `publish_result.auth_mode=authenticated`: tell the user the site is **permanent** and saved to their account. No claim URL is needed.
- When `publish_result.auth_mode=anonymous`: tell the user the site **expires in 24 hours**. Share the claim URL (if `publish_result.claim_url` is non-empty and starts with `https://`) so they can keep it permanently. Copy it byte-for-byte as a clickable link — never shorten, redact, summarize, or replace any part of it with `...`; a modified claim link will not work. Warn that claim tokens are only returned once and cannot be recovered.
- Never tell the user to inspect `.herenow/state.json` for claim URLs or auth status.

For Drives:

- Do not describe Drive files as public URLs.
- Tell the user Drive contents are private unless shared with a scoped token.
- When sharing access with another agent, prefer a scoped token with a narrow `pathPrefix` and short TTL.

## publish.sh options

| Flag                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| `--slug {slug}`        | Update an existing site instead of creating |
| `--workspace {subdomain}` | Publish into a workspace (team account) you belong to |
| `--claim-token {token}`| Override claim token for anonymous updates    |
| `--overwrite`          | Skip the stale-base check and replace the live version |
| `--title {text}`       | Viewer title (non-HTML sites)             |
| `--description {text}` | Viewer description                            |
| `--folder {name-or-id}` | File the Site in a dashboard folder (a name is created if missing; authenticated only) |
| `--ttl {seconds}`      | Set expiry (authenticated only)               |
| `--client {name}`      | Agent harness for attribution — the platform you run in (`hermes` here), not your bot/persona name; optionally append it: `hermes/research-bot` |
| `--base-url {url}`     | API base URL (default: `https://here.now`; `https://api.here.now` is the same API via Cloudflare for networks that cannot reach the default) |
| `--allow-nonherenow-base-url` | Allow sending auth to a `--base-url` other than the two above |
| `--api-key {key}`      | API key override (prefer credentials file)    |
| `--spa`                | Enable SPA routing (serve index.html for unknown paths) |
| `--cross-origin-isolation` | Send COOP/COEP headers so threaded WebAssembly (SharedArrayBuffer) runs; see https://here.now/docs#wasm |

## Beyond publish.sh

For Drive operations, use `drive.sh` or the Drive API. For broader account and Site management — Site Data, search, analytics, profiles, delete, metadata, access control, domains, variables, proxy routes, duplication, and more — see the current docs:

→ **https://here.now/docs**

Full docs: https://here.now/docs
