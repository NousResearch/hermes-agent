# Notion Setup, Auth, CLI, and SDK

Sources:

- `https://developers.notion.com/reference/authentication.md`
- `https://developers.notion.com/guides/get-started/authorization.md`
- `https://developers.notion.com/guides/get-started/internal-connections.md`
- `https://developers.notion.com/guides/get-started/public-connections.md`
- `https://developers.notion.com/guides/get-started/personal-access-tokens.md`
- `https://developers.notion.com/guides/get-started/handling-api-keys.md`
- `https://developers.notion.com/cli/get-started/*.md`
- `https://raw.githubusercontent.com/makenotion/notion-sdk-js/main/README.md`

## Token variables

Hermes environment:

```dotenv
NOTION_API_KEY=ntn_xxx_or_secret_xxx
```

`ntn` environment:

```dotenv
NOTION_API_TOKEN=ntn_xxx_or_secret_xxx
```

If both are needed, duplicate the literal token value. Do not use shell expansion in `~/.hermes/.env`.

Avoid shell-style `export` lines or variable expansion in `.env`. Hermes' dotenv loader reads literal `KEY=value` lines and does not evaluate shell syntax.

## REST headers

Every REST request needs:

```text
Authorization: Bearer <token>
Notion-Version: 2026-03-11
```

JSON body requests also need:

```text
Content-Type: application/json
```

## Internal connections

Use for one workspace, server-side/bot automation, and controlled scripts.

Facts:

- Created by a Workspace Owner.
- Acts as its own bot identity.
- Has no content access by default.
- Grant access in Developer Portal or Notion UI by adding the connection to a page/database.
- Parent access grants child access.
- Access persists even if the user who added the connection leaves.
- Workspace Owners can see internal connections.

Failure patterns:

- `404 object_not_found`: page/data source not shared with the connection.
- `403 restricted_resource`: missing capability or permission.

## Public OAuth connections

Use for multi-user products or third-party apps.

Flow:

1. Redirect user to Notion authorization URL.
2. Include `client_id`, `redirect_uri`, `response_type=code`, `owner=user`, optional `state`.
3. User selects pages/databases they can fully access.
4. Notion redirects back with temporary `code`.
5. Exchange code at `POST https://api.notion.com/v1/oauth/token` with HTTP Basic auth using `CLIENT_ID:CLIENT_SECRET`.
6. Store `access_token`, `refresh_token`, `bot_id`, workspace metadata, and owner data.

Use `state` for CSRF protection and app-state restoration.

Redirect URI rule from docs:

- Required in token exchange if supplied in authorization URL or if multiple redirect URIs are configured.
- Not allowed in token exchange if exactly one redirect URI is configured and the auth URL did not include it.

Refresh behavior:

- Refreshing returns a new access token and a new refresh token.
- Store both returned tokens atomically; a half-written refresh result can strand the authorization record.
- No `expires_in` or token expiry timestamp is documented in the focused official token response; use introspection/refresh/error handling rather than invented expiry math.
- Treat refresh token families as single-owner secrets; do not share across tools without a cutover plan.

## Personal access tokens

Use for trusted personal scripts, CLI workflows, Workers, and development.

Facts:

- Belongs to one user in one workspace.
- Acts as the user who created it.
- Uses the creator's page/workspace permissions.
- Expires one year after creation.
- Guests/restricted members cannot create PATs or log into `ntn`.
- Admins can view/revoke PATs but cannot reveal another member's secret.
- PATs cannot list all workspace users; use `/v1/users/me` or retrieve the token creator instead.
- Admin policy changes or creator access loss can invalidate API access even if the literal token string still exists.

Use public OAuth rather than PATs for products used by multiple users.

## Capabilities

Main capability families:

- Content: read, insert, update.
- Comments: read, insert.
- Users: none, without email, with email.

Capability + access are both required. Sharing a page is not enough if the token lacks the endpoint capability.

## `ntn` CLI

Install:

```bash
curl -fsSL https://ntn.dev | bash
# or
npm install --global ntn
```

Requirements from docs:

- macOS/Linux, x64/arm64.
- npm install requires Node 22+ and npm 10+.
- Windows native support was listed as coming soon; use curl or WSL2.

Auth modes:

```bash
ntn login
```

Interactive browser login stores workspace-scoped tokens in the OS keychain. Headless login prints a URL, verification code, and `ntn login poll` command.

For unattended/CI/PAT use:

```bash
NOTION_API_TOKEN=ntn_xxx ntn api v1/users/me
```

Environment variables documented by `ntn`:

- `NOTION_API_TOKEN` — takes precedence over keychain auth.
- `NOTION_WORKSPACE_ID` — target a specific workspace for one command.
- `NOTION_KEYRING=0` — opt out of OS keychain for `ntn login`; stores plain JSON `auth.json`.
- `NOTION_HOME` — config directory override.
- `NOTION_ENV` — environment selection.

API requests:

```bash
ntn api v1/users/me
ntn api v1/data_sources/${DATA_SOURCE_ID}/query page_size:=50
ntn api v1/pages/${PAGE_ID}/markdown
ntn api v1/pages/${PAGE_ID}/markdown -X PATCH command[type]=replace_content command[new_str]="# New body"
```

Inspection:

```bash
ntn api ls
ntn api v1/pages --spec -X POST
ntn api v1/pages --docs -X POST
ntn --verbose api v1/users/me
```

Do not use `--unsafe-verbose` with real tokens unless you intentionally want secrets in logs.

## Official JS/TS SDK

Install:

```bash
npm install @notionhq/client
```

Initialize with explicit API version for new code:

```javascript
const { Client } = require("@notionhq/client");

const notion = new Client({
  auth: process.env.NOTION_API_KEY,
  notionVersion: "2026-03-11",
});
```

README facts from 2026-05-18 fetch:

- Runtime: Node >=18; optional TypeScript >=5.9.
- SDK v5+ minimum recommended Notion API version: `2025-09-03`.
- Current SDK supports `2025-09-03` and `2026-03-11`.
- Default SDK version is `2025-09-03`; pass `notionVersion: "2026-03-11"` to opt into latest.
- Retries: 429 for all methods; 500/503 for idempotent GET/DELETE. Defaults to 2 retries with exponential backoff/jitter and respects `Retry-After`.
- Useful helpers: `iteratePaginatedAPI`, `collectPaginatedAPI`, `isFullPage`, `isFullBlock`, `isFullDataSource`, `isNotionClientError`, `APIErrorCode`.

## Credential safety

- Treat all Notion tokens like passwords.
- Do not commit or paste tokens into chat, skills, docs, logs, screenshots, or webhook payload examples.
- Use separate tokens per environment/script.
- Rotate/revoke on compromise.
- Store webhook verification tokens like secrets.
