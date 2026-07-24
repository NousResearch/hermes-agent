# Auth and Setup

Hermes uses **Client Credentials Grant (CCG)** — a Platform App authenticates as its **service account**, not the human using Hermes.

## Create a CCG Platform App

1. Sign in at [Developer Console](https://app.box.com/developers/console) (free developer accounts work).
2. **Create New App** → **Platform App** → **Client Credentials Grant**.
3. Enable **2FA** on your Box account (required to view/copy client secret).
4. On **Configuration**: copy **Client ID** and **Client Secret**.
5. On **General Settings**: copy **Enterprise ID**.
6. Set **Application Scopes** for the operations you need (read/write files, manage webhooks, etc.).

Auth method is **locked at creation** — you cannot switch to OAuth or JWT on the same app.

## Authorization

| Account type | What happens |
| --- | --- |
| Free developer (Individual) | App is **auto-authorized** on creation |
| Enterprise admin/co-admin | Click **Authorize** on the Configuration tab |
| Enterprise non-admin | Click **Submit** and wait for admin approval |

Docs: [Platform App Approval](https://developer.box.com/guides/authorization/platform-app-approval)

## Store secrets in Hermes

Add to `~/.hermes/.env` (never paste secrets into chat):

```
BOX_CLIENT_ID=your_client_id
BOX_CLIENT_SECRET=your_client_secret
BOX_ENTERPRISE_ID=your_enterprise_id
```

## Wire Box CLI

Copy `templates/ccg-config.json.example`, substitute values from `.env`, then run via `terminal`:

```bash
box configure:environments:add /path/to/ccg-config.json --ccg-auth --name hermes --set-as-current
box users:get me --json --fields id,name,login
```

If multiple environments exist: `box configure:environments:set-current hermes`.

Safe auth check: `box users:get me --json`. Do **not** use `box configure:environments:get --current` routinely — it can print sensitive details.

## Service account content model

The CCG service account has its **own folder tree** (starts empty). It cannot see a managed user's "My Box" unless you grant access.

### Find the service account email

After the app is **authorized**, Box creates a service account user. Find its email:

- **Developer Console** — open your app → **General Settings** tab → **Service Account** section. Format: `AutomationUser_<app-id>_…@boxdevedition.com` ([User Types docs](https://developer.box.com/platform/user-types#service-account)).
- **CLI** — `box users:get me --json --fields id,name,login` → `login` field.

### Invite the service account to folders

In the Box web app: open the folder → **Invite People** → paste the service account email → assign **Viewer**, **Editor**, or **Co-owner** as needed. Collaborate the **parent** folder when Hermes needs the whole subtree.

Via CLI (when you already have access to the folder):

```bash
box collaborations:create <FOLDER_ID> folder --role editor --login AutomationUser_...@boxdevedition.com --json
```

### Access patterns

1. **Shared folders (recommended for existing content)** — user invites the service account email to team/personal folders Hermes should use.
2. **Hermes workspace** — upload to or create folders under the service account root (`folder id 0`).
3. **User impersonation (advanced)** — enable **App + Enterprise Access** and **Generate User Access Tokens** in Developer Console, re-authorize the app, then:
   ```bash
   box configure:environments:add /path/to/ccg-config.json --ccg-auth --ccg-user USER_ID --name hermes-as-user --set-as-current
   ```

## Actors summary

| Actor | When |
| --- | --- |
| Service account (default) | Unattended Hermes, team workspace, bot-owned content |
| Managed user (`--ccg-user` / `--as-user`) | Act on a specific user's existing content (requires app config + admin) |
| App user | Per-tenant isolation in multi-tenant apps (SDK path) |

Always log which actor ran each operation — most Box bugs are actor mismatches.

## Install Box CLI

Requires Node.js 18+:

```bash
npm install -g @box/cli
box --version
```

## Official links

- CCG setup: https://developer.box.com/guides/authentication/client-credentials/client-credentials-setup
- User types: https://developer.box.com/platform/user-types
- CLI auth: https://developer.box.com/guides/cli/quick-start
