# One-time OAuth setup

Zoho uses OAuth 2.0. You create one client, approve it once in the browser and keep the refresh token. The refresh token does not expire unless it is revoked; access tokens last one hour and the helper renews them.

Do the code exchange yourself in a terminal, not through the agent, so the refresh token never lands in a chat transcript.

## 1. Find your data center

The domain you log in to decides everything:

| Login at | `ZOHO_DC` |
|---|---|
| zoho.com | `com` |
| zoho.eu | `eu` |
| zoho.in | `in` |
| zoho.com.au | `com.au` |
| zoho.jp | `jp` |
| zohocloud.ca | `ca` |
| zoho.sa | `sa` |
| zoho.com.cn | `com.cn` |

Every host (accounts, CRM, Mail, WorkDrive) must be from the same data center. Mixing them gives `invalid_client` or WorkDrive `F7003`, which look like auth errors.

## 2. Create a Server-based client

1. Open the API Console of your region (`https://api-console.zoho.com`, `https://api-console.zoho.eu`, ...).
2. **Add Client**, choose **Server-based Applications**.
3. Homepage URL: any URL you own. Authorized Redirect URI: `http://localhost:8080/callback` (nothing needs to listen there).
4. Note the client ID and client secret.

Why not a Self Client: its codes expire within seconds, it gives no refresh token in some flows, and WorkDrive rejects its tokens with `F7003`.

## 3. Approve and exchange the code

Put the client into `${HERMES_HOME:-~/.hermes}/.env` first:

```
ZOHO_CLIENT_ID=1000.xxxxxxxx
ZOHO_CLIENT_SECRET=xxxxxxxx
ZOHO_DC=eu
```

Then, in your own terminal, with those variables loaded:

```
python3 scripts/zoho_api.py auth-url
```

Open the printed URL, log in, approve. The browser then fails to load `localhost:8080`; that is expected. Copy the `code=` value from the address bar and, within two minutes:

```
python3 scripts/zoho_api.py exchange-code <code>
```

Store the printed value as `ZOHO_REFRESH_TOKEN` in `${HERMES_HOME:-~/.hermes}/.env` and run `check`.

`auth-url` asks for `access_type=offline` (otherwise no refresh token) and `prompt=consent` (otherwise Zoho may skip scopes you added later).

## 4. Scopes

Default scope list of `auth-url`:

```
ZohoCRM.modules.ALL, ZohoCRM.settings.READ, ZohoCRM.users.READ, ZohoCRM.coql.READ,
ZohoMail.accounts.READ, ZohoMail.folders.READ, ZohoMail.messages.ALL,
ZohoMail.messages.CREATE, ZohoMail.messages.READ, ZohoMail.messages.UPDATE,
WorkDrive.files.ALL, WorkDrive.teamfolders.ALL, WorkDrive.workspace.ALL, WorkDrive.users.READ
```

Pass `--scopes` for a narrower set, for example read only: `ZohoCRM.modules.READ,ZohoMail.accounts.READ,ZohoMail.folders.READ,ZohoMail.messages.READ,WorkDrive.files.READ`.

Notes on scope names (they are inconsistent between products):

- List `ZohoMail.messages.UPDATE` explicitly. Moving or flagging mails fails without it, even with `messages.ALL`.
- `WorkDrive.users.READ` is needed to list teams. `/users/me` works without it.
- `WorkDrive.files.ALL` does not include sharing; add `WorkDrive.files.sharing.ALL` if you need share links.
- Desk scopes have no `Zoho` prefix (`Desk.tickets.READ`).
- If the console rejects one scope, remove only that one and retry.

To add scopes later, run `auth-url --scopes ...` again with the full new list and replace the refresh token.

## 5. Rate limit on the token endpoint

After roughly 20 refreshes in quick succession, Zoho answers `too many requests continuously` for several minutes. The helper caches the access token in `$HERMES_HOME/cache/zoho-one/access.json` and reuses it across runs until five minutes before expiry. Scripts you write yourself should do the same: one refresh per run, not per request.
