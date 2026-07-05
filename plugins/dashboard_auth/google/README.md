# Google OAuth dashboard auth provider — Deployment Guide

This plugin adds **"Sign in with Google"** to the Hermes web dashboard on a public HTTPS domain.

It was created based on the built-in Nous Portal OAuth provider that ships with Hermes.

For: `https://<your-dashboard-domain>`  
Callback: `https://<your-dashboard-domain>/auth/callback`

Replace `<your-dashboard-domain>` with your actual public domain.

---

## What this plugin does

Hermes ships with a built-in Nous Portal OAuth provider. It does **not** ship with a Google OAuth provider out of the box. The official docs state:

> To plug a non-Nous OAuth provider (e.g. Google, GitHub, custom OIDC), create a plugin that registers a `DashboardAuthProvider`.

The `self-hosted` OIDC provider bundled with Hermes **does not work with Google** because it only supports public PKCE clients, and Google Web OAuth clients are **confidential clients** that require a `client_secret`.

This plugin registers a `DashboardAuthProvider` that implements Google OAuth 2.0 with PKCE.

---

## 1. Create OAuth 2.0 credentials in Google Cloud Console

1. Open [https://console.cloud.google.com/auth/clients](https://console.cloud.google.com/auth/clients) or go to **APIs & Services → Credentials**.
2. Select or create a project (e.g. `hermes-dashboard`).
3. If you have not configured the OAuth consent screen yet:
   - Go to **APIs & Services → OAuth consent screen**.
   - Choose **External** (unless you have a Google Workspace).
   - Fill in:
     - App name: `Hermes Dashboard`
     - User support email: your email
     - Developer contact information: your email
   - Save and continue.
4. Go to **Credentials** and click **Create credentials → OAuth client ID**.
5. Application type: **Web application**.
6. Name: `Hermes Dashboard Web`.
7. Under **Authorized redirect URIs**, click **Add URI** and enter exactly your dashboard's callback URL, for example:

   ```text
   https://<your-dashboard-domain>/auth/callback
   ```

   Replace `<your-dashboard-domain>` with your actual public domain.

8. Click **Create**.
9. Copy the **Client ID** and **Client secret** from the popup.

---

## 2. Configure Hermes

Edit `/root/.hermes/config.yaml`:

```yaml
dashboard:
  public_url: https://<your-dashboard-domain>
  oauth:
    google:
      client_id: "123456789012-abcdefghijklmnopqrstuvwxyz.apps.googleusercontent.com"
      client_secret: "GOCSPX-abcdefghijklmnopqrstuvwxyz"
```

Replace `<your-dashboard-domain>` with your actual domain.

Or use environment variables (they take precedence):

```bash
HERMES_DASHBOARD_PUBLIC_URL=https://<your-dashboard-domain>
HERMES_DASHBOARD_GOOGLE_CLIENT_ID=123456789012-abcdefghijklmnopqrstuvwxyz.apps.googleusercontent.com
HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET=GOCSPX-abcdefghijklmnopqrstuvwxyz
HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS=your.email@example.com
```

`HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS` is **optional** but strongly recommended for personal or single-user dashboards. It is a comma-separated list. If left empty, any Google account can log in.

For a systemd service, put the env vars in `~/.hermes/.env` and make sure the unit has:

```ini
EnvironmentFile=%h/.hermes/.env
```

### Set config via CLI

```bash
hermes config set dashboard.public_url https://<your-dashboard-domain>
hermes config set dashboard.oauth.google.client_id "YOUR_CLIENT_ID"
hermes config set dashboard.oauth.google.client_secret "YOUR_CLIENT_SECRET"
hermes config set dashboard.oauth.google.allowed_emails "your.email@example.com"
```

---

## 3. Install the plugin

Source of truth:

```text
/home/github/projects/hermes/plugins/google-oauth-auth-provider/
├── __init__.py
├── plugin.yaml
└── README.md   # this file
```

Symlink into the Hermes plugin directory:

```bash
sudo ln -s /home/github/projects/hermes/plugins/google-oauth-auth-provider \
  /usr/local/lib/hermes-agent/plugins/google-oauth-auth-provider
```

Or copy the directory if you prefer not to use a symlink:

```bash
sudo cp -r /home/github/projects/hermes/plugins/google-oauth-auth-provider \
  /usr/local/lib/hermes-agent/plugins/
```

---

## 4. Restart the dashboard

```bash
sudo systemctl restart hermes-dashboard.service
```

Check the logs:

```bash
sudo journalctl -u hermes-dashboard.service -f
```

Look for:

```text
google-oauth-auth-provider: registered provider
```

---

## 5. Verify

```bash
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
```

Expected output:

```json
true
["google-oauth"]
```

Open `https://<your-dashboard-domain>/login` in a browser (replace with your actual domain). You should see a **Continue with Google** button.

---

## How the OAuth flow works

1. User clicks **Continue with Google**.
2. Backend generates a PKCE `code_verifier` and `state`, stores them in a short-lived cookie (`hermes_session_pkce`), and redirects the browser to:

   ```text
   https://accounts.google.com/o/oauth2/v2/auth?
     response_type=code
     &client_id=YOUR_CLIENT_ID
     &redirect_uri=https://<your-dashboard-domain>/auth/callback
     &scope=openid%20email%20profile
     &state=...
     &code_challenge=...
     &code_challenge_method=S256
     &access_type=offline
     &prompt=consent
   ```

3. User logs in at Google and is redirected back to:

   ```text
   https://<your-dashboard-domain>/auth/callback?code=...&state=...
   ```

4. Backend checks the `state` and PKCE verifier, then exchanges the code for tokens at `https://oauth2.googleapis.com/token`.
5. The access token is verified through Google's `tokeninfo` endpoint. The plugin validates that the token's `aud` claim equals your Google Client ID to prevent token substitution.
6. The tokeninfo verification result is cached for 5 minutes so the Google endpoint is not hit on every request.
7. If `HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS` is set, the user's email must match one of the listed addresses; otherwise login is rejected.
8. A session is created from the token claims (`sub`, `email`, `name`) and stored in HttpOnly cookies.

---

## Session storage

- **Credentials**: `client_id` and `client_secret` live in `/root/.hermes/config.yaml` or `~/.hermes/.env`. Never commit them to git.
- **User session**: stored in **HttpOnly, Secure, SameSite=Lax** cookies in the browser. No local user database is used.
- **Access token**: Google access token, verified via `https://oauth2.googleapis.com/tokeninfo` on every request. The verification result is cached for 5 minutes to reduce latency and avoid Google rate limits.
- **Refresh token**: stored in the `hermes_session_rt` cookie. Used to obtain a new access token when the old one expires.
- **Google does not return a new `id_token` on refresh**, so the provider uses the **access token** as the session handle, not the `id_token`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `auth_providers: ["nous"]` only | plugin not loaded or credentials empty | Check plugin path, config values, and logs. |
| Google error `redirect_uri_mismatch` | callback URI mismatch | Add exactly `https://<your-dashboard-domain>/auth/callback` to Google credentials. |
| `This app isn't verified` warning | consent screen not verified | Login still works; complete verification in Google Console for clean UI. |
| `tokeninfo aud mismatch` | wrong client_id in config | Make sure `client_id` matches the Google credential. |
| Refresh fails after some time | Google only issues refresh token on first consent | Add `access_type=offline` and `prompt=consent` (already in this plugin). Re-authorize if needed. |

---

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Provider implementation and plugin registration |
| `plugin.yaml` | Plugin manifest |
| `README.md` | This guide |

---

## Security notes

- Keep the `client_secret` out of git.
- The callback URI must be HTTPS on public hosts.
- Google OAuth `refresh_token` is issued only the first time the user consents. Store it; forcing re-consent may be required to get a new one.
- Use `HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS` to restrict login to specific Google accounts. If it is empty, any Google account can log in.

---

## References

- Hermes dashboard auth docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/web-dashboard
- Google OAuth 2.0 for web server apps: https://developers.google.com/identity/protocols/oauth2/web-server
- Google tokeninfo endpoint: https://oauth2.googleapis.com/tokeninfo
