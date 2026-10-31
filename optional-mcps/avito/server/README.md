# Avito MCP connector — server

Local MCP server for Avito's API, wrapping the open-source `avito-api`
PyPI client (https://pypi.org/project/avito-api/) against the real Avito
REST API (`https://api.avito.ru`). Catalog metadata is in `../manifest.yaml`.

Business-logic guidance for which Avito API covers what lives in the
`avito-api-*` Hermes skills (category `avito`) — this server is pure
transport (auth + HTTP calls).

## Setup

The server runs in its own virtualenv (kept separate from Hermes's own
dependencies on purpose — this install's `PYTHONPATH` points at Hermes's
venv, which breaks a plain system/venv `pip install mcp` unless cleared):

```bash
python3 -m venv ~/.hermes/mcp-servers/avito/venv
PYTHONPATH="" ~/.hermes/mcp-servers/avito/venv/bin/pip install -r requirements.txt
```

The catalog manifest (`../manifest.yaml`) points `transport.command` at
that venv's `python3` and `transport.args` at `avito_mcp_server.py` in
this directory. If you move the venv, update the manifest's `command`
path to match.

## Auth

Two login modes, both exposed as MCP tools (no Hermes-managed OAuth —
Avito's OAuth2 is handled inside this server):

- `avito_login_app_auth` — app-level, client_credentials grant, no browser.
- `avito_login_start` / `avito_login_wait_callback` / `avito_login_finish` —
  personal, authorization_code grant, browser-based OAuth:
  - `avito_login_start(scopes)` returns a URL to open in a browser.
  - Avito **requires** the app's Redirect URI to be a real, owned-domain
    HTTPS endpoint (its app-registration form explicitly rejects bare/
    third-party catchers like webhook.site — "domain must belong to you").
    This install uses a shared callback catcher behind
    `https://cb.mstylehome.ru/avito` — a Cloudflare named Tunnel
    (`mstyle-oauth-callback`, zone `mstylehome.ru`) forwarding to a small
    local HTTP server (`~/.hermes/mcp-servers/oauth-callback/
    callback_server.py`) on `127.0.0.1:8743`, which writes the incoming
    `code` to `~/.hermes/mcp-tokens/oauth-callbacks/avito.json`.
  - After approving in the browser, call `avito_login_wait_callback()` —
    it polls that file and finishes the login automatically, no manual
    code copy-paste. `avito_login_finish(code)` still works directly if
    you have the code some other way.
  - **Both the callback catcher and `cloudflared tunnel run` must be
    running** for this to work (they are separate long-lived processes,
    not managed by this MCP server) — check `ps aux | grep -E
    "callback_server|cloudflared"` and restart them if the host rebooted.
  - A different Avito app (different Redirect URI) can still be used by
    passing `redirect_uri` explicitly to `avito_login_start`, then reading
    the code back manually with `avito_login_finish`.

`AVITO_CLIENT_ID` / `AVITO_CLIENT_SECRET` come from an app registered at
avito.ru/developers; the catalog install flow prompts for them and saves
them to `~/.hermes/.env`. Tokens are cached at `~/.hermes/mcp-tokens/avito.json`.

## Coverage

Wired: Messenger (chats/messages), listings (Объявления), user profile.
Not wired: everything else in Avito's ~26-API catalog — no verified
endpoint spec was available when this was built (see `avito-api-catalog`
skill's Pitfalls on the site's anti-bot block). Extend `avito_mcp_server.py`
only against a confirmed spec — never guess a path.
