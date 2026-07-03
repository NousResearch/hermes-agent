# Mikrus Hermes Telegram operations overlay

This directory captures a sanitized deployment overlay for running Hermes on a
Mikrus/VPS host with Telegram as the primary control surface.

It intentionally does not include `.env`, API keys, Telegram bot tokens, SSH
credentials, Composio auth state, or provider secrets.

## What This Overlay Covers

- Persistent Hermes data mounted from the host into `/opt/data`.
- A local Docker image that pins `firecrawl-py==4.17.0` and installs
  `faster-whisper` for local Polish speech-to-text.
- Safe in-container `docker` and `docker-compose` shims, so Telegram terminal
  commands can restart or inspect Hermes without exposing the host Docker
  daemon socket.
- `hermesctl` for status, restart, logs, model tests, and Firecrawl checks.
- `facebook_public_search` for public indexed Facebook results without login,
  cookies, CAPTCHA bypasses, or copied browser sessions.
- `composio` CLI mounting through the terminal tool, with auth/config persisted
  under `/opt/data`.
- Telegram prompt hints that tell Hermes to present Composio as a terminal
  capability and to avoid approval-gate endings when concrete commands can be
  given.

## Host Layout

The production layout this mirrors is:

```text
/root/hermes/
  Dockerfile
  docker-compose.yml

/root/.hermes/
  .env                  # secrets, not committed
  config.yaml           # runtime config, keep secrets out
  AGENTS.md
  .local/bin/hermesctl
  .local/bin/docker
  .local/bin/docker-compose
  .local/bin/facebook_public_search
  .local/bin/composio
  composio-cli/         # installed Composio CLI bundle
```

## Deploy

Copy or sync these files to the matching host paths, then build and restart:

```bash
mkdir -p /root/hermes /root/.hermes/.local/bin
cp Dockerfile /root/hermes/Dockerfile
cp docker-compose.yml /root/hermes/docker-compose.yml
cp AGENTS.md /root/.hermes/AGENTS.md
cp scripts/* /root/.hermes/.local/bin/
chmod +x /root/.hermes/.local/bin/*

docker compose -f /root/hermes/docker-compose.yml build hermes
docker compose -f /root/hermes/docker-compose.yml up -d hermes
```

Install Composio CLI into `/root/.hermes/composio-cli` before relying on the
`composio` wrapper.

## Config Snippet

Merge `config.example.yaml` into `/root/.hermes/config.yaml`. Keep API keys and
tokens in `/root/.hermes/.env` only.

After changing config, clear cached Telegram system prompts if old instructions
keep appearing:

```bash
python3 - <<'PY'
import sqlite3
con = sqlite3.connect('/root/.hermes/state.db')
cur = con.cursor()
cur.execute("update sessions set system_prompt=NULL where source='telegram'")
print('cleared', cur.rowcount)
con.commit()
con.close()
PY
docker compose -f /root/hermes/docker-compose.yml restart hermes
```

## Verification

Run these on the host:

```bash
docker exec hermes sh -lc 'hermesctl status'
docker exec hermes sh -lc 'hermesctl model-test'
docker exec --user 10000:10000 hermes sh -lc 'composio whoami && composio connections list'
docker exec hermes sh -lc 'facebook_public_search "Brzeg Dolny wydarzenia" --limit 3'
```

Expected high-level results:

- Telegram platform is `connected`.
- `hermesctl model-test` prints `OK`.
- `composio whoami` shows the logged-in Composio account.
- `facebook_public_search` returns public indexed Facebook URLs or a clear
  no-results message without asking for a Facebook login.
