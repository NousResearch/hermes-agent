# Hindsight production template for Angie Hermes

This is a production-shaped template for Mode B/C of the Angie Hermes migration. It is not enabled by default and must not be copied to production until Joe approves the exact production mutation scope.

## Properties

- DB and API services are separate.
- API port binds to `127.0.0.1` only.
- Runtime data uses bind-mounted `./data/...` directories so backup/restore is explicit.
- `.env.example` contains placeholders only.
- The `.env` created on production must be owner-only (`0600`) and treated as secret-bearing.

## Prepare on production after approval

```bash
set -euo pipefail
cd /home/angie/.hermes
mkdir -p hindsight-compose
cd hindsight-compose
umask 077
install -m 0600 /dev/null .env
# Copy compose.yaml and .env.example from the approved repo artifact, then edit .env manually.
```

## Start

```bash
set -euo pipefail
cd /home/angie/.hermes/hindsight-compose
docker compose --env-file .env up -d
```

## Health

```bash
docker compose ps
curl -fsS http://127.0.0.1:${HINDSIGHT_API_PORT:-18080}/health || curl -fsS http://127.0.0.1:${HINDSIGHT_API_PORT:-18080}/
```

## Hermes smoke

Before Mode B/C, verify the actual installed Hermes Hindsight provider contract from the code/tests in that deployed version: `memory.provider` value, local API URL key, health/version endpoint, and non-sensitive retain/recall smoke command. If any part is unknown, keep go-live at Mode A and disable/fail-open memory.

## Stop / fallback

```bash
cd /home/angie/.hermes/hindsight-compose
docker compose stop api
# If memory blocks Slack runtime, set Hermes memory provider to builtin/disabled per approved rollback sheet and restart gateway only after approval.
```

## Backup

Hindsight data is sensitive user-memory data. Back it up only into the approved owner-only backup root, never into Git, Notion, Slack, or shared web paths.

```bash
set -euo pipefail
cd /home/angie/.hermes/hindsight-compose
docker compose stop api
mkdir -p "$BACKUP/dynamic/hindsight"
cp -a ./compose.yaml ./.env ./data "$BACKUP/dynamic/hindsight/"
docker compose start api
```

## Restore rehearsal

Restore into a temporary directory first and verify layout before restoring over production.

```bash
set -euo pipefail
RESTORE_TEST=/home/angie/hermes-restore-tests/hindsight-$TS
mkdir -p "$RESTORE_TEST"
cp -a "$BACKUP/dynamic/hindsight"/. "$RESTORE_TEST"/
test -f "$RESTORE_TEST/compose.yaml"
test -f "$RESTORE_TEST/.env"
test -d "$RESTORE_TEST/data"
```
