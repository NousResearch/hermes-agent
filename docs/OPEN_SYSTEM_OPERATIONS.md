# Open-System production operations

This fork (`hillstreet-ph/open-system`) runs Hermes Agent in production from the
**compat image**: the official `nousresearch/hermes-agent:vX.Y.Z` release, pinned by
digest, plus the Open-System MCP tunnel and the `mcp-serve`, `spotify` and `langfuse`
plugins (`deploy/open-system-compat.Dockerfile`).

## Topology

| Item | Value |
|---|---|
| Runtime | Zeabur project `open-system-project`, service `hermes-agent-recovery` |
| Public dashboard / Desktop gateway | `https://open-system.space` (port 9119, login required) |
| Health endpoint | `https://open-system.space/api/status` |
| Persistent state | Zeabur volume `hermes-data` mounted at `/opt/data` (`HERMES_HOME`) |
| Image registry | Docker Hub `hillstreet/open-system:<version>-compat-<id>` (immutable tags) |
| Deployed release | `deploy/open-system.release` (written by the pipeline) |

Zeabur has announced it will shut down all services on **2026-12-31**. The pipeline
keeps the image portable (any Docker host with a volume at `/opt/data`); only the
deploy/heal steps call Zeabur.

## Automation

`.github/workflows/open-system-compat-validation.yml` (**Open-System release**)

1. Daily (06:43 Manila) checks Docker Hub for a newer upstream `vX.Y.Z`.
2. Bumps the Dockerfile version + digest, builds, and verifies tunnel, plugins,
   dependency drift and `hermes --version`.
3. Publishes an immutable image tag.
4. Takes a full `hermes backup` on the volume, switches the Zeabur image tag, and
   waits for `/api/status` to report the new version, a running gateway and a
   connected Telegram platform.
5. On failure it rolls back to the tag in `deploy/open-system.release` and opens an
   `incident` issue. On success it commits the bump + release record to `main`.

Pushes to `main` that touch the Dockerfile run the same build/validate/deploy path.
Branches named `chore/hermes-v*-compat` only build, validate and publish.

`.github/workflows/open-system-ops.yml` (**Open-System ops**)

- Health every 20 minutes; restarts the service via Zeabur if unhealthy and keeps a
  single `incident` issue updated (closed automatically on recovery).
- Daily backup (03:17 Manila) to `/opt/data/backups` (newest 7 kept) plus an
  AES-256 encrypted copy uploaded as a 30-day workflow artifact.

## Required settings (names only)

| Name | Kind | Purpose |
|---|---|---|
| `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | secret | publish the image |
| `ZEABUR_TOKEN` | secret | deploy, rollback, restart, backup (Zeabur → Settings → Developer → API keys) |
| `ZEABUR_SERVICE_ID` | variable | `6aab46350f0f0a129012d37e` (hermes-agent-recovery) |
| `ZEABUR_ENVIRONMENT_ID` | variable | production environment id of `open-system-project` |
| `BACKUP_PASSPHRASE` | secret | encrypts off-site backups (store it in your password manager too) |
| `OPEN_SYSTEM_STATUS_URL` | variable, optional | defaults to `https://open-system.space/api/status` |

Create a GitHub environment named `production` (Settings → Environments); the deploy
job runs in it, so required reviewers can be added there later if wanted.

## Restore

- From the volume: `hermes import /opt/data/backups/<file>.zip --force` in the service
  console, then restart the service.
- From an off-site artifact: download it from the ops run, then
  `gpg -d hermes-backup-*.zip.gpg > backup.zip` and `hermes import backup.zip --force`.
- New host: run the last image tag from `deploy/open-system.release` with a volume at
  `/opt/data`, restore the zip, and recreate the service variables.

## Manual rollback

Set the Zeabur service image tag to the `tag=` value from a previous commit of
`deploy/open-system.release`, or run the release workflow with `deploy` on an older
commit.

## Connecting Hermes Desktop (macOS)

Settings → Gateways → **Remote gateway**, URL `https://open-system.space`, sign in with
the dashboard credentials, then **Test** (checks both HTTP and WebSocket).
