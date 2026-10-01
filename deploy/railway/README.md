# Employee deployment

This directory contains the runtime configuration for a Railway deployment.
No server login is needed to develop or run the local protocol tests.
The deployment runs Linux; local development supports macOS and Linux/WSL2.
Native Windows is unsupported because the responsibility filesystem uses POSIX
directory operations. Use WSL2 for this fork on Windows.

## Services

Create three services in one Railway environment when ready:

| Service | Build/config | Persistent storage | Public ports |
| --- | --- | --- | --- |
| `hermes` | Repository `Dockerfile`; start command `sleep infinity` | `/opt/data` | Dashboard `9119`; webhook listener `8648` on a separate domain |
| `hindsight` | `deploy/railway/hindsight/Dockerfile`; image's default start command | Database below | None |
| `postgres` | PostgreSQL with pgvector installed | PostgreSQL data directory | None |

Use one replica of each. The Hermes image uses native s6 supervision for its
profile gateway and dashboard, plus the private Codex inference service on
port `8879`. Do not publish port `8879`. Hindsight listens privately on `8888`.
The PostgreSQL image must support `CREATE EXTENSION vector`; an ordinary image
without pgvector is insufficient. See [pgvector's Docker instructions](https://github.com/pgvector/pgvector#docker).

Set each service's Dockerfile path and start command in Railway's service settings
as shown above. Railway rejects the old `railwayConfigFile` setting; no TOML
deployment config is used. Keep one replica, disable service sleeping, and select
the on-failure restart policy with 10 retries. Mount volumes explicitly before
the first deployment; the Hermes Dockerfile does not create an anonymous volume.
For Hermes, set `PORT=9119` and the healthcheck path to `/api/health` with a
300-second timeout. For Hindsight, use `PORT=8888` and `/health` after Hermes is
available. These probes check service startup, not model access. Enable daily
and weekly Railway backups on both persistent volumes.
Private DNS names assume the services are named `hermes` and `hindsight`.
If renamed, change `hindsight.url` in Hermes config and `HINDSIGHT_CODEX_URL`
on Hindsight. Both listeners bind IPv6 for Railway private networking.
See [private networking](https://docs.railway.com/networking/private-networking)
and [persistent volumes](https://docs.railway.com/volumes).

## Credentials and initial settings

Railway variables own infrastructure credentials. An optional `OPENROUTER_API_KEY`
on the Hermes service seeds its profile store once; later dashboard edits win.
Supply it on Hermes, not only on Hindsight.

Railway variables:

| Variable | Service | Purpose |
| --- | --- | --- |
| `HERMES_DASHBOARD=1` | Hermes | Start native dashboard |
| `HERMES_GATEWAY_BOOTSTRAP_STATE=running` | Hermes | Start the gateway on a fresh volume |
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Hermes | Shared administrator username |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | Hermes | Strong shared administrator password |
| `HINDSIGHT_API_KEY` | Both | Same randomly generated private Hindsight API secret |
| `HINDSIGHT_INFERENCE_KEY` | Both | A separate random secret for private Codex inference |
| `DATABASE_URL` | Hindsight | Private PostgreSQL connection string |

Use the hosted **Service keys** page for native profile secrets for `TELEGRAM_BOT_TOKEN`,
`BROWSER_USE_API_KEY`, `PARALLEL_API_KEY`, and `OPENROUTER_API_KEY`
(video analysis, Qwen embeddings and Cohere reranking). Set the Telegram allowlist before using the bot. Do not also
set these variables in Railway: two competing credential sources make rotation
confusing. Infrastructure secrets in the table are rotated in Railway, on both
services together where applicable. At boot, Hermes copies its Railway-owned
`HINDSIGHT_API_KEY` into the boot profile's native secret store so scoped turns
can authenticate. Additional profiles must explicitly configure their own key;
they never inherit the process credential.

On first boot only, `config.yaml` is seeded from `deploy/railway/config.yaml`.
Subsequent boots retain administrator edits. Open the dashboard root or `/settings`
for profile instructions, models, Telegram access, service keys and shared-admin
password changes. Login variables bootstrap the account; a password saved in the
UI takes precedence on later boots. Advanced native pages remain available for
owner identity, webhook public URL and other settings. Fixed tool/memory/review
rules live in code. Native configuration remains mutable; this is not a terminal sandbox.

For local CLI personal memory, set `employee.owner` to the human's platform
identity, for example `telegram:123456789`. `employee.identity_links` explicitly
maps additional platform identities to that canonical identity; names never
merge people automatically.

## Sign in on the server

Open **Models → Connect Codex** in the deployed settings page. Complete the native
device-code flow in your browser. The auth store stays on `/opt/data`; never copy
a rotating token from the development machine. Chat, image generation and the
private Hindsight endpoint use the same native credential owner and refresh lock.
The private endpoint accepts only the selected memory model, without a paid API fallback.

Hindsight takes its complete policy from the checked-in reference snapshot.
Only endpoints, authentication, database location, worker identity and the three
UI memory-model settings change. The supervisor reads these settings and the
OpenRouter key from Hermes over the private authenticated inference port, then
restarts its children when the revision changes. The UI reports the applied
revision’s service health. It never receives the private credentials.
Existing banks are reconciled using the copied managed-bank reconciler on every
Hindsight start; new banks inherit the template. Reconciliation errors are logged.
Browser Use follows native session and account behavior. First boot provisions
the CLI through native package management. Local Whisper includes the native
multilingual base model in the image; first boot seeds its native cache.

## Deployment acceptance

1. Confirm dashboard password login, gateway startup and private service health.
2. Sign into Codex; run one main-model request and one image generation.
3. Retain a fact in Hindsight, wait for async processing, then recall and reflect.
   Confirm Luna entitlement and structured Responses compatibility on the real account.
4. Open a Browser Use session, establish a harmless login, close and reconnect;
   verify the account's native persistence behavior. Transcribe an audio message offline.
5. Configure a Telegram group/topic on the Access page. Create a responsibility
   and guarded schedule; verify one execution, delivery and native conversation history.
6. Set `webhook.public_url`; create a signed declaration, register its returned URL,
   send/retry an event, then archive the package and verify ingress stops.
7. Redeploy and restore from backups: preserve `/opt/data` and PostgreSQL separately.
   Restore both to an isolated environment before directing real traffic there.

The pinned Hindsight image was built and started against a disposable local
pgvector database; migrations and health checks passed. The account-dependent
checks above must pass on the deployed profile. Local HTTP
protocol tests validate the code paths, not subscription entitlement or hosted
service availability.

## Native runtime and bundled transcription

Browser Use uses the native implementation selected by this deployment's
config. The fork does not provision or guarantee a durable cloud browser profile.
Native CLI/configuration and administration endpoints remain available.

The image installs `stt-whisper` and downloads the multilingual `base` weights
at build time. First boot copies that repository to the native Hugging Face
cache as the Hermes user, only if absent. The seeded `stt.local.model: base`
loads offline from that cache. Selecting another model uses native download
behavior. No model file or credential belongs in Git.
