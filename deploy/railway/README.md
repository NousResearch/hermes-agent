# Employee deployment

This directory prepares a deployment; it does not create a Railway project.
No server login is needed to develop or run the local protocol tests.
The deployment runs Linux; local development supports macOS and Linux/WSL2.
Native Windows is unsupported because the responsibility filesystem uses POSIX
directory operations. Use WSL2 for this fork on Windows.

## Services

Create three services in one Railway environment when ready:

| Service | Build/config | Persistent storage | Public ports |
| --- | --- | --- | --- |
| `hermes` | Repository Dockerfile; `deploy/railway/hermes.toml` | `/opt/data` | Dashboard `9119`; webhook listener `8648` on a separate domain |
| `hindsight` | `deploy/railway/hindsight/Dockerfile`; `deploy/railway/hindsight.toml` | Database below | None |
| `postgres` | PostgreSQL with pgvector installed | PostgreSQL data directory | None |

Use one replica of each. The Hermes image uses native s6 supervision for its
profile gateway and dashboard, plus the private Codex inference service on
port `8879`. Do not publish port `8879`. Hindsight listens privately on `8888`.
The PostgreSQL image must support `CREATE EXTENSION vector`; an ordinary image
without pgvector is insufficient. See [pgvector's Docker instructions](https://github.com/pgvector/pgvector#docker).

Configure Railway's config-file path for each repository service as shown above.
Private DNS names assume the services are named `hermes` and `hindsight`.
If renamed, change `hindsight.url` in Hermes config and `HINDSIGHT_CODEX_URL`
on Hindsight. Both listeners bind IPv6 for Railway private networking.
See [private networking](https://docs.railway.com/networking/private-networking)
and [persistent volumes](https://docs.railway.com/volumes).

## Credentials and initial settings

Railway variables own infrastructure credentials:

| Variable | Service | Purpose |
| --- | --- | --- |
| `HERMES_DASHBOARD=1` | Hermes | Start native dashboard |
| `HERMES_GATEWAY_BOOTSTRAP_STATE=running` | Hermes | Start the gateway on a fresh volume |
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Hermes | Shared administrator username |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | Hermes | Strong shared administrator password |
| `HINDSIGHT_API_KEY` | Both | Same randomly generated private Hindsight API secret |
| `HINDSIGHT_INFERENCE_KEY` | Both | A separate random secret for private Codex inference |
| `OPENROUTER_API_KEY` | Hindsight | Qwen embeddings and Cohere reranking |
| `DATABASE_URL` | Hindsight | Private PostgreSQL connection string |

Use native dashboard-managed profile secrets for `TELEGRAM_BOT_TOKEN`,
`BROWSER_USE_API_KEY`, `PARALLEL_API_KEY`, and Hermes' `OPENROUTER_API_KEY`
(video analysis). Set the Telegram allowlist before using the bot. Do not also
set these variables in Railway: two competing credential sources make rotation
confusing. Infrastructure secrets in the table are rotated in Railway, on both
services together where applicable. At boot, Hermes copies its Railway-owned
`HINDSIGHT_API_KEY` into the boot profile's native secret store so scoped turns
can authenticate. Additional profiles must explicitly configure their own key;
they never inherit the process credential.

On first boot only, `config.yaml` is seeded from `deploy/railway/config.yaml`.
Subsequent boots retain administrator edits. The Config editor controls employee
name/instructions, the Codex main model, owner identity, group/topic policy,
webhook public URL, and native display preferences. Fixed tool/memory/review
rules live in code. File tools cannot rewrite config, auth or product guides.
This is not a terminal sandbox.

For local CLI personal memory, set `employee.owner` to the human's platform
identity, for example `telegram:123456789`. `employee.identity_links` explicitly
maps additional platform identities to that canonical identity; names never
merge people automatically.

## Sign in on the server

Open a terminal **inside the deployed Hermes container**, then run:

```sh
hermes auth add openai-codex
```

Complete the native sign-in flow. Its auth store stays on `/opt/data`; never
copy a rotating token from the development machine. Main inference and native
Codex image generation use this store. The private Hindsight endpoint resolves
and refreshes the same store through native Hermes locking. It accepts only the
configured Luna model and has no paid API fallback.

Hindsight takes its complete policy from the checked-in reference snapshot.
Only endpoints, authentication, database location and worker identity change.
Existing banks are reconciled using the copied managed-bank reconciler on every
Hindsight start; new banks inherit the template. Reconciliation errors are logged.
Browser sessions provision/reuse a durable cloud profile ID under the Hermes
volume. The first boot provisions the Browser Use CLI through native package management;
subsequent boots reuse it. Local Whisper uses faster-whisper; its model downloads on first use.

## Deployment acceptance (requires the future server)

1. Confirm dashboard password login, gateway startup and private service health.
2. Sign into Codex; run one main-model request and one image generation.
3. Retain a fact in Hindsight, wait for async processing, then recall and reflect.
   Confirm Luna entitlement and structured Responses compatibility on the real account.
4. Open a Browser Use session, establish a harmless login, close and reconnect;
   confirm it survives with the saved cloud profile. Transcribe an audio message locally.
5. Configure a Telegram group/topic in the Config editor. Create a responsibility
   and guarded schedule; verify one execution, delivery and next-turn report context.
6. Set `webhook.public_url`; create a signed declaration, register its returned URL,
   send/retry an event, then archive the package and verify ingress stops.
7. Redeploy and restore from backups: preserve `/opt/data` and PostgreSQL separately.
   Restore both to an isolated environment before directing real traffic there.

The pinned Hindsight image was built and started against a disposable local
pgvector database; migrations and health checks passed. The account and Railway
checks above have not been run: no Railway project exists yet. Local HTTP
protocol tests validate the code paths, not subscription entitlement or hosted
service availability.
