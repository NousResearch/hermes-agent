# Postiz VPS Compose

This directory contains Sahil's self-hosted Postiz Docker Compose setup in repo-safe form.

It is intended to replace the older live working directory at:

`/home/kensei/apps/postiz-docker`

Do not delete the old directory until this repo version has been used to recreate the running stack and verified.

## Files

- `docker-compose.yaml` — base self-hosted Postiz stack, including Postiz, Redis, Postgres, Temporal, Temporal UI, Elasticsearch and Spotlight.
- `docker-compose.dev.yaml` — upstream dev overlay retained for compatibility.
- `patch-and-start.sh` — runtime patch applied inside the Postiz container before PM2 starts.
- `dynamicconfig/` — Temporal dynamic config required by the compose file.
- `.env.example` — required local environment variables without secrets.
- `docker-compose.override.example.yml` — optional local override template.
- `.gitignore` — prevents local `.env`, overrides and local patches being committed.

## Runtime patch

`patch-and-start.sh` does two local self-hosting fixes:

1. Removes LinkedIn `prompt=none` and reduces LinkedIn personal-profile scopes to `openid`, `profile`, and `w_member_social`.
2. Blocks `/api/copilot/chat` at nginx with a fast `503` JSON response, avoiding upstream timeouts while stock Postiz does not have proven OpenAI-compatible provider routing.

The base compose mounts this script and starts Postiz via:

`sh /patch-and-start.sh`

So a fresh clone works with a `.env` only.

## First run

```bash
cd /home/kensei/repos/KenseiAgent/postiz
cp .env.example .env
openssl rand -hex 32
# Put the generated value into JWT_SECRET in .env
# Fill OAuth credentials only for platforms you are actively connecting.
docker compose config >/tmp/postiz-compose-check.yml
docker compose up -d
```

Postiz is exposed at:

`http://localhost:4007`

Sahil normally accesses it through an SSH tunnel, not public internet exposure.

## Required .env values

Minimum viable local values:

```env
JWT_SECRET=replace-with-openssl-rand-hex-32
DATABASE_URL=postgresql://postiz-user:postiz-password@postiz-postgres:5432/postiz-db-local
POSTIZ_DB_USER=postiz-user
POSTIZ_DB_PASSWORD=postiz-password
POSTIZ_DB_NAME=postiz-db-local
NOT_SECURED=true
DISABLE_REGISTRATION=true
```

OAuth credentials are optional until each social platform is wired.

## Verification

```bash
docker compose config >/tmp/postiz-compose-check.yml
docker compose up -d
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -I http://127.0.0.1:4007/
curl -i -X POST http://127.0.0.1:4007/api/copilot/chat -H 'Content-Type: application/json' -d '{}'
docker exec postiz sh -c 'pm2 list'
```

Expected:

- `postiz`, `postiz-postgres`, `postiz-redis`, `temporal`, `temporal-postgresql`, `temporal-elasticsearch`, `temporal-ui`, `spotlight` are running.
- `http://127.0.0.1:4007/` redirects to `/auth` or loads the app.
- `/api/copilot/chat` returns immediate `503` JSON, not a timeout.
- PM2 shows backend, frontend and orchestrator online.

## Migration from old live directory

Current old live directory:

`/home/kensei/apps/postiz-docker`

Recommended migration:

1. Back up the old live `.env` and `docker-compose.override.yml` securely.
2. Copy secret values from the old override into this directory's ignored `.env`.
3. Run `docker compose config` from this directory.
4. Stop the old stack from the old directory.
5. Start from this directory.
6. Verify HTTP, PM2, integrations and the Copilot block.
7. Keep the old directory as a backup until this has survived a restart.

Do not commit `.env` or `docker-compose.override.yml`.
