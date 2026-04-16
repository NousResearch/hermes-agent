# Hermes Agent on Coolify + Nebius Token Factory

Self-hosted Hermes Agent deployed via Coolify, using Nebius Token Factory
as the inference backend. The agent runs as a long-lived gateway service
with full tool access (cron scheduling, shell/python execution, file ops,
web, browser, skills, todo) so it can self-direct and fire scheduled tasks
unattended.

## Design — why a separate directory?

Everything that makes this a Coolify deployment lives in `deploy/coolify/`.
The upstream Dockerfile, entrypoint, and Python source tree are **not
modified**. That means:

- `git merge upstream/main` always fast-forwards cleanly — you can pull in
  new Hermes features as they land upstream without resolving conflicts.
- Your own modifications elsewhere in the repo are left untouched.
- The Coolify container builds from the same upstream image the Nix
  container and Docker quickstart use.

## What you get

| Capability | Enabled by |
|---|---|
| Cron / scheduled tasks | `cronjob` toolset + `hermes gateway run` scheduler |
| Shell + Python execution | `terminal` toolset, YOLO mode |
| File ops, web scraping, browser | `file` + `web` + `browser` toolsets |
| Skills (self-authored tools) | `skills` toolset, persisted in volume |
| Long-term memory + session history | `/opt/data` persistent volume |
| Auto-failover on Nebius errors | `fallback_model:` (opt-in via `HERMES_FALLBACK_MODEL`) |

## Model choices

Default primary: **`MiniMaxAI/MiniMax-M2.5`** — post-trained for
interleaved-thinking tool calls and long multi-step coding/office
workflows. Strongest agent-use fit in the Nebius Token Factory catalog.

Fallback: **off by default**. Set `HERMES_FALLBACK_MODEL` in Coolify to a
verified Nebius slug to enable one-shot auto-retry on rate-limit / 5xx.

Override either in Coolify's env vars UI:

| `HERMES_PRIMARY_MODEL` | When to pick |
|---|---|
| `MiniMaxAI/MiniMax-M2.5` *(default)* | Best agentic tool-calling reliability |
| `nvidia/Nemotron-3-Super-120b-a12b` | 1M context + 127 Tok/s; good balance of speed & reasoning |
| `deepseek-ai/DeepSeek-V3.2` | Cheapest strong reasoning ($0.30/$0.45) |
| `moonshotai/Kimi-K2.5` | Native multimodal agent training |
| `NousResearch/Hermes-4-405B` | Canonical pairing (same team), but 20 Tok/s + $1/$3 |

Confirm the exact slug on the model's page in the Nebius console before
setting it — the strings above are best-guesses based on Nebius's naming
convention and may need tweaking.

## Deploy

### 1. Add the Nebius key to Coolify — do NOT paste it into the repo

1. In Coolify, open your application → **Environment Variables**.
2. Add `NEBIUS_API_KEY`, paste the token, tick **Is Secret**.
3. Add any messaging tokens you want (Telegram, Discord, Slack) the same way.

### 2. Create the resource

1. **Resources → New → Application → Docker Compose**.
2. Repository: this repo. Branch: the one you're deploying (e.g.
   `claude/coolify-nebius-setup-V5eS1`).
3. **Base Directory**: `/deploy/coolify`.
4. **Compose file**: `docker-compose.yml` (default).
5. Persistent storage: Coolify picks up the named `hermes-data` volume
   automatically from compose.
6. Deploy.

### 3. Verify

```bash
# From the Coolify terminal button, or `docker exec`:
docker exec -it hermes-agent bash
# Inside the container:
hermes status
hermes doctor
hermes cron list   # should work immediately — scheduler is live
```

Create a test cron from a chat session (Telegram / Discord / Slack):

> "Every Monday at 09:00, run `uptime` and post the result here."

## Updating — pull in new Hermes features

```bash
# One-time: wire up upstream
git remote add upstream https://github.com/NousResearch/hermes-agent.git

# Sync:
git fetch upstream
git checkout main
git merge upstream/main        # fast-forward — deploy/coolify untouched
git push origin main

# Redeploy in Coolify (or let its auto-deploy on push do it for you).
```

Your modifications elsewhere in the repo and the `deploy/coolify/` directory
are preserved across upstream syncs. If you've edited upstream-owned files
yourself, resolve those conflicts as you would normally — none of them
overlap with this directory.

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Coolify service definition: build context, volume, env, healthcheck |
| `entrypoint.sh` | Bind-mounted wrapper: templates `config.yaml` from env, then hands off to the upstream entrypoint |
| `config.template.yaml` | `~/.hermes/config.yaml` generated at first boot — Nebius + fallback + toolsets + approvals |
| `.env.example` | Reference list of env vars Coolify needs (do not commit real values) |

## Security notes

- `NEBIUS_API_KEY` lives only in Coolify's secret store and in
  `/opt/data/.env` (chmod 600, on the private named volume). The repo
  itself never contains the key.
- The container runs as non-root uid 10000 (`hermes`), sandboxed by Docker.
  YOLO mode skips approval prompts because the container isolates any
  damage to `/opt/data`.
- If you front the agent with a public messaging platform, always set
  `TELEGRAM_ALLOWED_USERS` / `DISCORD_ALLOWED_USERS` / the equivalent —
  otherwise anyone who finds the bot can issue commands.

## Tweaking the running config

Three levels of control, most-to-least permanent:

1. **Edit `config.template.yaml`**, commit, set
   `HERMES_FORCE_CONFIG_REWRITE=1` in Coolify, redeploy once, then unset
   the flag. Regenerates the live `config.yaml` from the template.
2. **`docker exec -it hermes-agent bash`**, then `hermes config set …`
   or edit `/opt/data/config.yaml` directly. Survives restarts, lost on
   `HERMES_FORCE_CONFIG_REWRITE=1`.
3. **Per-session overrides**: `hermes chat --model …` inside a chat
   session, or `/model` slash commands.
