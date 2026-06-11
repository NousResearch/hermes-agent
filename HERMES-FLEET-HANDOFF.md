# Hermes multi-tenant fleet — deployment status & integration handoff

## What this is
A single Railway service running native Hermes (NousResearch/hermes-agent) that
hosts ONE Hermes profile per Avocado customer. Each profile = one Telegram bot,
pointed at that customer's own Avocado account via the Avocado MCP. Concierge
provisioning (by hand) is proven; self-serve (HTTP) is the next build.

## Infra (DONE — deployed and running)
- Fork: github.com/Wandersonjack/hermes-agent-avocado (main)
- Railway project: 6a99c622-2535-4283-88dc-aec51d734801, service "hermes-agent-avocado"
- Auto-deploy on push to main is ON.
- Persistence: Railway Volume mounted at /opt/data (= HERMES_HOME). Holds every
  customer's profile, memory, and state across restarts/redeploys.
- Service-level env: HERMES_HOME=/opt/data, AUTO_UPDATE=false. NO API keys at
  service level (see "Keys" below).
- Polling mode = outbound-only. No public domain, port, or webhook. No healthcheck.

## Key fixes made to get it deploying (all on main)
1. Synced fork v0.14.0 -> v0.16.0 (clean fast-forward). v0.14.0 used tini and
   LACKED the s6 per-profile supervision the multi-tenant model needs. 0.16.0
   ships s6-overlay /init (PID 1), docker/s6-rc.d/, cont-init.d/02-reconcile-profiles,
   and hermes_cli/service_manager.py.
2. Removed Docker `VOLUME ["/opt/data"]` from the Dockerfile — Railway rejects
   the VOLUME directive outright. Persistence is the Railway Volume instead.
3. Baked `CMD ["gateway","run"]` into the Dockerfile and removed any Railway
   custom start command. A Railway startCommand OVERRIDES the ENTRYPOINT, bypassing
   /init (s6) and failing with "executable `gateway` could not be found". The image
   must run its own ENTRYPOINT (/init -> main-wrapper.sh -> `hermes gateway run`).
4. provision-in-container.sh: chowns written config to hermes when run as root
   (railway ssh lands as root); OPENROUTER_API_KEY now optional (shared-key beta).
5. hermes_cli/container_boot.py: auto-start ALL registered profiles on container
   boot unless a `.paused` marker exists (upstream only revives state "running",
   which leaves every bot down after any redeploy — see Validation #4).

## How it runs (architecture)
- /init (s6-overlay) is PID 1. It runs cont-init hooks, starts s6-rc services,
  then execs the container CMD (`gateway run`) as the "main program" via
  docker/main-wrapper.sh, which routes it to `hermes gateway run` and drops to
  the unprivileged `hermes` user.
- The default profile's `gateway run` just idles (no bot token = no Telegram loop,
  no reasoning calls, no keys needed).
- s6 `main-hermes` service is intentionally a no-op (sleep infinity); the dashboard
  s6 service binds 127.0.0.1 only (not exposed).
- Per-customer profiles live at /opt/data/profiles/<slug>/ with config.yaml + .env.
- `hermes -p <slug> gateway start` registers an s6-supervised gateway slot at
  runtime (service_manager detects s6). On every container boot,
  cont-init.d/02-reconcile-profiles (-> hermes_cli/container_boot.py) auto-restarts
  profiles whose last recorded state was "running". This is why bots survive redeploys.

## Per-customer provisioning (concierge — PROVEN, pilot-1 live)
Script: /opt/hermes/provision-in-container.sh (in the image). Refuses to run unless
$RAILWAY_ENVIRONMENT is set (can never touch a local Hermes).

HOW TO GET A SHELL INSIDE THE RUNNING CONTAINER (this is the part the Railway docs
make confusing): use the Railway CLI's `railway ssh`, NOT a Railway Function (fresh
Bun container — no image, no script), NOT the Railway chat-agent sandbox, NOT your
laptop. `railway ssh` opens a session inside the live deployment where the script
exists and $RAILWAY_ENVIRONMENT is set:

    railway ssh --project <PROJECT_ID> --service hermes-agent-avocado \
      --environment production \
      'SLUG=<slug> TELEGRAM_BOT_TOKEN=<...> TELEGRAM_USER_ID=<numeric> \
       AVOCADO_MCP_KEY=<avk_...> OPENROUTER_API_KEY=<sk-or-...capped> \
       sh /opt/hermes/provision-in-container.sh'

GOTCHA — `railway ssh` lands you as ROOT, not the hermes user. The script writes
config.yaml + .env, and (as of the chown patch) chowns them to hermes:hermes so the
supervised gateway (UID 10000) can read them. Without that chown you get
`PermissionError: /opt/data/profiles/<slug>/.env` at gateway start. If you ever see
that on an older script, fix with:
    railway ssh ... 'chown -R hermes:hermes /opt/data/profiles/<slug> && hermes -p <slug> gateway start'

What the script does: creates the profile, writes config.yaml (Avocado MCP scoped to
the customer's key, manual approvals, cron deny, safe toolset: image_gen/vision/tts/
web/memory/session_search/messaging/clarify/todo), writes .env (OpenRouter key + bot
token + TELEGRAM_ALLOWED_USERS allowlist), chowns to hermes, and starts the gateway.
Idempotent. On success the gateway registers a dynamic s6 service slot at
/run/service/gateway-<slug>, so it's supervised and survives container restarts.

## Validation checklist (Part C)
1. Bot replies on Telegram from the customer's account. ✅ proven (pilot-1)
2. Image generates to the customer's Avocado account (their credits drop, not yours).
3. A different Telegram account is ignored (allowlist holds).
4. Survives a service restart/redeploy. ❌ FAILED 2026-06-12 on stock upstream, then
   FIXED by fork patch: a redeploy SIGTERMs every profile gateway → gateway_state.json
   records "stopped" → upstream's 02-reconcile-profiles only auto-starts state
   "running" → ALL customer bots stay down after EVERY redeploy. Fork patch in
   hermes_cli/container_boot.py: auto-start every registered profile unless a
   ``.paused`` marker file exists in the profile dir. Pause SOP:
   ``hermes -p <slug> gateway stop && touch /opt/data/profiles/<slug>/.paused``;
   unpause: remove the marker + ``gateway start``. RE-VERIFY after upstream syncs.
5. (Optional, paid model only) A $0.10 test-capped OpenRouter key, once exhausted,
   hard-stops the agent instead of continuing to spend. N/A under the shared-key beta.

## Keys / cost model (IMPORTANT for the Avocado integration)
- BETA MODEL (since 2026-06-12): ONE SHARED OpenRouter key for the whole fleet, set as
  the Railway service variable OPENROUTER_API_KEY (with a monthly credit limit set on
  the key itself in OpenRouter — that limit is the global blast-radius cap). Profiles
  inherit it via s6 with-contenv; the provisioner no longer requires OPENROUTER_API_KEY.
  Consequences: no per-customer spend ceiling (one heavy user can drain the shared
  budget; mitigations: agent.max_turns=40, cheap model, trusted beta users only) and
  no per-customer cost attribution at OpenRouter (recover later from per-profile
  state.db usage if needed).
- PAID/SELF-SERVE MODEL (deferred): per-customer OpenRouter key with a ~$10/mo hard
  cap, passed explicitly to the provisioner (still supported — a key passed via
  OPENROUTER_API_KEY is written to the profile .env and overrides the shared one).
- Reasoning model default: xiaomi/mimo-v2.5-pro (verified live OpenRouter slug,
  1M ctx, ~$0.44/$0.87 per M tok in/out). Override per customer with MODEL=.
- Image/video/audio generation goes through the customer's Avocado MCP key
  (https://www.avocadoai.co/api/mcp, Bearer auth) — scopes all generations +
  credits to THEIR Avocado account. No FAL/Anthropic/ElevenLabs keys in Hermes.
- Allowlist: TELEGRAM_ALLOWED_USERS locks each bot to the customer's own Telegram ID.

## NEXT BUILD on the Avocado side — self-serve provisioning (Part D)
Turn the manual shell command into a customer-facing button:
1. Add provisioner/server.py to the fork as a supervised s6 service (not yet present),
   guarded by PROVISIONER_SECRET, exposed only on Railway's PRIVATE network.
2. Avocado backend endpoint POST /api/hermes/provision that:
   - mints an Avocado MCP key scoped to the customer's account,
   - creates a $10-capped OpenRouter key (OpenRouter provisioning API),
   - calls the Hermes sidecar over the private network with SLUG + the 4 secrets.
3. Supabase registry table: (customer_id, slug, telegram_user_id, bot_username,
   openrouter_key_id, avocado_mcp_key_id, status, created_at).
4. UI: "Connect your Telegram agent" — customer pastes their BotFather token +
   numeric Telegram ID; backend does the rest. Same engine as the shell command,
   driven over HTTP.

## Open items / gotchas
- provisioner/server.py does NOT exist yet — build it for Part D.
- Keep AUTO_UPDATE=false so a paid customer's bot never breaks from an unattended
  upstream pull. Re-sync to upstream deliberately, then re-test the 3 fork patches
  above (VOLUME removal, CMD, model default) since they diverge from upstream.
- Never expose the dashboard/API server publicly per customer — customers use
  Telegram only.

## Repo files specific to this deploy
- Dockerfile          — patched: no VOLUME directive; CMD ["gateway","run"].
- railway.json        — DOCKERFILE builder; restart ON_FAILURE; NO startCommand.
- provision-in-container.sh — concierge per-customer provisioner (Railway-only guard).
- HANDOFF-DEPLOY.md    — original deploy runbook (STEP 0 sync, Parts A-D).
- HERMES-FLEET-HANDOFF.md — this file.
