#!/bin/bash
# Hermes post-rebuild config restore
# Usage: docker exec hermes bash /opt/data/restore-config.sh
# Or from host: ./roosync-cluster/scripts/hermes-restore-config.sh
#
# Secrets are NEVER hardcoded here. They come from:
#   1. /opt/data/.env.secrets (if present — copy from host .env.secrets before running)
#   2. Environment variables (Docker -e flags)

set -e
DATA="/opt/data"
SECRETS_FILE="$DATA/.env.secrets"

echo "Restoring Hermes deployment config..."

# 0. Load secrets from file if present, otherwise use env vars
if [ -f "$SECRETS_FILE" ]; then
    echo "  -> Loading secrets from $SECRETS_FILE"
    set -a
    source "$SECRETS_FILE"
    set +a
else
    echo "  -> No .env.secrets file, using environment variables"
fi

# 1. Overwrite model config — claudish proxy route (Phase 2 v3 restored 2026-08-07)
# Routes via claudish proxy (po-2023:3000): claude-sonnet-4-6 -> gc@glm-5.2.
# Auth via x-proxy-key (mandatory since claudish auth hardening 2026-07-10) carried
# in ANTHROPIC_CUSTOM_HEADERS. This is the prescribed fix from claudish (msg
# msg-20260807T185007-xvk0jc) after the z.ai-native fallback was rejected by user
# as not following claudish's instructions. GLM_API_KEY retained for aux tasks
# (compression, image, browser, web).
echo "  -> Setting model: claude-sonnet-4-6 (anthropic wire via claudish -> gc@glm-5.2)"
# Extended regex, optional quotes — robust to upstream stage2-hook format changes.
sed -i -E 's/^  default: "?(anthropic\/claude-opus-4\.6|claude-sonnet-4-6|glm-5\.2|glm-5-turbo)"?$/  default: "claude-sonnet-4-6"/' "$DATA/config.yaml"

# Ensure provider is set to anthropic (built-in profile, transport=anthropic_messages).
# x-proxy-key + ANTHROPIC_BASE_URL (step 4) route the gateway to claudish.
if grep -q '^  provider:' "$DATA/config.yaml"; then
    sed -i -E 's/^  provider: "?(auto|openrouter|anthropic|zai)"?$/  provider: "anthropic"/' "$DATA/config.yaml"
else
    sed -i '/^  default: "claude-sonnet-4-6"/a\  provider: "anthropic"' "$DATA/config.yaml"
fi

# 1c. Set compression threshold for GLM-5.2 1M context.
# Default upstream is 0.5 (= 500k on 1M, compresses too late). We pin to 0.24
# so summarization triggers at ~250k tokens, leaving headroom while letting
# long coordination sessions use most of the 1M window before compacting.
if command -v yq >/dev/null 2>&1; then
    yq -i '.compression.threshold = 0.24' "$DATA/config.yaml"
else
    sed -i 's/^  threshold: 0.5/  threshold: 0.24/' "$DATA/config.yaml"
fi
echo "  -> Compression threshold: 0.24 (~250k on 1M context)"

# 1a. api_max_retries: 3 -> 7 (backoff crest-riding, NOT max-absorb).
# z.ai HTTP 429 code 1305 "service overloaded" = sustained crest (40-44/h
# nocturne). Existing jittered_backoff (base 2s, cap 60s) is correct but
# with 5 retries abandons after ~62s BEFORE reaching the cap -> run dies,
# coordinator blind 2h. 7 lets retries reach the 60s cap (attempts 6-7 wait
# ~60-90s, matching the ~82s inter-429 gap) = genuine back-off that rides
# the crest. NOT hammering: later retries SPACE OUT (60s) not quick-fire.
# Model downgrade rejected (glm-5.1 superseded by 5.2).
if command -v yq >/dev/null 2>&1; then
    yq -i '.agent.api_max_retries = 7' "$DATA/config.yaml"
else
    sed -i 's/^  api_max_retries: [0-9][0-9]*/  api_max_retries: 7/' "$DATA/config.yaml"
fi
echo "  -> api_max_retries: 7 (reach 60s backoff cap, ride overload crests)"


# 1d. Patch jittered_backoff call site for HTTP 429 (conversation_loop.py:3439).
# 2026-06-25: z.ai hardened concurrent access quota. Upstream default
# base_delay=2.0s, max_delay=60.0s is too tight — each Hermes request retries
# too fast under saturation, adding concurrent pressure on the very pool that
# is overloaded. Patched to base_delay=5.0s, max_delay=120.0s — aligns with
# the in-code _retry_after cap (120s) and the invalid-response site at :1357
# which already uses 5.0/120.0. Combined with api_max_retries=7 (step 1a):
# retries now 5->10->20->40->80->120->120s ~= 6.5 min total back-off, well
# below the 30 min cron gateway_timeout. Idempotent: skips if already patched.
echo "  -> Checking conversation_loop.py backoff tuning (line 3439)"
CONV_LOOP="/opt/hermes/agent/conversation_loop.py"
if [ -f "$CONV_LOOP" ]; then
    if grep -q 'jittered_backoff(retry_count, base_delay=2\.0, max_delay=60\.0)' "$CONV_LOOP"; then
        sed -i 's|jittered_backoff(retry_count, base_delay=2\.0, max_delay=60\.0)|jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)|g' "$CONV_LOOP"
        find /opt/hermes/agent/__pycache__/ -name "conversation_loop*" -delete 2>/dev/null || true
        echo "  -> backoff patched: base 2.0->5.0s, max 60->120s (429 path) + .pyc cleared"
    else
        echo "  -> backoff already tuned to 5.0/120.0 (no-op)"
    fi
fi

# 1b. CRITICAL: Remove DUPLICATE provider: "auto" that upstream expansion adds
echo "  -> Checking for duplicate provider: auto..."
grep -n '^ *provider: "auto"' "$DATA/config.yaml" | while read line; do
    linenum=$(echo "$line" | cut -d: -f1)
    sed -i "${linenum}s|^ *provider: \"auto\"|  # provider: \"auto\"  # REMOVED: duplicate overrides zai|" "$DATA/config.yaml"
done

# 2. Remove OpenRouter base_url contamination
echo "  -> Checking for OpenRouter base_url contamination..."
if grep -q '^ *base_url: "https://openrouter.ai/api/v1"' "$DATA/config.yaml"; then
    echo "  -> FOUND OpenRouter base_url — commenting out"
    sed -i 's|^ *base_url: "https://openrouter.ai/api/v1"|#  base_url: "https://openrouter.ai/api/v1"  # REMOVED: use zai native|' "$DATA/config.yaml"
fi

# 3. Replace/add RooSync deployment section
# Remove ALL RooSync sections (from previous runs) and original upstream sections
# that we override, to prevent YAML duplicate key errors.
echo "  -> Removing old RooSync and upstream sections..."
LINE=$(grep -n '# --- RooSync deployment config' "$DATA/config.yaml" | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    head -n $((LINE - 1)) "$DATA/config.yaml" > /tmp/config_strip.yaml
    mv /tmp/config_strip.yaml "$DATA/config.yaml"
fi
# Remove original auxiliary: section (upstream) — we replace it
LINE=$(grep -n '^auxiliary:' "$DATA/config.yaml" | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    head -n $((LINE - 1)) "$DATA/config.yaml" > /tmp/config_aux.yaml
    mv /tmp/config_aux.yaml "$DATA/config.yaml"
fi
# Remove original stt: section (upstream) — we replace it
LINE=$(grep -n '^stt:' "$DATA/config.yaml" | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    head -n $((LINE - 1)) "$DATA/config.yaml" > /tmp/config_stt.yaml
    mv /tmp/config_stt.yaml "$DATA/config.yaml"
fi
# Remove any approvals: section
LINE=$(grep -n '^approvals:' "$DATA/config.yaml" | tail -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    head -n $((LINE - 1)) "$DATA/config.yaml" > /tmp/config_appr.yaml
    mv /tmp/config_appr.yaml "$DATA/config.yaml"
fi

# Require secrets — NO hardcoded fallbacks
STT_API_KEY="${WHISPER_BEARER_TOKEN:-}"
MCP_AUTH="${MCP_AUTH_TOKEN:-}"
if [ -z "$STT_API_KEY" ]; then echo "ERROR: WHISPER_BEARER_TOKEN not set (add to .env.secrets)"; exit 1; fi
if [ -z "$MCP_AUTH" ]; then echo "ERROR: MCP_AUTH_TOKEN not set (add to .env.secrets)"; exit 1; fi

cat >> "$DATA/config.yaml" << EOF

# --- RooSync deployment config (2026-05-16) ---
# Memory char-limit tuning (anti-livelock, incident 2026-08-11). Upstream
# defaults 2200/1375 are too tight for a coordinator bot that accrues one
# durable entry per incident/decision. Near-full MEMORY.md + a normal batch
# overflowed -> write rejected -> bot reinterpreted as "consolidate first"
# -> livelock (~20 min, ~30 leaked Telegram messages). Raising 2x gives
# headroom so a normal batch no longer overflows -> no rejection -> no loop.
memory:
  memory_char_limit: 4400
  user_char_limit: 2750

# All auxiliary tasks use z.ai provider (openrouter/nous cause 401 in gateway)
auxiliary:
  compression:
    provider: "zai"
    model: "glm-4.5-air"
  image:
    provider: "zai"
  browser:
    provider: "zai"
  web:
    provider: "zai"

# STT — Cluster Whisper endpoint (self-hosted on po-2023, OpenAI-compatible)
stt:
  enabled: true
  provider: "openai"
  openai:
    model: "whisper-1"
    base_url: "https://whisper-api.myia.io/v1"
    api_key: "${STT_API_KEY}"
EOF

# 3b. MCP servers — detect if ai-01 proxy is available, fallback to local
MCP_PROXY_UP=false
if command -v curl &>/dev/null; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
       "http://192.168.0.47:9090/sk-agent/mcp" \
       -H "Authorization: Bearer ${MCP_AUTH}" \
       -H "Content-Type: application/json" \
       -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"1.0"}}}' \
       --connect-timeout 5 --max-time 10 2>/dev/null || true)
    case "$HTTP_CODE" in
        200|401|403|405) MCP_PROXY_UP=true ;;
    esac
fi

if [ "$MCP_PROXY_UP" = true ]; then
    echo "  -> MCP proxy (192.168.0.47:9090) reachable — using remote"
    cat >> "$DATA/config.yaml" << EOF

# MCP servers — LAN direct to ai-01 proxy
mcp_servers:
  roo-state-manager:
    command: npx
    args:
      - -y
      - mcp-remote
      - http://192.168.0.47:9090/roo-state-manager/mcp
      - --allow-http
      - --header
      - "Authorization:Bearer ${MCP_AUTH}"
  sk-agent:
    command: npx
    args:
      - -y
      - mcp-remote
      - http://192.168.0.47:9090/sk-agent/mcp
      - --allow-http
      - --header
      - "Authorization:Bearer ${MCP_AUTH}"
  searxng:
    command: npx
    args:
      - -y
      - mcp-remote
      - http://192.168.0.47:9090/searxng/mcp
      - --allow-http
      - --header
      - "Authorization:Bearer ${MCP_AUTH}"
EOF
else
    echo "  -> MCP proxy (192.168.0.47:9090) unreachable — using local fallback"
    # CRITICAL: Copy roo-state-manager to /tmp BEFORE patching.
    # The volume mount is rw — sed -i on /opt/roo-state-manager would corrupt
    # the host .env (see regression #560). We patch ONLY the copy.
    # node_modules is a symlink to the host install, so we prefer cp -L to
    # dereference it. Some host installs carry a dangling self-symlink
    # (node_modules/roo-state-manager) that makes cp -L abort; tolerate it by
    # falling back to plain cp -r (keep links). NEVER let this abort the restore
    # — .env/config.yaml generation must run regardless of MCP copy success.
    echo "  -> Copying roo-state-manager to /tmp (isolated from host)"
    rm -rf /tmp/roo-state-manager 2>/dev/null
    cp -rL /opt/roo-state-manager /tmp/roo-state-manager 2>/dev/null \
        || cp -r /opt/roo-state-manager /tmp/roo-state-manager 2>/dev/null \
        || echo "  -> WARN: roo-state-manager copy incomplete (MCP may be degraded), continuing"

    # Patch the COPY (.env) for container mode
    # GDrive virtual drive can't be Docker-mounted — clear shared path
    # Qdrant is on ai-01 (down) — disable to prevent FATAL crash
    if [ -f /tmp/roo-state-manager/.env ]; then
        echo "  -> Patching /tmp/roo-state-manager/.env for container mode"
        sed -i 's|^ROOSYNC_SHARED_PATH=.*|ROOSYNC_SHARED_PATH=|' /tmp/roo-state-manager/.env
        sed -i 's|^QDRANT_URL=.*|QDRANT_URL=http://localhost:1|' /tmp/roo-state-manager/.env
        # Keep ROOSYNC_AUTO_SYNC=true — dashboard/messages work in-memory without GDrive
    fi
    # Patch the COPY (index.js) — unhandled rejection handler to NOT crash on Qdrant/fetch errors
    # The server's process.on('unhandledRejection') calls process.exit(1) for any
    # error that isn't IO or shared-path related. Qdrant fetch failures trigger this.
    if [ -f /tmp/roo-state-manager/build/index.js ]; then
        echo "  -> Patching /tmp copy: index.js unhandledRejection handler (container-safe)"
        sed -i "s/logger.error('Unhandled rejection (FATAL) at:', { promise: String(promise), \.\.\.reasonInfo });/logger.error('Unhandled rejection (degraded, container mode):', { promise: String(promise), ...reasonInfo });/" /tmp/roo-state-manager/build/index.js
        sed -i "/Unhandled rejection (degraded, container mode)/{n;s/process.exit(1);/return; \/\/ patched: don\\'t crash/}" /tmp/roo-state-manager/build/index.js
    fi
    cat >> "$DATA/config.yaml" << EOF

# MCP servers — LOCAL fallback (ai-01 proxy down)
# roo-state-manager: stdio direct (COPY in /tmp, patched for container — NEVER touches host)
# sk-agent + searxng: via local mcp-proxy container on port 9092
mcp_servers:
  roo-state-manager:
    command: node
    args:
      - /tmp/roo-state-manager/mcp-wrapper.cjs
    env:
      NODE_PATH: /tmp/roo-state-manager/node_modules
      DEFAULT_WORKSPACE: /opt/data
  sk-agent:
    command: npx
    args:
      - -y
      - mcp-remote
      - http://host.docker.internal:9092/sk-agent/mcp
      - --allow-http
      - --header
      - "Authorization:Bearer ${MCP_AUTH}"
  searxng:
    command: npx
    args:
      - -y
      - mcp-remote
      - http://host.docker.internal:9092/searxng/mcp
      - --allow-http
      - --header
      - "Authorization:Bearer ${MCP_AUTH}"
EOF
fi

# Auto-approve for gateway cron jobs (no user to approve in gateway mode)
cat >> "$DATA/config.yaml" << EOF

approvals:
  mode: off
  cron_mode: approve
EOF

# 4. Restore .env — non-secret config + secrets from env/file
echo "  -> Restoring .env with all tokens"
cat > "$DATA/.env" << EOF
TELEGRAM_ALLOWED_USERS=6541428999
TELEGRAM_GROUP_ALLOWED_USERS=6541428999
TELEGRAM_HOME_CHANNEL=-1003904676273
GATEWAY_ALLOW_ALL_USERS=false

# HOME — prevent subprocess/terminal tools from resolving to /root
# (s6-overlay injects HOME=/root; load_dotenv restores this before each cron run)
HOME=/opt/data

# XDG — prevent gateway-locks from landing in /root/.local/state
XDG_STATE_HOME=/opt/data/.local/state

# z.ai / GLM provider (still used by auxiliary tasks: compression, image, browser, web)
GLM_API_KEY=${GLM_API_KEY:-}
GLM_BASE_URL=${GLM_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}

# Anthropic / claudish — VALIDATED WORKING 2026-08-07.
# Gateway routes via claudish proxy (po-2023:3000), claude-sonnet-4-6 -> glm-5.2.
# Auth: claudish accepts Hermes's native x-api-key header (a third-party anthropic
# endpoint makes build_anthropic_client send api_key as x-api-key). x-proxy-key is
# NOT required on /v1/messages — proven by a direct probe (HTTP 200 + glm-5.2 body)
# and an end-to-end `hermes -z` returning a correct, instruction-following reply.
# NOTE: ANTHROPIC_CUSTOM_HEADERS is NOT read by Hermes (grep across .py = 0 refs),
# so x-proxy-key cannot be sent via env regardless; if claudish later enforces it,
# a custom_providers block (transport: anthropic_messages + extra_headers) is the
# only mechanism. CLAUDISH_PROXY_KEY stays provisioned in .env.secrets for that day;
# it is unused now. NEVER log the key.
ANTHROPIC_BASE_URL=http://192.168.0.46:3000
ANTHROPIC_TOKEN=placeholder

# Telegram bot
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}

# GitHub tokens
GH_TOKEN_CLUSTERMANAGER=${GH_TOKEN_CLUSTERMANAGER:-}
GH_TOKEN_JSBOIGEEPITA=${GH_TOKEN_JSBOIGEEPITA:-}
GH_TOKEN_JSBOIGE=${GH_TOKEN_JSBOIGE:-}
GH_TOKEN=${GH_TOKEN_CLUSTERMANAGER:-}
EOF

# 5. Fix jobs.json format
echo "  -> Checking jobs.json format"
if [ -f "$DATA/cron/jobs.json" ]; then
    python3 -c "
import json
with open('$DATA/cron/jobs.json', 'r') as f:
    data = json.load(f)
if isinstance(data, list):
    data = {'jobs': data}
for job in data.get('jobs', []):
    sched = job.get('schedule')
    if isinstance(sched, str):
        job['schedule'] = {'kind': 'cron', 'expr': sched, 'display': sched}
    if job.get('repeat') == 'forever':
        job['repeat'] = None
    # Remove toolset restrictions — cluster crons need full access
    if 'enabled_toolsets' in job:
        del job['enabled_toolsets']
    # Re-enable paused jobs
    if job.get('enabled') is False:
        job['enabled'] = True
        job.pop('paused_at', None)
        job.pop('paused_reason', None)
        job['state'] = 'scheduled'
    # Enforce 30min interval for ping-intercom (alternance with NanoClaw :15,:45)
    if 'ping-intercom' in job.get('name', '') and job.get('schedule', {}).get('kind') == 'interval':
        job['schedule']['minutes'] = 30
        job['schedule']['display'] = 'every 30m'
        job['schedule_display'] = 'every 30m'
with open('$DATA/cron/jobs.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('  -> jobs.json OK')
" 2>/dev/null || echo "  -> Warning: could not check jobs.json format"
fi

# 5b. Ensure #2505 read-body directive in pr-review prompt (issue #2505, 2026-06-23)
# Idempotent: inject the "read all existing reviews before posting" discipline into
# the hermes-pr-review cron prompt if the marker is absent. Guarantees the directive
# survives a full rebuild (jobs.json prompts are NOT regenerated elsewhere).
echo "  -> Checking #2505 read-body directive in pr-review prompt"
if [ -f "$DATA/cron/jobs.json" ]; then
python3 -c "
import json
path = '$DATA/cron/jobs.json'
with open(path, 'r') as f:
    data = json.load(f)
MARKER = 'READ-BODY issue #2505'
DIRECTIVE = (
    '\n\n## \u26a0\ufe0f REGLE ANTI-SPAM \u2014 PRIORITE ABSOLUE - READ-BODY issue #2505\n\n'
    '**Discipline read-body (issue #2505, 2026-06-23) :** avant de poster, lis TOUTES '
    'les reviews + commentaires existants (humains ET bots, **dont NanoClaw**) sur la PR '
    '(`gh pr view NNN --json reviews,comments`). Ne poste que du **contenu reellement neuf**. '
    'Ne **jamais dupliquer** ni **contredire aveuglement** une review deja postee - en cas de '
    'desaccord, lis d abord sa justification et sois explicite sur ce qui est neuf/incorrect. '
    'Tout deja couvert -> silence (ou bref \`[ACK]\`).\n'
)
changed = False
for job in data.get('jobs', []):
    if job.get('name') == 'hermes-pr-review':
        p = job.get('prompt', '')
        if MARKER not in p:
            # Insert right after the first line (role declaration)
            if '\n' in p:
                head, rest = p.split('\n', 1)
                job['prompt'] = head + DIRECTIVE + rest
            else:
                job['prompt'] = p + DIRECTIVE
            changed = True
if changed:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('  -> #2505 read-body directive injected into pr-review prompt')
else:
    print('  -> #2505 read-body directive already present (no-op)')
" 2>/dev/null || echo "  -> Warning: could not check #2505 directive"
fi

# 5c. Idempotent: inject the "sequential review processing" directive into the
# hermes-pr-review prompt to prevent any future fan-out via Task tool / sub-agents.
# 2026-06-25: z.ai hardened concurrent access -> reducing parallel API emission
# from the bot side (claudish handles the proxy side via 529 patient-backoff).
# Hermes is implicitly sequential today but this anchor prevents regression.
echo "  -> Checking sequential-review directive in pr-review prompt"
if [ -f "$DATA/cron/jobs.json" ]; then
python3 -c "
import json
path = '$DATA/cron/jobs.json'
with open(path, 'r') as f:
    data = json.load(f)
MARKER = 'TRAITEMENT SEQUENTIEL STRICT'
DIRECTIVE = (
    '

## TRAITEMENT SEQUENTIEL STRICT (#debit-zai 2026-06-25)

'
    '**Pour limiter la pression concurrente sur z.ai (durcissement quotas 2026-06-25)** : '
    'traiter les PRs **une apres l autre dans ce tour principal**. INTERDIT :
'
    '- Lancer des sous-agents (Task tool) pour paralleliser les reviews
'
    '- Emettre plusieurs gh API calls en parallele (parallel tool calls dans un meme tour)

'
    'Finir 1 PR completement (fetch + dedup + diff + review post) avant de passer a la '
    'suivante. Plafond \`Max 5 PRs/cycle\` inchange - juste **etale sequentiellement** '
    'au lieu de fan-out. Cron lance toutes les 1h, donc largement le temps.
'
)
changed = False
for job in data.get('jobs', []):
    if job.get('name') == 'hermes-pr-review':
        p = job.get('prompt', '')
        # Match both accented and de-accented marker variants
        if MARKER not in p and 'TRAITEMENT SEQUENTIEL' not in p.upper() and 'SEQUENTIEL STRICT' not in p.upper():
            anchor_a = '## REGLE TOKENS'
            anchor_b = '## RULE TOKENS'
            inserted = False
            for anc in ('## REGLE TOKENS', '## RULE TOKENS', '## TOKENS PAR REPO'):
                if anc in p:
                    p = p.replace(anc, DIRECTIVE.strip() + '

' + anc, 1)
                    inserted = True
                    break
            if not inserted:
                p = p + DIRECTIVE
            job['prompt'] = p
            changed = True
if changed:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('  -> sequential-review directive injected into pr-review prompt')
else:
    print('  -> sequential-review directive already present (no-op)')
" 2>/dev/null || echo "  -> Warning: could not check sequential-review directive"
fi

# 5d. Idempotent: enforce Telegram presence for cluster-tour + pr-review crons.
# 2026-06-26: Hermes appeared silent in the group chat because every cron
# returned [SILENT] when the board was covered / cluster stable. The cron
# scheduler injects a system directive letting the agent emit "[SILENT]" to
# suppress delivery (cron/scheduler.py:1139-1142, detected at :2064). We
# counter-mand it at the JOB-PROMPT level so Hermes always posts at least a
# one-line status (heartbeat), developing (<=3 lines) only when something is
# notable. inbox-poll (30min cadence) is intentionally left SILENT-on-RAS to
# avoid ~48 msgs/day noise; its value is silent background processing.
echo "  -> Checking chat-presence directive in cluster-tour + pr-review prompts"
if [ -f "$DATA/cron/jobs.json" ]; then
python3 -c "
import json
path = '$DATA/cron/jobs.json'
with open(path, 'r') as f:
    data = json.load(f)
MARKER = 'PRESENCE-CHAT 2026-06-26'
HEADER = (
    '\n\n## DIRECTIVE PRESENCE-CHAT 2026-06-26\n\n'
    '**Presence chat obligatoire.** Meme si rien de notable, poste au moins 1 phrase '
    'disant ce que tu as verifie (ex: tour 18:00Z - cluster 6/6 stable, board couvert, RAS). '
    'Developpe (jusqu a 3 lignes) uniquement si quelque chose est notable. '
    '**Ne reponds JAMAIS [SILENT] dans ce job** - le marker systeme du scheduler est '
    'contre-mande ici ; ta reponse est TOUJOURS livree a Telegram.\n'
)
CLUSTER_NEW = 'Poste TOUJOURS un message - JAMAIS [SILENT]. 1 phrase en RAS, jusqu a 3 lignes si delta notable.'
PRREV_NEW = 'Poste TOUJOURS ce bilan meme si X=0 (ex: pr-review 19:23Z - board couvert, 0 nouveaute, Z SHA-skips). Ne reponds jamais [SILENT].'
changed = False
for job in data.get('jobs', []):
    name = job.get('name', '')
    p = job.get('prompt', '')
    if name in ('hermes-cluster-tour', 'hermes-pr-review'):
        if MARKER not in p:
            if name == 'hermes-cluster-tour':
                p = p.replace('Silence si RAS', CLUSTER_NEW)
            else:
                p = p.replace('Ne liste que les reviews EFFECTIVEMENT postees.', 'Ne liste que les reviews EFFECTIVEMENT postees. ' + PRREV_NEW)
            if '\n## FORMAT OUTPUT' in p:
                p = p.replace('\n## FORMAT OUTPUT', HEADER + '\n## FORMAT OUTPUT', 1)
            else:
                p = p + HEADER
            job['prompt'] = p
            changed = True
if changed:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('  -> chat-presence directive injected into cluster-tour + pr-review prompts')
else:
    print('  -> chat-presence directive already present (no-op)')
" 2>/dev/null || echo "  -> Warning: could not check chat-presence directive"
fi


# 5e. Idempotent: anti-false-positive health-check guardrail for cluster-tour.
# 2026-07-03: Hermes escalated "vLLM:5002 DOWN ~13h" x5 (nearly triggered an
# unnecessary ai-01 vLLM restart). Root cause: from inside its container the bot
# curled 127.0.0.1:5002 (empty container loopback) instead of the real ai-01
# endpoint 192.168.0.47:5002 (which was UP, HTTP 401). Firsthand host diagnosis.
# We pin roosync_inventory type=health as the authoritative source and forbid
# ad-hoc loopback curls at the job-prompt level so the false positive cannot recur.
echo "  -> Checking health-source directive in cluster-tour prompt"
if [ -f "$DATA/cron/jobs.json" ]; then
python3 -c "
import json
path = '$DATA/cron/jobs.json'
with open(path, 'r') as f:
    data = json.load(f)
MARKER = 'HEALTH-SOURCE 2026-07-03'
BLOCK = (
    '\n\n## DIRECTIVE HEALTH-SOURCE 2026-07-03\n\n'
    '**Source autoritaire sante fleet.** roosync_inventory type=health est LA source '
    'de verite pour etat des services. Ne lance JAMAIS de curl ad-hoc pour declarer un '
    'service DOWN. Depuis ce container, 127.0.0.1 ne voit AUCUN service LAN : vLLM tourne '
    'sur ai-01 (192.168.0.47:5002), embeddings sur host.docker.internal:8004. Un RST sur '
    'loopback ne prouve JAMAIS une panne. HTTP 401 = service UP (auth-gated), pas DOWN. '
    'Ne signale un service DOWN QUE si roosync_inventory type=health le rapporte '
    'unreachable. Incident 2026-07-02 : faux positif vLLM:5002 escalade x5 via curl '
    '127.0.0.1 (loopback container vide) alors que le vrai endpoint ai-01 etait UP.\n'
)
changed = False
for job in data.get('jobs', []):
    if job.get('name') == 'hermes-cluster-tour':
        p = job.get('prompt', '')
        if MARKER not in p:
            if '\n## FORMAT OUTPUT' in p:
                p = p.replace('\n## FORMAT OUTPUT', BLOCK + '\n## FORMAT OUTPUT', 1)
            else:
                p = p + BLOCK
            job['prompt'] = p
            changed = True
if changed:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('  -> health-source directive injected into cluster-tour prompt')
else:
    print('  -> health-source directive already present (no-op)')
" 2>/dev/null || echo "  -> Warning: could not check health-source directive"
fi


# 6. Install croniter
echo "  -> Checking croniter"
/opt/hermes/.venv/bin/python3 -c 'import croniter' 2>/dev/null && echo "  -> croniter already installed" || {
    echo "  -> Installing croniter..."
    uv pip install --python /opt/hermes/.venv/bin/python3 --force-reinstall croniter 2>/dev/null && echo "  -> croniter installed" || echo "  -> Warning: croniter install failed"
}

# 7. Install gh CLI if missing
echo "  -> Checking gh CLI"
if ! command -v gh &>/dev/null; then
    echo "  -> Installing gh CLI..."
    ARCH=$(dpkg --print-architecture 2>/dev/null || echo "amd64")
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    echo "deb [arch=$ARCH signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
    apt-get update -qq 2>/dev/null && apt-get install -y -qq gh 2>/dev/null
    gh --version 2>/dev/null && echo "  -> gh CLI installed" || echo "  -> Warning: gh CLI install failed"
else
    echo "  -> gh CLI already installed: $(gh --version 2>/dev/null | head -1)"
fi

# 7b. Install jq if missing
echo "  -> Checking jq"
if ! command -v jq &>/dev/null; then
    echo "  -> Installing jq..."
    apt-get update -qq 2>/dev/null && apt-get install -y -qq jq 2>/dev/null
    jq --version 2>/dev/null && echo "  -> jq installed" || echo "  -> Warning: jq install failed"
else
    echo "  -> jq already installed: $(jq --version 2>/dev/null)"
fi

# 7c. gh auth login with GH_TOKEN (persists to /opt/data so it survives restarts)
echo "  -> Configuring gh CLI auth"
GH_TOKEN_VAL="${GH_TOKEN_CLUSTERMANAGER:-}"
if [ -n "$GH_TOKEN_VAL" ]; then
    # Persist gh config to /opt/data so it survives container restarts
    mkdir -p /opt/data/.config/gh
    ln -sf /opt/data/.config/gh /root/.config/gh 2>/dev/null || true
    echo "$GH_TOKEN_VAL" | gh auth login --with-token 2>/dev/null || true
    if gh auth status &>/dev/null; then
        echo "  -> gh auth configured (token from GH_TOKEN_CLUSTERMANAGER)"
    else
        echo "  -> Warning: gh auth login failed"
    fi
else
    # Try reusing existing auth if already in /opt/data/.config/gh
    if [ -d "/opt/data/.config/gh" ]; then
        ln -sf /opt/data/.config/gh /root/.config/gh 2>/dev/null || true
        if gh auth status &>/dev/null; then
            echo "  -> gh auth restored from /opt/data/.config/gh"
        else
            echo "  -> Warning: GH_TOKEN_CLUSTERMANAGER not set, gh auth not configured"
        fi
    else
        echo "  -> Warning: GH_TOKEN_CLUSTERMANAGER not set, gh auth not configured"
    fi
fi

# 8. Inject secrets into s6 container environment so they're available
# to the CMD (main-wrapper.sh) via with-contenv, and thus to cron terminal tool.
S6_ENV="/run/s6/container_environment"
if [ -d "$S6_ENV" ]; then
    for var in GLM_API_KEY TELEGRAM_BOT_TOKEN GH_TOKEN_CLUSTERMANAGER GH_TOKEN_JSBOIGEEPITA GH_TOKEN_JSBOIGE GH_TOKEN MCP_AUTH_TOKEN WHISPER_BEARER_TOKEN; do
        val="${!var:-}"
        if [ -n "$val" ]; then
            printf '%s' "$val" > "$S6_ENV/$var"
        fi
    done
    # XDG_STATE_HOME: prevent gateway-locks from landing in /root/.local/state
    # (regression from s6-overlay HOME=/root in container_environment)
    printf '/opt/data/.local/state' > "$S6_ENV/XDG_STATE_HOME"
    # Fix HOME to point at the persistent volume instead of /root
    printf '/opt/data' > "$S6_ENV/HOME"
    echo "  -> Secrets + XDG_STATE_HOME + HOME injected into s6 container environment"
fi

# 8a. Fix ownership
chown hermes:hermes "$DATA/config.yaml" "$DATA/.env" "$DATA/cron/jobs.json" "$DATA/SOUL.md" "$DATA/.config" "$DATA/kanban.db" "$DATA/kanban.db-wal" "$DATA/kanban.db-shm" 2>/dev/null || true

# 8b. Symlink config files into hermes home (~/.hermes = $DATA/.hermes/)
# Hermes reads config from ~/.hermes/ but restore writes to $DATA/ (volume root).
HERMES_HOME="$DATA/.hermes"
mkdir -p "$HERMES_HOME/cron" 2>/dev/null || true
ln -sf "$DATA/config.yaml" "$HERMES_HOME/config.yaml" 2>/dev/null || true
ln -sf "$DATA/.env" "$HERMES_HOME/.env" 2>/dev/null || true
ln -sf "$DATA/.env.secrets" "$HERMES_HOME/.env.secrets" 2>/dev/null || true
ln -sf "$DATA/cron/jobs.json" "$HERMES_HOME/cron/jobs.json" 2>/dev/null || true

# 8b. Patch kanban SCHEMA_SQL — remove premature CREATE INDEX for session_id
# Upstream bug: SCHEMA_SQL has CREATE INDEX on session_id before migration adds the column
KANBAN_FILE="/opt/hermes/hermes_cli/kanban_db.py"
if [ -f "$KANBAN_FILE" ]; then
    echo "  -> Patching kanban SCHEMA_SQL (session_id index)"
    # Comment out the premature index creation
    if grep -q 'CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)' "$KANBAN_FILE"; then
        sed -i 's|CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)|-- CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)  -- PATCHED: migration creates it after ALTER TABLE|' "$KANBAN_FILE"
        # Clear __pycache__ so patch takes effect
        find /opt/hermes -name '*.pyc' -delete 2>/dev/null || true
        find /opt/hermes -name '__pycache__' -type d -empty -delete 2>/dev/null || true
        echo "  -> Kanban SCHEMA_SQL patched"
    else
        echo "  -> Kanban SCHEMA_SQL already patched (or upstream fixed)"
    fi
else
    echo "  -> Warning: kanban_db.py not found at $KANBAN_FILE"
fi

# 8b. Fix /root permissions for hermes user
# The gateway drops to hermes (UID 10000) via s6-setuidgid, but Python's
# os.path.expanduser('~') and various library calls resolve to /root when
# HOME is inherited from the Docker env. /root is 700 by default (Debian),
# causing Permission denied on terminal, file, and browser tools.
chmod 755 /root

# 8c. claudish thinking-block SDK patch (2026-08-24 incident)
# claudish (192.168.0.46) started emitting Anthropic-style thinking
# content blocks WITHOUT the required `signature` field. The Anthropic
# SDK's streaming accumulate_event() then leaves event.content_block as
# a plain dict and crashes on .model_dump() -> every LLM cron failed
# 7/7 retries (inbox-poll, pr-review). Fix: sitecustomize module that
# retypes dict thinking blocks into proper ThinkingBlock instances
# before accumulation. Idempotent: skips if .pth already present.
SP_DIR="/opt/hermes/.venv/lib/python3.13/site-packages"
if [ ! -f "$SP_DIR/zz_claudish_thinking_patch.pth" ]; then
    cat > "$SP_DIR/zz_claudish_thinking_patch.py" <<'ZCEOF'
"""claudish/GLM emits thinking content blocks without the required signature
field; the Anthropic SDK then leaves event.content_block as a plain dict and
accumulate_event() crashes on .model_dump(). Retype them properly.

Loaded via zz_claudish_thinking_patch.pth in site-packages.
Operator patch 2026-08-24 (po-2026): incident inbox-poll/pr-review 7/7
AttributeError after claudish .46 started emitting signature-less thinking.
"""
import anthropic.lib.streaming._messages as _am
from anthropic.types import ThinkingBlock

if getattr(_am.accumulate_event, "__name__", "") != "_retyped_accumulate":
    _orig_accumulate = _am.accumulate_event

    def _retyped_accumulate(*args, **kwargs):
        ev = kwargs.get("event") if "event" in kwargs else (args[0] if args else None)
        cb = getattr(ev, "content_block", None)
        if isinstance(cb, dict) and cb.get("type") == "thinking":
            try:
                ev.content_block = ThinkingBlock(
                    type="thinking",
                    thinking=cb.get("thinking") or "",
                    signature=cb.get("signature") or "",
                )
            except Exception:
                pass
        return _orig_accumulate(*args, **kwargs)

    _am.accumulate_event = _retyped_accumulate
ZCEOF
    echo "import zz_claudish_thinking_patch" > "$SP_DIR/zz_claudish_thinking_patch.pth"
    echo "  -> claudish thinking patch installed (sitecustomize)"
else
    echo "  -> claudish thinking patch already present"
fi

# 9. Verify everything
echo ""
echo "=== VERIFICATION ==="
PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    # Accept any detail starting with "OK" (e.g. "OK (value masked)") as a pass.
    case "$result" in
        OK*)
            echo "  [PASS] $label${result#OK}"
            PASS=$((PASS + 1))
            ;;
        *)
            echo "  [FAIL] $label — $result"
            FAIL=$((FAIL + 1))
            ;;
    esac
}

# Model — claudish route: must be claude-sonnet-4-6 (anthropic wire via claudish, restored 2026-08-07)
MODEL=$(grep '^  default:' "$DATA/config.yaml" | head -1)
[[ "$MODEL" == *claude-sonnet-4-6* ]] && check "Model" "OK" || check "Model" "got: $MODEL"

# YAML valid (no duplicate keys)
DUP_AUX=$(grep -c '^auxiliary:' "$DATA/config.yaml")
DUP_STT=$(grep -c '^stt:' "$DATA/config.yaml")
DUP_MCP=$(grep -c '^mcp_servers:' "$DATA/config.yaml")
DUP_APPR=$(grep -c '^approvals:' "$DATA/config.yaml")
if [ "$DUP_AUX" -le 1 ] && [ "$DUP_STT" -le 1 ] && [ "$DUP_MCP" -le 1 ] && [ "$DUP_APPR" -le 1 ]; then
    check "YAML no duplicate keys" "OK"
else
    check "YAML duplicates" "aux=$DUP_AUX stt=$DUP_STT mcp=$DUP_MCP appr=$DUP_APPR"
fi

# Provider — claudish route (restored 2026-08-07): main provider is anthropic.
# Auxiliary providers stay on zai (compression, image, browser, web).
PROV=$(grep '^  provider:' "$DATA/config.yaml" | head -1)
[[ "$PROV" == *anthropic* ]] && check "Provider (main=anthropic)" "OK" || check "Provider" "got: $PROV"

# ANTHROPIC_BASE_URL must point at claudish proxy (po-2023:3000).
ANTH_URL=$(grep -c '^ANTHROPIC_BASE_URL=http://192.168.0.46:3000' "$DATA/.env" 2>/dev/null || true)
[ "$ANTH_URL" = "1" ] && check "ANTHROPIC_BASE_URL (claudish)" "OK" || check "ANTHROPIC_BASE_URL" "missing/wrong (count=$ANTH_URL)"

# CLAUDISH_PROXY_KEY provisioned in .env.secrets for future use (if claudish ever
# enforces x-proxy-key, a custom_providers block will send it). Verify presence
# WITHOUT printing the value. Unused today — claudish accepts native x-api-key auth.
KEY_LEN=$(grep -o '^CLAUDISH_PROXY_KEY=[0-9a-f]\{64\}$' "$DATA/.env.secrets" 2>/dev/null | head -1 | wc -c)
[ "$KEY_LEN" -gt 0 ] && check "CLAUDISH_PROXY_KEY provisioned" "OK (unused, value masked)" || check "CLAUDISH_PROXY_KEY" "not provisioned (OK if claudish stays x-api-key)"

# No duplicate provider: auto
DUP=$(grep -c '^ *provider: "auto"' "$DATA/config.yaml" || true)
[ "$DUP" = "0" ] && check "No duplicate provider:auto" "OK" || check "No duplicate provider:auto" "found $DUP"

# No OpenRouter contamination
OR_URL=$(grep -c 'openrouter.ai' "$DATA/config.yaml" || true)
[ "$OR_URL" = "0" ] && check "No OpenRouter base_url" "OK" || check "No OpenRouter base_url" "found $OR_URL"

# Auxiliary
AUX=$(grep -c 'provider: "zai"' "$DATA/config.yaml")
[ "$AUX" -ge 4 ] && check "Auxiliary providers (zai)" "OK" || check "Auxiliary providers" "only $AUX zai entries"

# STT
STT=$(grep -c 'whisper-api.myia.io' "$DATA/config.yaml")
[ "$STT" -ge 1 ] && check "STT endpoint" "OK" || check "STT endpoint" "not found"

# MCP servers
MCPS=$(grep -c 'mcp_servers:' "$DATA/config.yaml")
[ "$MCPS" -ge 1 ] && check "MCP servers section" "OK" || check "MCP servers section" "not found"
MCP_COUNT=$(grep -c '192.168.0.47:9090' "$DATA/config.yaml")
[ "$MCP_COUNT" -ge 3 ] && check "MCP bridges (3)" "OK" || check "MCP bridges" "only $MCP_COUNT"

# Approvals
APPR=$(grep -c 'cron_mode: approve' "$DATA/config.yaml")
[ "$APPR" -ge 1 ] && check "Approvals (cron_mode)" "OK" || check "Approvals" "not found"

# .env secrets
for tok in GLM_API_KEY TELEGRAM_BOT_TOKEN GH_TOKEN_CLUSTERMANAGER; do
    VAL=$(grep "^${tok}=" "$DATA/.env" | cut -d= -f2)
    [ -n "$VAL" ] && check ".env $tok" "OK" || check ".env $tok" "EMPTY"
done

# jobs.json
if [ -f "$DATA/cron/jobs.json" ]; then
    JOBS=$(python3 -c "import json; d=json.load(open('$DATA/cron/jobs.json')); print(len(d.get('jobs',[])))" 2>/dev/null || echo 0)
    [ "$JOBS" -ge 1 ] && check "Cron jobs ($JOBS)" "OK" || check "Cron jobs" "found $JOBS"
    TOOLSETS=$(python3 -c "import json; d=json.load(open('$DATA/cron/jobs.json')); print(sum(1 for j in d.get('jobs',[]) if 'enabled_toolsets' in j))" 2>/dev/null || echo "?")
    [ "$TOOLSETS" = "0" ] && check "No enabled_toolsets restrictions" "OK" || check "No enabled_toolsets" "$TOOLSETS jobs have restrictions"
else
    check "Cron jobs" "jobs.json not found"
fi

# gh CLI
command -v gh &>/dev/null && check "gh CLI" "OK" || check "gh CLI" "not installed"

# croniter
/opt/hermes/.venv/bin/python3 -c 'import croniter' 2>/dev/null && check "croniter" "OK" || check "croniter" "not installed"

# jq
command -v jq &>/dev/null && check "jq" "OK" || check "jq" "not installed"

# gh auth
gh auth status &>/dev/null && check "gh auth" "OK" || check "gh auth" "not configured"

# gh config persisted
[ -d "/opt/data/.config/gh" ] && check "gh config persisted" "OK" || check "gh config persisted" "not found in /opt/data"

# kanban patch
if [ -f "$KANBAN_FILE" ]; then
    KPATCH=$(grep -c 'PATCHED: migration creates it' "$KANBAN_FILE" || true)
    [ "$KPATCH" -ge 1 ] && check "Kanban session_id patch" "OK" || check "Kanban session_id patch" "not applied"
fi

# Symlinks hermes home (config must be visible from ~/.hermes/)
SYMLINK_OK=0
for f in config.yaml .env .env.secrets cron/jobs.json; do
    TARGET="$DATA/.hermes/$f"
    if [ -L "$TARGET" ]; then
        LINK=$(readlink "$TARGET")
        SYMLINK_OK=$((SYMLINK_OK + 1))
    elif [ -f "$TARGET" ]; then
        check "Symlink .hermes/$f" "FILE EXISTS (not symlink)"
    else
        check "Symlink .hermes/$f" "MISSING"
    fi
done
[ "$SYMLINK_OK" -ge 4 ] && check "Symlinks hermes home ($SYMLINK_OK/4)" "OK" || check "Symlinks hermes home" "only $SYMLINK_OK/4"

# Kanban DB writable
KDB="$DATA/.hermes/kanban.db"
if [ -f "$KDB" ] || [ -L "$KDB" ]; then
    REAL_KDB=$(readlink -f "$KDB" 2>/dev/null || echo "$KDB")
    KW=$(/opt/hermes/.venv/bin/python3 -c "
import sqlite3
conn=sqlite3.connect('$REAL_KDB')
conn.execute('CREATE TABLE IF NOT EXISTS _write_test (id INTEGER)')
conn.execute('DROP TABLE IF EXISTS _write_test')
conn.commit()
conn.close()
print('OK')" 2>/dev/null || echo "FAIL: not writable as hermes")
    check "Kanban DB writable" "$KW"
else
    check "Kanban DB" "not found (will be created on first use)"
fi

# Gateway Telegram state (check after process has had time to start)
sleep 3
GS_FILE=""
for gf in "$DATA/.hermes/gateway_state.json" "$DATA/gateway_state.json"; do
    if [ -f "$gf" ]; then GS_FILE="$gf"; break; fi
done
if [ -n "$GS_FILE" ]; then
    GS=$(python3 -c "
import json
d=json.load(open('$GS_FILE'))
tg=d.get('platforms',{}).get('telegram',{}).get('state','unknown')
print(tg)" 2>/dev/null || echo "parse_error")
    [[ "$GS" == *"connected"* ]] && check "Telegram state" "OK" || check "Telegram state" "$GS"
else
    check "Gateway state" "not yet written (normal during boot)"
fi

echo ""
if [ "$FAIL" = "0" ]; then
    echo "=== ALL $PASS CHECKS PASSED ==="
else
    echo "=== $PASS passed, $FAIL FAILED ==="
fi
echo "Done. Restart container to apply: docker restart hermes"
