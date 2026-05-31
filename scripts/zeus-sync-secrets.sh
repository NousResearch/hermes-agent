#!/usr/bin/env bash
# Sincroniza secretos de Infisical → ~/.hermes/runtime-secrets.env
# para servicios systemd de Zeus (gateway, dashboard, workspace).
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
BOOTSTRAP="${HERMES_HOME}/infisical-bootstrap.env"
OUTPUT="${HERMES_HOME}/runtime-secrets.env"
TMP="$(mktemp)"

cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

mkdir -p "$HERMES_HOME"
chmod 700 "$HERMES_HOME"

if [[ ! -f "$BOOTSTRAP" ]]; then
  echo "[zeus-secrets] Falta $BOOTSTRAP — copia infisical-bootstrap.env.example y agrega credenciales." >&2
  : >"$OUTPUT"
  chmod 600 "$OUTPUT"
  exit 0
fi

# shellcheck disable=SC1090
source "$BOOTSTRAP"

if [[ -z "${INFISICAL_CLIENT_ID:-}" || -z "${INFISICAL_CLIENT_SECRET:-}" ]]; then
  echo "[zeus-secrets] INFISICAL_CLIENT_ID/SECRET vacíos; se mantiene runtime-secrets.env mínimo." >&2
  cat >"$OUTPUT" <<EOF
# Generado por zeus-sync-secrets.sh — sin credenciales Infisical aún
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
EOF
  chmod 600 "$OUTPUT"
  exit 0
fi

export INFISICAL_API_URL="${INFISICAL_API_URL:-${INFISICAL_SITE_URL%/}/api}"
SITE="${INFISICAL_SITE_URL:-http://100.68.195.19:8080}"

INFISICAL_BIN="$(command -v infisical || true)"
if [[ -z "$INFISICAL_BIN" ]]; then
  echo "[zeus-secrets] infisical CLI no encontrado en PATH." >&2
  exit 1
fi

TOKEN="$("$INFISICAL_BIN" login \
  --method=universal-auth \
  --client-id="$INFISICAL_CLIENT_ID" \
  --client-secret="$INFISICAL_CLIENT_SECRET" \
  --domain="$SITE" \
  --silent --plain 2>/dev/null || true)"

if [[ -z "$TOKEN" ]]; then
  echo "[zeus-secrets] No se pudo obtener token universal-auth; revisa credenciales/proyecto." >&2
  exit 1
fi

export INFISICAL_TOKEN="$TOKEN"

if [[ -z "${INFISICAL_PROJECT_ID:-}" ]]; then
  echo "[zeus-secrets] Falta INFISICAL_PROJECT_ID (UUID del proyecto en Infisical UI, no el Identity ID)." >&2
  cat >"$OUTPUT" <<EOF
# Generado sin project id — completa INFISICAL_PROJECT_ID en infisical-bootstrap.env
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
EOF
  chmod 600 "$OUTPUT"
  exit 1
fi

if [[ "${INFISICAL_PROJECT_ID}" == "${INFISICAL_IDENTITY_ID:-}" ]]; then
  echo "[zeus-secrets] INFISICAL_PROJECT_ID coincide con Identity ID; usa el UUID del proyecto agent-zeus." >&2
  exit 1
fi

EXPORT_ARGS=(export --format=dotenv --domain="$SITE" --env="${INFISICAL_ENV:-prod}" --projectId="$INFISICAL_PROJECT_ID")

if ! "$INFISICAL_BIN" "${EXPORT_ARGS[@]}" >"$TMP" 2>"${HERMES_HOME}/logs/infisical-sync.log"; then
  echo "[zeus-secrets] export falló; ver ${HERMES_HOME}/logs/infisical-sync.log" >&2
  exit 1
fi

{
  echo "# Zeus runtime secrets — generado $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "API_SERVER_ENABLED=true"
  echo "API_SERVER_HOST=127.0.0.1"
  echo "API_SERVER_PORT=8642"
  grep -Ev '^(HERMES_PASSWORD|CLAUDE_PASSWORD|HERMES_API_TOKEN)=' "$TMP" || true
  api_key_line="$(grep -E '^API_SERVER_KEY=' "$TMP" | head -1 || true)"
  if [[ -n "$api_key_line" ]]; then
    echo "${api_key_line/API_SERVER_KEY/HERMES_API_TOKEN}"
  fi
} >"$OUTPUT"
python3 - "$OUTPUT" <<'PY'
from pathlib import Path
from urllib.parse import urlparse, unquote, quote
import sys
path = Path(sys.argv[1])
values = {}
order = []
for raw in path.read_text().splitlines():
    if not raw or raw.startswith('#') or '=' not in raw:
        continue
    k, v = raw.split('=', 1)
    values[k] = v.strip().strip('"').strip("'")
    order.append(k)

def set_if_missing(key, value):
    if not value:
        return
    if not values.get(key):
        values[key] = value
        order.append(key)

for pwd_key, url_key in {
    'AGENT_DB_RUNTIME_PASSWORD': 'AGENT_DATABASE_URL',
    'FACTORY_DB_RUNTIME_PASSWORD': 'FACTORY_DATABASE_URL',
    'CALENDAR_DB_RUNTIME_PASSWORD': 'CALENDAR_DATABASE_URL',
    'CRM_DB_RUNTIME_PASSWORD': 'CRM_DATABASE_URL',
}.items():
    url = values.get(url_key, '')
    if url:
        parsed = urlparse(url)
        if parsed.password:
            set_if_missing(pwd_key, unquote(parsed.password))

def docker_url(user_key, pwd_key, db_key, out_key):
    user = values.get(user_key)
    pwd = values.get(pwd_key)
    db = values.get(db_key)
    if user and pwd and db:
        values[out_key] = f"postgresql://{quote(user)}:{quote(pwd)}@agent-postgres:5432/{db}"
        if out_key not in order:
            order.append(out_key)

docker_url('AGENT_DB_RUNTIME_USER', 'AGENT_DB_RUNTIME_PASSWORD', 'AGENT_DB_NAME', 'AGENT_DATABASE_URL_DOCKER')
docker_url('FACTORY_DB_RUNTIME_USER', 'FACTORY_DB_RUNTIME_PASSWORD', 'AGENT_DB_NAME', 'FACTORY_DATABASE_URL_DOCKER')
docker_url('CALENDAR_DB_RUNTIME_USER', 'CALENDAR_DB_RUNTIME_PASSWORD', 'AGENT_CALENDAR_DB_NAME', 'CALENDAR_DATABASE_URL_DOCKER')
docker_url('CRM_DB_RUNTIME_USER', 'CRM_DB_RUNTIME_PASSWORD', 'AGENT_CRM_DB_NAME', 'CRM_DATABASE_URL_DOCKER')

# Agent-facing aliases for the generic calendar toolset. ACCOUNT_API_KEY is
# the Nettu account key; keep a Hermes-native alias so the tool contract stays
# scheduler-agnostic, and point local host tools to the published scheduler port.
set_if_missing('HERMES_CALENDAR_BASE_URL', values.get('CALENDAR_API_BASE_URL') or 'http://127.0.0.1:5055/api/v1')
set_if_missing('HERMES_CALENDAR_API_KEY', values.get('ACCOUNT_API_KEY'))
set_if_missing('NETTU_ACCOUNT_API_KEY', values.get('ACCOUNT_API_KEY'))

lines = [line for line in path.read_text().splitlines() if line.startswith('#')]
seen = set()
for k in order:
    if k in seen or k not in values:
        continue
    seen.add(k)
    lines.append(f"{k}={values[k]}")
path.write_text('\n'.join(lines) + '\n')
PY
chmod 600 "$OUTPUT"
echo "[zeus-secrets] OK → $OUTPUT ($(wc -l <"$OUTPUT") líneas)"
