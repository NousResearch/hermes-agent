#!/usr/bin/env bash
# Launch the local Collective Wisdom demo as one coordinated, foreground stack.
#
# The supervisor keeps Portal, Gateway, the Hermes messaging gateway, Dashboard,
# and Desktop on the default Hermes profile while isolating local demo
# credentials and logs. Press Ctrl-C to stop application processes; database
# and object-store containers are intentionally left running so demo state
# survives.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORTAL_APP="${WISDOM_PORTAL_APP:-${ROOT}/../Nous/hermes-portal-wisdom/apps/nous-account-service}"
GATEWAY_REPO="${WISDOM_GATEWAY_REPO:-${ROOT}/../Nous/gateway-gateway}"
DEMO_HOME="${WISDOM_DEMO_HOME:-${HOME}/.hermes/wisdom-local-demo}"
AGENT_HOME="${WISDOM_AGENT_HOME:-${HERMES_HOME:-${HOME}/.hermes}}"
STATE_DIR="${WISDOM_DEMO_STATE_DIR:-${DEMO_HOME}/demo-stack}"
PUBLISHER_HOME="${WISDOM_PUBLISHER_HOME:-${DEMO_HOME}/publisher}"
PUBLISHER_METADATA="${STATE_DIR}/publisher-fixture.json"
PORTAL_URL="${WISDOM_PORTAL_URL:-http://127.0.0.1:3111}"
GATEWAY_URL="${WISDOM_GATEWAY_URL:-http://127.0.0.1:8787}"
TEAM_ORG_ID="${WISDOM_TEAM_ORG_ID:-nas_organisation:wisdom-local}"
TEAM_ORG_SLUG="${WISDOM_TEAM_ORG_SLUG:-wisdom-local}"
PRIVY_DID="${WISDOM_PRIVY_DID:-did:privy:user-9399fb3c}"
MINIO_CONTAINER="${WISDOM_MINIO_CONTAINER:-codex-wisdom-minio}"
MINIO_BUCKET="${WISDOM_MINIO_BUCKET:-wisdom}"
MINIO_ENDPOINT="${WISDOM_MINIO_ENDPOINT:-http://127.0.0.1:20900}"
PG_URL="postgresql://postgres:password@127.0.0.1:5436"

declare -a CHILD_PIDS=()

usage() {
  cat <<'EOF'
Usage: scripts/wisdom-demo-stack.sh [up|login|status|publisher-setup|publisher-v1|publisher-v2]

  up      Start the complete local demo in this terminal (default).
  login   Refresh the foreground Portal browser's local demo session.
  status  Show the expected listener state without changing anything.
  publisher-setup  Create or refresh the isolated second-member fixture.
  publisher-v1     Publish a unique v1 skill as the second team member.
  publisher-v2     Publish v2 of the fixture skill after recipient install.

Useful overrides:
  WISDOM_PORTAL_APP, WISDOM_GATEWAY_REPO, HERMES_HOME,
  WISDOM_AGENT_HOME, WISDOM_DEMO_HOME,
  WISDOM_PRIVY_DID, WISDOM_TEAM_ORG_ID, WISDOM_TEAM_ORG_SLUG
EOF
}

port_pid() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1
}

messaging_gateway_pid() {
  (
    export HERMES_HOME="${AGENT_HOME}"
    export HERMES_SHARED_AUTH_DIR="${DEMO_HOME}/shared"
    export HERMES_WISDOM_QUIET=1
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"
    "${HERMES_WISDOM_PYTHON}" - "${AGENT_HOME}" <<'PY'
from pathlib import Path
import sys

from gateway.control_socket import identify_gateway

identity = identify_gateway(Path(sys.argv[1]), timeout=0.25)
if identity and identity.get("pid"):
    print(identity["pid"])
PY
  )
}

status() {
  local port label pid messaging_pid
  while read -r port label; do
    pid="$(port_pid "${port}" || true)"
    if [[ -n "${pid}" ]]; then
      printf '%-18s up (:%s, pid %s)\n' "${label}" "${port}" "${pid}"
    else
      printf '%-18s down (:%s)\n' "${label}" "${port}"
    fi
  done <<'EOF'
3111 Portal
3112 Privy mock
3114 LiteLLM mock
8787 Gateway
9119 Dashboard API
5173 Dashboard UI
5174 Desktop renderer
EOF

  messaging_pid="$(messaging_gateway_pid 2>/dev/null || true)"
  if [[ -n "${messaging_pid}" ]]; then
    printf '%-18s up (pid %s)\n' "Messaging gateway" "${messaging_pid}"
  else
    printf '%-18s down\n' "Messaging gateway"
  fi
}

require_directory() {
  [[ -d "$1" ]] || {
    echo "error: required checkout not found: $1" >&2
    exit 1
  }
}

require_free_port() {
  local port="$1" label="$2" pid
  pid="$(port_pid "${port}" || true)"
  [[ -z "${pid}" ]] || {
    echo "error: ${label} port ${port} is already in use by pid ${pid}" >&2
    echo "hint: stop the existing demo stack, then run this command again" >&2
    exit 1
  }
}

wait_for_port() {
  local port="$1" label="$2"
  for _ in $(seq 1 120); do
    if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
      echo "ready: ${label} (:${port})"
      return 0
    fi
    sleep 0.5
  done
  echo "error: ${label} did not bind port ${port}" >&2
  return 1
}

wait_for_http() {
  local url="$1" expected="$2" label="$3" status_code
  for _ in $(seq 1 90); do
    status_code="$(curl -sS -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
    if [[ "${status_code}" == "${expected}" ]]; then
      echo "ready: ${label} (HTTP ${status_code})"
      return 0
    fi
    sleep 1
  done
  echo "error: ${label} did not return HTTP ${expected}" >&2
  return 1
}

wait_for_messaging_gateway() {
  local pid
  for _ in $(seq 1 120); do
    pid="$(messaging_gateway_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      echo "ready: Hermes messaging gateway (pid ${pid})"
      return 0
    fi
    sleep 0.5
  done
  echo "error: Hermes messaging gateway control socket did not become ready" >&2
  tail -n 40 "${STATE_DIR}/messaging-gateway.log" >&2 || true
  return 1
}

start_service() {
  local name="$1" cwd="$2"
  shift 2
  echo "starting: ${name} (log: ${STATE_DIR}/${name}.log)"
  (
    cd "${cwd}"
    exec "$@"
  ) >"${STATE_DIR}/${name}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

kill_tree() {
  local pid="$1" child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -TERM "${pid}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "stopping Collective Wisdom demo applications..."
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill_tree "${pid}"
  done
  echo "application processes stopped; Docker data was preserved"
}

start_portal_process() {
  local name="$1" command="$2"
  echo "starting: ${name} (log: ${STATE_DIR}/${name}.log)"
  (
    cd "${PORTAL_APP}"
    set -a
    # shellcheck disable=SC1091
    source e2e/browser/env.source
    set +a
    export HERMES_WISDOM_LOCAL=1
    export HERMES_SYNC_PLANE_URL="${GATEWAY_URL}"
    ulimit -n 65536
    exec bash -lc "${command}"
  ) >"${STATE_DIR}/${name}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

start_agent_process() {
  local name="$1" cwd="$2"
  shift 2
  echo "starting: ${name} (log: ${STATE_DIR}/${name}.log)"
  (
    cd "${cwd}"
    export HERMES_HOME="${AGENT_HOME}"
    export HERMES_SHARED_AUTH_DIR="${DEMO_HOME}/shared"
    export GATEWAY_URL
    export HERMES_WISDOM_QUIET=1
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"
    exec "$@"
  ) >"${STATE_DIR}/${name}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

start_messaging_gateway_process() {
  local name="messaging-gateway"
  echo "starting: Hermes messaging gateway (log: ${STATE_DIR}/${name}.log)"
  (
    cd "${ROOT}"
    export HERMES_HOME="${AGENT_HOME}"
    export HERMES_SHARED_AUTH_DIR="${DEMO_HOME}/shared"
    export GATEWAY_URL
    export HERMES_WISDOM_QUIET=1
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"

    while true; do
      set +e
      hermes gateway run --replace --external-supervisor -v
      gateway_exit=$?
      set -e

      case "${gateway_exit}" in
        0)
          echo "Hermes messaging gateway stopped cleanly"
          exit 0
          ;;
        75)
          echo "Hermes messaging gateway requested restart; relaunching"
          ;;
        78)
          echo "Hermes messaging gateway has a fatal configuration error; not restarting" >&2
          exit 78
          ;;
        *)
          echo "Hermes messaging gateway exited with status ${gateway_exit}; retrying in 1 second" >&2
          sleep 1
          ;;
      esac
    done
  ) >"${STATE_DIR}/${name}.log" 2>&1 &
  CHILD_PIDS+=("$!")
}

authenticate_demo() {
  local auth_record="${STATE_DIR}/portal-login.json"
  local -a relogin_args=(
    --privy-did "${PRIVY_DID}"
    --org-id "${TEAM_ORG_ID}"
  )

  # The Portal helper needs --hermes-home only for the first device-code
  # bootstrap. On repeat launches an existing shared credential is imported
  # without a device code, which the helper correctly refuses to treat as a
  # fresh authorization. The Wisdom setup below refreshes and revalidates the
  # existing team-scoped credential instead.
  if [[ ! -s "${DEMO_HOME}/shared/nous_auth.json" ]]; then
    relogin_args+=(--hermes-home "${DEMO_HOME}")
  fi

  export HERMES_HOME="${AGENT_HOME}"
  export HERMES_SHARED_AUTH_DIR="${DEMO_HOME}/shared"
  export GATEWAY_URL
  export HERMES_WISDOM_QUIET=1
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/wisdom-demo-env.sh"

  (
    cd "${PORTAL_APP}"
    set -a
    # shellcheck disable=SC1091
    source e2e/browser/env.source
    set +a
    pnpm exec tsx scripts/relogin-account.ts "${relogin_args[@]}"
  ) >"${auth_record}"
  chmod 600 "${auth_record}"

  hermes config set sync.base_url "${GATEWAY_URL}" --force >/dev/null
  hermes config set wisdom.portal_url "${PORTAL_URL}" --force >/dev/null
  local setup_ok=""
  for _ in $(seq 1 30); do
    if hermes wisdom setup --accept-disclosure --json \
      >"${STATE_DIR}/wisdom-setup.json" 2>"${STATE_DIR}/wisdom-setup.log"; then
      setup_ok=1
      break
    fi
    sleep 1
  done
  if [[ -z "${setup_ok}" ]]; then
    echo "error: Hermes Wisdom setup did not become ready" >&2
    tail -n 20 "${STATE_DIR}/wisdom-setup.log" >&2 || true
    return 1
  fi

  open_portal_login "${auth_record}"
}

open_portal_login() {
  local auth_record="$1"
  local handoff_pid

  AUTH_FILE="${auth_record}" \
    PORTAL_REDIRECT="${PORTAL_URL}/orgs/${TEAM_ORG_SLUG}/wisdom" \
    node -e '
      const http = require("node:http");
      const fs = require("node:fs");
      const record = JSON.parse(fs.readFileSync(process.env.AUTH_FILE, "utf8"));
      const timeout = setTimeout(() => {
        server.close(() => {
          process.exitCode = 1;
        });
      }, 30000);
      const server = http.createServer((_request, response) => {
        clearTimeout(timeout);
        response.statusCode = 302;
        for (const cookie of record.cookies) {
          response.appendHeader(
            "Set-Cookie",
            `${cookie.name}=${cookie.value}; Path=/; SameSite=Lax`,
          );
        }
        response.setHeader("Location", process.env.PORTAL_REDIRECT);
        response.end();
        setTimeout(() => server.close(), 1000);
      });
      server.listen(3120, "127.0.0.1");
    ' >"${STATE_DIR}/browser-login.log" 2>&1 &
  handoff_pid="$!"
  CHILD_PIDS+=("${handoff_pid}")
  wait_for_port 3120 "browser login handoff"
  open "http://127.0.0.1:3120/login"
  wait "${handoff_pid}"
}

login() {
  local auth_record="${STATE_DIR}/portal-login.json"
  local -a relogin_args=(
    --privy-did "${PRIVY_DID}"
    --org-id "${TEAM_ORG_ID}"
  )

  require_directory "${PORTAL_APP}"
  mkdir -p "${STATE_DIR}"
  require_free_port 3120 "browser-login"
  [[ -n "$(port_pid 3111 || true)" ]] || {
    echo "error: Portal is not running on ${PORTAL_URL}" >&2
    echo "hint: start the demo with scripts/wisdom-demo-stack.sh up" >&2
    exit 1
  }

  (
    cd "${PORTAL_APP}"
    set -a
    # shellcheck disable=SC1091
    source e2e/browser/env.source
    set +a
    pnpm exec tsx scripts/relogin-account.ts "${relogin_args[@]}"
  ) >"${auth_record}"
  chmod 600 "${auth_record}"

  open_portal_login "${auth_record}"
  echo "Portal demo login refreshed for ${TEAM_ORG_SLUG}"
}

require_publisher_demo_stack() {
  local port
  [[ "${TEAM_ORG_ID}" == "nas_organisation:wisdom-local" ]] || {
    echo "error: publisher fixture only supports nas_organisation:wisdom-local" >&2
    exit 1
  }
  [[ "${TEAM_ORG_SLUG}" == "wisdom-local" ]] || {
    echo "error: publisher fixture only supports the wisdom-local slug" >&2
    exit 1
  }
  [[ "${PORTAL_URL}" =~ ^http://(127\.0\.0\.1|localhost)(:[0-9]+)?$ ]] || {
    echo "error: publisher fixture requires a loopback HTTP Portal" >&2
    exit 1
  }
  [[ "${GATEWAY_URL}" =~ ^http://(127\.0\.0\.1|localhost)(:[0-9]+)?$ ]] || {
    echo "error: publisher fixture requires a loopback HTTP Gateway" >&2
    exit 1
  }
  command -v jq >/dev/null || {
    echo "error: publisher fixture requires jq" >&2
    exit 1
  }
  for port in 3111 3112 8787; do
    [[ -n "$(port_pid "${port}" || true)" ]] || {
      echo "error: local demo service is not listening on port ${port}" >&2
      exit 1
    }
  done
}

publisher_portal_script() {
  (
    cd "${PORTAL_APP}"
    set -a
    # shellcheck disable=SC1091
    source e2e/browser/env.source
    set +a
    export HERMES_WISDOM_LOCAL=1
    export HERMES_WISDOM_QUIET=1
    # Ensure helpers which spawn `hermes` use this worktree and a complete
    # runtime, not whichever global CLI happens to be first on PATH.
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"
    pnpm exec tsx "$@"
  )
}

publisher_hermes() {
  (
    export HERMES_HOME="${PUBLISHER_HOME}"
    export HERMES_SHARED_AUTH_DIR="${PUBLISHER_HOME}/shared"
    export GATEWAY_URL
    export HERMES_WISDOM_QUIET=1
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"
    hermes "$@"
  )
}

publisher_setup() {
  require_directory "${PORTAL_APP}"
  require_publisher_demo_stack
  mkdir -p "${STATE_DIR}" "${PUBLISHER_HOME}/shared"

  local account_record fixture_record publisher_did publisher_user recipient_user
  account_record="$(mktemp "${STATE_DIR}/publisher-account.XXXXXX")"
  fixture_record="$(mktemp "${STATE_DIR}/publisher-membership.XXXXXX")"
  trap "rm -f -- '${account_record}' '${fixture_record}'" RETURN

  if [[ -s "${PUBLISHER_METADATA}" ]]; then
    publisher_did="$(jq -er '.publisherPrivyDid' "${PUBLISHER_METADATA}")"
  else
    publisher_portal_script scripts/mint-account.ts \
      --email "wisdom-publisher-$(date +%s)@example.com" \
      --name "Wisdom Demo Publisher" >"${account_record}"
    chmod 600 "${account_record}"
    publisher_did="$(jq -er '.privyDid' "${account_record}")"
  fi

  publisher_portal_script scripts/prepare-wisdom-demo-member.ts \
    --publisher-privy-did "${publisher_did}" \
    --recipient-privy-did "${PRIVY_DID}" \
    --org-id "${TEAM_ORG_ID}" >"${fixture_record}"
  publisher_user="$(jq -er '.publisherUserId' "${fixture_record}")"
  recipient_user="$(jq -er '.recipientUserId' "${fixture_record}")"
  [[ "${publisher_user}" != "${recipient_user}" ]] || {
    echo "error: publisher and recipient unexpectedly resolve to the same user" >&2
    exit 1
  }
  [[ "$(jq -er '.organizationId' "${fixture_record}")" == "${TEAM_ORG_ID}" ]] || {
    echo "error: publisher fixture resolved the wrong organization" >&2
    exit 1
  }

  jq -s '.[0] * .[1]' \
    <(if [[ -s "${PUBLISHER_METADATA}" ]]; then cat "${PUBLISHER_METADATA}"; else echo '{}'; fi) \
    "${fixture_record}" >"${PUBLISHER_METADATA}.tmp"
  mv "${PUBLISHER_METADATA}.tmp" "${PUBLISHER_METADATA}"
  chmod 600 "${PUBLISHER_METADATA}"

  if [[ ! -s "${PUBLISHER_HOME}/shared/nous_auth.json" ]]; then
    publisher_portal_script scripts/relogin-account.ts \
      --privy-did "${publisher_did}" \
      --org-id "${TEAM_ORG_ID}" \
      --hermes-home "${PUBLISHER_HOME}" >"${account_record}"
  fi

  publisher_hermes config set sync.base_url "${GATEWAY_URL}" --force >/dev/null
  publisher_hermes config set wisdom.portal_url "${PORTAL_URL}" --force >/dev/null
  publisher_hermes config set auxiliary.background_review.provider codex --force >/dev/null
  publisher_hermes config set auxiliary.background_review.model gpt-5.6-sol --force >/dev/null
  publisher_hermes wisdom setup --accept-disclosure --json \
    >"${STATE_DIR}/publisher-wisdom-setup.json"

  echo "Publisher fixture ready"
  echo "  publisher: ${publisher_user}"
  echo "  recipient: ${recipient_user}"
  echo "  org:       ${TEAM_ORG_ID}"
  echo "  home:      ${PUBLISHER_HOME}"
}

publisher_write_skill() {
  local slug="$1" version="$2" skill_dir="${PUBLISHER_HOME}/skills/${1}"
  mkdir -p "${skill_dir}"
  printf '%s\n' \
    '---' \
    "name: ${slug}" \
    "description: Verify cross-surface Collective Wisdom delivery for version ${version}." \
    'metadata:' \
    '  hermes:' \
    '    editorial_name: Cross-Surface Notification Demo' \
    "    editorial_description: A local team skill used to verify install and update notifications (v${version})." \
    '---' \
    '# Cross-Surface Notification Demo' \
    '' \
    "When asked to run the Wisdom notification demo, report that version ${version} is active." \
    >"${skill_dir}/SKILL.md"
}

recipient_approve_wisdom_proposal() {
  local proposal_n="$1"
  (
    export HERMES_HOME="${AGENT_HOME}"
    export HERMES_SHARED_AUTH_DIR="${DEMO_HOME}/shared"
    export HERMES_WISDOM_QUIET=1
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/wisdom-demo-env.sh"
    "${HERMES_WISDOM_PYTHON}" - \
      "${DEMO_HOME}/shared/nous_auth.json" \
      "${GATEWAY_URL}" \
      "${proposal_n}" <<'PY'
import json
import sys
from pathlib import Path

import requests

auth_path, gateway_url, proposal_n = sys.argv[1:]
token = json.loads(Path(auth_path).read_text(encoding="utf-8"))["access_token"]
response = requests.post(
    f"{gateway_url.rstrip('/')}/v1/sync/org/proposals/{proposal_n}/approve",
    headers={"Authorization": f"Bearer {token}"},
    json={},
    timeout=30,
)
if response.status_code != 200:
    raise SystemExit(
        f"local recipient moderation failed ({response.status_code}): "
        f"{response.text[:500]}"
    )
print(response.text)
PY
  )
}

publisher_publish_current() {
  local slug="$1" version="$2" submitted draft_id approved state proposal_n moderated
  publisher_hermes wisdom suggest "${slug}" \
    --description "A local team skill used to verify install and update notifications (v${version})." \
    --json >"${STATE_DIR}/publisher-v${version}-prepared.json"
  submitted="$(publisher_hermes wisdom suggest "${slug}" \
    --description "A local team skill used to verify install and update notifications (v${version})." \
    --system-specification-json '{"hermes":{"minimum_version":"0.20.5"}}' \
    --json)"
  printf '%s\n' "${submitted}" >"${STATE_DIR}/publisher-v${version}-submitted.json"
  draft_id="$(jq -er '.draft.id' <<<"${submitted}")"
  publisher_hermes wisdom review "${draft_id}" --acknowledge --json \
    >"${STATE_DIR}/publisher-v${version}-review.json"
  approved="$(publisher_hermes wisdom approve "${draft_id}" --json)"
  printf '%s\n' "${approved}" >"${STATE_DIR}/publisher-v${version}-approval.json"
  state="$(jq -r '.publication.state // .state // empty' <<<"${approved}")"
  if [[ "${state}" == "pending_moderation" ]]; then
    proposal_n="$(jq -er '.publication.proposal_id' <<<"${approved}")"
    moderated="$(recipient_approve_wisdom_proposal "${proposal_n}")"
    printf '%s\n' "${moderated}" \
      >"${STATE_DIR}/publisher-v${version}-moderation.json"
    state="$(jq -r '.state // empty' <<<"${moderated}")"
  fi
  [[ "${state}" == "published" || "${state}" == "approved" ]] || {
    echo "error: publisher v${version} ended in '${state:-unknown}', expected published" >&2
    exit 1
  }
  jq --arg slug "${slug}" --argjson version "${version}" \
    '.skillSlug=$slug | .publishedVersion=$version' \
    "${PUBLISHER_METADATA}" >"${PUBLISHER_METADATA}.tmp"
  mv "${PUBLISHER_METADATA}.tmp" "${PUBLISHER_METADATA}"
  chmod 600 "${PUBLISHER_METADATA}"
  echo "Published ${slug} v${version} as a different wisdom-local member"
}

publisher_v1() {
  publisher_setup >/dev/null
  local slug
  slug="${WISDOM_PUBLISHER_SKILL_SLUG:-org-notification-demo-$(date +%Y%m%d%H%M%S)}"
  publisher_write_skill "${slug}" 1
  publisher_publish_current "${slug}" 1
}

publisher_v2() {
  publisher_setup >/dev/null
  [[ -s "${PUBLISHER_METADATA}" ]] || {
    echo "error: run publisher-v1 first" >&2
    exit 1
  }
  local slug published_version
  slug="$(jq -er '.skillSlug' "${PUBLISHER_METADATA}")"
  published_version="$(jq -er '.publishedVersion' "${PUBLISHER_METADATA}")"
  [[ "${published_version}" -eq 1 ]] || {
    echo "error: publisher-v2 requires a completed v1 fixture" >&2
    exit 1
  }
  publisher_write_skill "${slug}" 2
  publisher_publish_current "${slug}" 2
}

up() {
  require_directory "${PORTAL_APP}"
  require_directory "${GATEWAY_REPO}"
  require_directory "${ROOT}/apps/desktop"
  mkdir -p "${STATE_DIR}" "${DEMO_HOME}/shared"

  for spec in \
    "3111 Portal" \
    "3112 Privy-mock" \
    "3114 LiteLLM-mock" \
    "8787 Gateway" \
    "9119 Dashboard-API" \
    "5173 Dashboard-UI" \
    "5174 Desktop-renderer" \
    "3120 browser-login"; do
    # shellcheck disable=SC2086
    require_free_port ${spec}
  done

  trap cleanup EXIT INT TERM

  docker compose \
    -p codex-wisdom-portal \
    -f "${PORTAL_APP}/docker-compose.yaml" \
    up -d postgres redis serverless-redis-http >/dev/null
  if docker inspect "${MINIO_CONTAINER}" >/dev/null 2>&1; then
    docker start "${MINIO_CONTAINER}" >/dev/null
  else
    echo "error: ${MINIO_CONTAINER} is missing; create the UAT MinIO container first" >&2
    exit 1
  fi

  start_portal_process "privy-mock" "exec npx tsx e2e/mocks/privy/main.ts"
  start_portal_process "litellm-mock" "exec npx tsx e2e/mocks/litellm/main.ts"
  wait_for_port 3112 "Privy mock"
  wait_for_port 3114 "LiteLLM mock"

  local minio_user minio_password
  minio_user="$(docker inspect "${MINIO_CONTAINER}" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MINIO_ROOT_USER=//p')"
  minio_password="$(docker inspect "${MINIO_CONTAINER}" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MINIO_ROOT_PASSWORD=//p')"
  start_service "gateway" "${GATEWAY_REPO}" env \
    PORT=8787 \
    DATABASE_URL="${PG_URL}/gateway_gateway" \
    NOUS_PORTAL_URL="${PORTAL_URL}" \
    SYNC_ENABLED=1 \
    SYNC_OBJECT_STORE_REGION=us-east-1 \
    SYNC_OBJECT_STORE_BUCKET="${MINIO_BUCKET}" \
    SYNC_OBJECT_STORE_ACCESS_KEY_ID="${minio_user}" \
    SYNC_OBJECT_STORE_SECRET_ACCESS_KEY="${minio_password}" \
    SYNC_OBJECT_STORE_ENDPOINT="${MINIO_ENDPOINT}" \
    WISDOM_REGISTRY=1 \
    npm run dev
  wait_for_port 8787 "Gateway"
  wait_for_http "${GATEWAY_URL}/healthz" 200 "Gateway health"

  start_portal_process \
    "portal" \
    "exec pnpm exec next dev --turbo --hostname 127.0.0.1 --port 3111"
  wait_for_port 3111 "Portal"
  wait_for_http "${PORTAL_URL}/api/oauth/account" 401 "Portal OAuth issuer"

  authenticate_demo

  start_messaging_gateway_process
  wait_for_messaging_gateway

  start_agent_process \
    "dashboard-api" "${ROOT}" \
    hermes dashboard --host 127.0.0.1 --port 9119 --no-open --isolated --skip-build
  wait_for_port 9119 "Dashboard API"

  start_service "dashboard-ui" "${ROOT}/web" env \
    HERMES_DASHBOARD_URL=http://127.0.0.1:9119 \
    npm run dev -- --host 127.0.0.1 --port 5173
  wait_for_port 5173 "Dashboard UI"

  start_agent_process "desktop" "${ROOT}" npm run dev --workspace apps/desktop
  wait_for_port 5174 "Desktop renderer"
  open "http://127.0.0.1:5173/skills"

  echo
  echo "Collective Wisdom demo is ready"
  echo "  Portal:    ${PORTAL_URL}/orgs/${TEAM_ORG_SLUG}/wisdom"
  echo "  Management:${PORTAL_URL}/orgs/${TEAM_ORG_SLUG}/wisdom/admin"
  echo "  Dashboard: http://127.0.0.1:5173/skills"
  echo "  Messaging: supervised (log: ${STATE_DIR}/messaging-gateway.log)"
  echo "  CLI:       HERMES_HOME=${AGENT_HOME} HERMES_SHARED_AUTH_DIR=${DEMO_HOME}/shared ${ROOT}/scripts/wisdom-demo-env.sh -- hermes wisdom status"
  echo
  echo "Keep this terminal open. Press Ctrl-C to stop application processes."
  wait
}

main() {
  case "${1:-up}" in
    up) up ;;
    login) login ;;
    status) status ;;
    publisher-setup) publisher_setup ;;
    publisher-v1) publisher_v1 ;;
    publisher-v2) publisher_v2 ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
