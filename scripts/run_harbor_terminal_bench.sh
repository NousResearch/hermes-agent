#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run Harbor Terminal-Bench with Hermes Agent through an OpenAI-compatible autoscaler gateway.

Required environment:
  AUTOSCALER_SSH_TARGET        SSH target for the gateway host, for example user@host
  AUTOSCALER_SSH_KEY           SSH private key path for the gateway host
  AUTOSCALER_REMOTE_GATEWAY    Remote gateway host:port reachable from AUTOSCALER_SSH_TARGET

Optional environment:
  HARBOR_DIR                   Harbor checkout path (default: ../harbor)
  HARBOR_REPO_URL              Harbor repo URL to clone if HARBOR_DIR is missing
  HARBOR_REF                   Harbor branch/tag/SHA to clone
  APPLY_HARBOR_PATCH=1         Apply bundled patch for unpatched upstream Harbor
  HERMES_MODEL                 Autoscaler model id (default: hermes-large)
  LOCAL_PORT                   Local forwarded port (default: 30090)
  N_TASKS                      Number of tasks to run (unset means full dataset)
  N_CONCURRENT                 Harbor trial concurrency (default: 1)
  JOB_NAME                     Harbor job name (default: hermes-large-tb-<timestamp>)
  EXCLUDE_TASK_NAME            Task glob to exclude (default: gpt2-codegolf)
  INCLUDE_TASK_NAME            Task glob to include instead of N_TASKS
  OPENAI_API_KEY               Dummy/custom provider key (default: dummy)
  ALLOW_HEAD_NODE_RUN=1        Override the head-node safety guard
  NO_TUNNEL=1                  Do not start an SSH tunnel; use an existing local gateway

Examples:
  AUTOSCALER_SSH_TARGET=user@example-host \
  AUTOSCALER_SSH_KEY=~/.ssh/id_ed25519 \
  AUTOSCALER_REMOTE_GATEWAY=10.0.0.10:30090 \
  ./scripts/run_harbor_terminal_bench.sh

  INCLUDE_TASK_NAME=cancel-async-tasks ./scripts/run_harbor_terminal_bench.sh
  N_TASKS=50 N_CONCURRENT=4 ./scripts/run_harbor_terminal_bench.sh
  unset N_TASKS; N_CONCURRENT=1 ./scripts/run_harbor_terminal_bench.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
harbor_dir="${HARBOR_DIR:-${repo_root}/../harbor}"
harbor_repo_url="${HARBOR_REPO_URL:-git@github.com:NousResearch/harbor-fork.git}"
harbor_ref="${HARBOR_REF:-hermes-custom-endpoint}"
apply_harbor_patch="${APPLY_HARBOR_PATCH:-0}"
harbor_patch="${HARBOR_PATCH:-${repo_root}/scripts/patches/harbor-hermes-custom-endpoint.patch}"
hermes_model="${HERMES_MODEL:-hermes-large}"
local_port="${LOCAL_PORT:-30090}"
n_tasks="${N_TASKS:-}"
n_concurrent="${N_CONCURRENT:-1}"
job_name="${JOB_NAME:-${hermes_model}-tb-$(date +%Y%m%d-%H%M%S)}"
exclude_task_name="${EXCLUDE_TASK_NAME:-gpt2-codegolf}"
include_task_name="${INCLUDE_TASK_NAME:-}"
api_key="${OPENAI_API_KEY:-dummy}"
local_base_url="http://127.0.0.1:${local_port}/v1"
docker_base_url="${DOCKER_BASE_URL:-http://host.docker.internal:${local_port}/v1}"

hostname_value="$(hostname -f 2>/dev/null || hostname)"
if [[ "${ALLOW_HEAD_NODE_RUN:-0}" != "1" ]] \
  && [[ "${hostname_value}" =~ (^|[-.])(hn[0-9]*|head)([-.]|$) ]]; then
  cat >&2 <<EOF
Refusing to launch Harbor eval on possible head node: ${hostname_value}

Run from a workstation or compute allocation. If you are certain this host is
safe, set ALLOW_HEAD_NODE_RUN=1.
EOF
  exit 64
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    usage >&2
    exit 2
  fi
}

ensure_harbor_checkout() {
  if [[ ! -f "${harbor_dir}/pyproject.toml" ]]; then
    if [[ -e "${harbor_dir}" ]]; then
      echo "HARBOR_DIR exists but is not a Harbor checkout: ${harbor_dir}" >&2
      exit 1
    fi

    echo "Cloning Harbor into ${harbor_dir}"
    if [[ -n "${harbor_ref}" ]]; then
      git clone --branch "${harbor_ref}" --depth 1 "${harbor_repo_url}" "${harbor_dir}"
    else
      git clone --depth 1 "${harbor_repo_url}" "${harbor_dir}"
    fi
  fi

  if [[ "${apply_harbor_patch}" == "1" ]]; then
    if [[ ! -f "${harbor_patch}" ]]; then
      echo "Missing Harbor patch: ${harbor_patch}" >&2
      exit 1
    fi

    (
      cd "${harbor_dir}"
      if git apply --check "${harbor_patch}" >/dev/null 2>&1; then
        git apply "${harbor_patch}"
        echo "Applied Harbor Hermes custom endpoint patch."
      elif git apply --reverse --check "${harbor_patch}" >/dev/null 2>&1; then
        echo "Harbor Hermes custom endpoint patch is already applied."
      else
        cat >&2 <<EOF
Could not apply Harbor patch cleanly.

This usually means Harbor changed upstream or already has a different version
of the Hermes custom endpoint fix. Inspect:
  ${harbor_patch}
  ${harbor_dir}/src/harbor/agents/installed/hermes.py
EOF
        exit 1
      fi
    )
  fi
}

curl_models() {
  curl -fsS --max-time 8 "${local_base_url}/models" >/dev/null
}

tunnel_pid=""
cleanup() {
  if [[ -n "${tunnel_pid}" ]]; then
    kill "${tunnel_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${NO_TUNNEL:-0}" != "1" ]] && ! curl_models; then
  require_env AUTOSCALER_SSH_TARGET
  require_env AUTOSCALER_SSH_KEY
  require_env AUTOSCALER_REMOTE_GATEWAY

  ssh -i "${AUTOSCALER_SSH_KEY}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -N \
    -L "${local_port}:${AUTOSCALER_REMOTE_GATEWAY}" \
    "${AUTOSCALER_SSH_TARGET}" &
  tunnel_pid="$!"
  sleep 2
  curl_models
fi

if ! docker run --rm curlimages/curl:latest -fsS --max-time 10 "${docker_base_url}/models" >/dev/null; then
  cat >&2 <<EOF
Docker could not reach the autoscaler at ${docker_base_url}.

On macOS/Windows, host.docker.internal should work. On Linux you may need to
run Harbor with host networking or set DOCKER_BASE_URL to a container-reachable
gateway URL.
EOF
  exit 1
fi

ensure_harbor_checkout

args=(
  --dataset terminal-bench@2.0
  --agent hermes
  --model "openai/${hermes_model}"
  --n-concurrent "${n_concurrent}"
  --job-name "${job_name}"
  -y
)

if [[ -n "${include_task_name}" ]]; then
  args+=(--include-task-name "${include_task_name}")
else
  if [[ -n "${n_tasks}" ]]; then
    args+=(--n-tasks "${n_tasks}")
  fi
  if [[ -n "${exclude_task_name}" ]]; then
    args+=(--exclude-task-name "${exclude_task_name}")
  fi
fi

cd "${harbor_dir}"
OPENAI_API_KEY="${api_key}" \
OPENAI_BASE_URL="${docker_base_url}" \
uv run --no-dev harbor run "${args[@]}"
