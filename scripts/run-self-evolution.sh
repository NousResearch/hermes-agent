#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_HOME="${HERMES_HOME:-$HOME/.hermes}"
SELF_EVO_REPO="${HERMES_SELF_EVOLUTION_REPO:-$PROFILE_HOME/hermes-agent-self-evolution}"
SELF_EVO_PYTHON="$SELF_EVO_REPO/.venv/bin/python"
EVAL_SOURCE="synthetic"
EXPORT_LIMIT="${SELF_EVO_SESSION_LIMIT:-250}"
HAS_OPTIMIZER_MODEL_ARG=0
HAS_EVAL_MODEL_ARG=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run-self-evolution.sh --skill <name> [--iterations N] [--eval-source synthetic|sessiondb|golden] [...]

What it does:
  - ensures the companion repo NousResearch/hermes-agent-self-evolution exists locally
  - installs its dependencies into a dedicated .venv via uv
  - when --eval-source sessiondb is used, exports Hermes state.db sessions into the
    legacy ~/.hermes/sessions/*.json shape expected by the companion repo
  - runs python -m evolution.skills.evolve_skill against THIS hermes-agent checkout

Environment overrides:
  HERMES_SELF_EVOLUTION_REPO   Override clone/install path for the companion repo
  SELF_EVO_SESSION_LIMIT       Max sessions exported for --eval-source sessiondb (default: 250)

Model selection:
  - upstream defaults stay in place when OPENAI_API_KEY is available
  - if OPENAI_API_KEY is missing but ANTHROPIC_API_KEY exists, the runner
    auto-adds --optimizer-model anthropic/claude-sonnet-4.6 and
    --eval-model anthropic/claude-sonnet-4.6
EOF
}

for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    --eval-source=sessiondb)
      EVAL_SOURCE="sessiondb"
      ;;
    --eval-source=synthetic)
      EVAL_SOURCE="synthetic"
      ;;
    --eval-source=golden)
      EVAL_SOURCE="golden"
      ;;
    --optimizer-model=*)
      HAS_OPTIMIZER_MODEL_ARG=1
      ;;
    --eval-model=*)
      HAS_EVAL_MODEL_ARG=1
      ;;
  esac
done

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[i]}" == "--eval-source" && $((i + 1)) -lt ${#args[@]} ]]; then
    EVAL_SOURCE="${args[i + 1]}"
  fi
  if [[ "${args[i]}" == "--optimizer-model" && $((i + 1)) -lt ${#args[@]} ]]; then
    HAS_OPTIMIZER_MODEL_ARG=1
  fi
  if [[ "${args[i]}" == "--eval-model" && $((i + 1)) -lt ${#args[@]} ]]; then
    HAS_EVAL_MODEL_ARG=1
  fi
done

mkdir -p "$PROFILE_HOME"
if [[ -f "$PROFILE_HOME/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROFILE_HOME/.env"
  set +a
fi

if [[ ! -d "$SELF_EVO_REPO/.git" ]]; then
  echo "[self-evo] cloning companion repo to $SELF_EVO_REPO"
  git clone --depth 1 https://github.com/NousResearch/hermes-agent-self-evolution "$SELF_EVO_REPO"
fi

if [[ ! -x "$SELF_EVO_PYTHON" ]]; then
  echo "[self-evo] creating venv"
  uv venv "$SELF_EVO_REPO/.venv"
fi

if ! "$SELF_EVO_PYTHON" -c 'import click, dspy, rich' >/dev/null 2>&1; then
  echo "[self-evo] installing companion repo dependencies"
  (
    cd "$SELF_EVO_REPO"
    uv pip install --python .venv/bin/python -e '.[dev]'
  )
fi

TEMP_HOME="$(mktemp -d)"
cleanup() {
  rm -rf "$TEMP_HOME"
}
trap cleanup EXIT

if [[ "$EVAL_SOURCE" == "sessiondb" ]]; then
  mkdir -p "$TEMP_HOME/.hermes/sessions"
  if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/venv/bin/activate"
  fi
  echo "[self-evo] exporting state.db sessions -> $TEMP_HOME/.hermes/sessions"
  python -m agent.self_evolution_bridge \
    --output-dir "$TEMP_HOME/.hermes/sessions" \
    --limit-sessions "$EXPORT_LIMIT"
fi

export HERMES_AGENT_REPO="${HERMES_AGENT_REPO:-$ROOT_DIR}"
if [[ "$EVAL_SOURCE" == "sessiondb" ]]; then
  export HOME="$TEMP_HOME"
fi

final_args=("$@")
if [[ $HAS_OPTIMIZER_MODEL_ARG -eq 0 && -z "${OPENAI_API_KEY:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  final_args+=(--optimizer-model "anthropic/claude-sonnet-4.6")
fi
if [[ $HAS_EVAL_MODEL_ARG -eq 0 && -z "${OPENAI_API_KEY:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  final_args+=(--eval-model "anthropic/claude-sonnet-4.6")
fi

exec "$SELF_EVO_PYTHON" -m evolution.skills.evolve_skill "${final_args[@]}"
