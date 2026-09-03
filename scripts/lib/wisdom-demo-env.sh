#!/usr/bin/env bash

# Shared environment resolver for the Collective Wisdom demo launchers.
#
# This file is sourced by scripts/wisdom-demo-env.sh and by the repo-local
# `hermes` shim. Keep it side-effect free until
# wisdom_demo_export_environment is called.

wisdom_demo_repo_root() {
  if [ -n "${HERMES_WISDOM_REPO:-}" ] \
    && [ -f "${HERMES_WISDOM_REPO}/hermes_cli/main.py" ]; then
    (cd "${HERMES_WISDOM_REPO}" && pwd)
    return
  fi

  local helper_dir
  helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${helper_dir}/../.." && pwd)
}

wisdom_demo_pick_python() {
  local repo_root="$1"
  local candidate

  if [ -n "${HERMES_WISDOM_PYTHON:-}" ]; then
    if [ ! -x "${HERMES_WISDOM_PYTHON}" ]; then
      echo "error: HERMES_WISDOM_PYTHON is not executable: ${HERMES_WISDOM_PYTHON}" >&2
      return 1
    fi
    printf '%s\n' "${HERMES_WISDOM_PYTHON}"
    return
  fi

  for candidate in \
    "${repo_root}/.venv/bin/python" \
    "${repo_root}/venv/bin/python" \
    "${HERMES_PYTHON:-}" \
    "${HERMES_HOME:-}/hermes-agent/venv/bin/python" \
    "${HOME:-}/.hermes/hermes-agent/venv/bin/python"
  do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  candidate="$(command -v python3 2>/dev/null || true)"
  if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return
  fi

  echo "error: no Python interpreter is available for the Wisdom worktree" >&2
  echo "hint: set HERMES_WISDOM_PYTHON to a Hermes development venv interpreter" >&2
  return 1
}

wisdom_demo_prepend_colon_path() {
  local value="$1"
  local current="$2"

  case ":${current}:" in
    *":${value}:"*) printf '%s\n' "${current}" ;;
    *)
      if [ -n "${current}" ]; then
        printf '%s:%s\n' "${value}" "${current}"
      else
        printf '%s\n' "${value}"
      fi
      ;;
  esac
}

wisdom_demo_export_environment() {
  local repo_root
  local python
  local demo_bin

  repo_root="$(wisdom_demo_repo_root)" || return 1
  python="$(wisdom_demo_pick_python "${repo_root}")" || return 1
  demo_bin="${repo_root}/scripts/wisdom-demo-bin"

  export HERMES_WISDOM_REPO="${repo_root}"
  export HERMES_WISDOM_PYTHON="${python}"
  export HERMES_PYTHON="${python}"
  export HERMES_DESKTOP_PYTHON="${python}"
  export HERMES_DESKTOP_HERMES_ROOT="${repo_root}"
  export PYTHONPATH
  PYTHONPATH="$(wisdom_demo_prepend_colon_path "${repo_root}" "${PYTHONPATH:-}")"
  export PATH
  PATH="$(wisdom_demo_prepend_colon_path "${demo_bin}" "${PATH:-}")"

  # Bash caches command lookups. Clear an older global `hermes` resolution so
  # the newly-prepended worktree shim takes effect immediately when sourced.
  hash -r 2>/dev/null || true
}
