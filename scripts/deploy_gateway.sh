#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

pick_python() {
  local candidate
  for candidate in \
    "${PROJECT_DIR}/.venv/bin/python" \
    "${PROJECT_DIR}/venv/bin/python" \
    python3 \
    python; do
    if [[ "${candidate}" == /* ]]; then
      [[ -x "${candidate}" ]] && printf '%s\n' "${candidate}" && return 0
      continue
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python interpreter found. Expected .venv/bin/python, venv/bin/python, python3, or python." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m hermes_cli.deploy_gateway "$@"
