#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PID=""

cleanup() {
  if [[ -n "${BACKEND_PID}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM EXIT

cd "${ROOT_DIR}"
uvicorn backend.main:app --reload --port 8647 &
BACKEND_PID=$!

cd frontend
npm run dev
