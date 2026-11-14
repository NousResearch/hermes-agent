#!/bin/bash
set -euo pipefail
companion_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$companion_root"
companion_mode="${1:-verify}"
companion_source="${2:-$HOME/.hermes/hermes-agent}"
if [[ "$companion_mode" != "verify" && "$companion_mode" != "build" && "$companion_mode" != "open" ]]; then
  printf 'Usage: ./rebuild.sh [verify|build|open] [path-to-hermes-source]\n' >&2
  exit 2
fi
if [[ "$companion_mode" == "open" ]]; then
  /usr/bin/open HermesCompanion.xcodeproj
  exit 0
fi
companion_run="BuildEvidence/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$companion_run"
if [[ "$companion_mode" == "verify" ]]; then
  python3 -m unittest discover -s scripts/tests -v 2>&1 | tee "$companion_run/checker-tests.log"
  python3 scripts/check_backend.py --hermes-source "$companion_source" | tee "$companion_run/backend-surface.json"
  swift test --package-path Core 2>&1 | tee "$companion_run/core-tests.log"
  swift test --package-path Features 2>&1 | tee "$companion_run/feature-tests.log"
fi
if [[ ! -d HermesCompanion.xcodeproj ]]; then
  command -v xcodegen >/dev/null 2>&1 || { printf 'Missing Xcode project; install XcodeGen to regenerate it.\n' >&2; exit 1; }
  xcodegen generate
fi
xcodebuild -version > "$companion_run/xcode-version.txt"
git rev-parse HEAD > "$companion_run/app-head.txt"
git --no-optional-locks status --porcelain > "$companion_run/app-status.txt"
xcodebuild -project HermesCompanion.xcodeproj -scheme HermesCompanion \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath BuildEvidence/DerivedData CODE_SIGNING_ALLOWED=NO build \
  > "$companion_run/build.log" 2>&1 || { tail -80 "$companion_run/build.log"; exit 1; }
printf 'Simulator build passed. Evidence: %s\n' "$companion_run"
printf 'Phone signing, authenticated runtime tests, and UI tests are separate checks.\n'
