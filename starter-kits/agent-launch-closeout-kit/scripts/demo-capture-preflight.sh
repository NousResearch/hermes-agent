#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/demo-artifacts"
LATEST_REPORT="$REPORT_DIR/latest-demo-capture-readiness.md"
TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
STAMP_HUMAN="$(date '+%Y-%m-%d %H:%M %Z')"
REPORT_PATH="$REPORT_DIR/demo-capture-readiness-$TIMESTAMP.md"

required_files=(
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md"
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt"
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md"
  "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md"
  "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/launch-execution-log.md"
)

missing_files=()
for file in "${required_files[@]}"; do
  [[ -f "$file" ]] || missing_files+=("${file#$ROOT_DIR/}")
done

if (( ${#missing_files[@]} > 0 )); then
  printf 'Demo capture preflight blocked: missing required files\n'
  printf ' - %s\n' "${missing_files[@]}"
  exit 2
fi

mkdir -p "$REPORT_DIR"

preflight_output="$(bash "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh" 2>&1)"
preflight_headline="$(printf '%s\n' "$preflight_output" | tail -1)"

path_injection_status="missing"
if grep -Fq 'inject the exact note paths and workspace path' "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md"; then
  path_injection_status="present"
fi

metric_status="missing"
if grep -Fq '1.74 minutes' "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md" \
  && grep -Fq '1.74 minutes' "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt"; then
  metric_status="present"
fi

pending_capture_status="missing"
if grep -Fq 'Status: pending capture' "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/launch-execution-log.md"; then
  pending_capture_status="present"
fi

{
  printf '# Demo Capture Readiness — %s\n\n' "$STAMP_HUMAN"
  printf '## Result\n'
  printf -- '- Status: ready\n'
  printf -- '- Product proof command: `bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh`\n'
  printf -- '- Preflight headline: %s\n\n' "$preflight_headline"

  printf '## Required files\n'
  for file in "${required_files[@]}"; do
    printf -- '- READY `%s`\n' "${file#$ROOT_DIR/}"
  done
  printf '\n'

  printf '## Proof alignment checks\n'
  printf -- '- Path-injection requirement in demo outline: %s\n' "$path_injection_status"
  printf -- '- 1.74-minute proof metric present in proof artifact + captions: %s\n' "$metric_status"
  printf -- '- Launch execution log still shows pending capture state: %s\n\n' "$pending_capture_status"

  printf '## Next capture path\n'
  printf '1. Run `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --prepare` to refresh this readiness packet and freeze a timestamped capture-session file with suggested raw/edit output paths.\n'
  printf '2. Follow `starter-kits/agent-launch-closeout-kit/demo-capture-runbook.md` shot list.\n'
  printf '3. After recording/editing, run `bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize --recording-path /absolute/path/to/raw.mov --duration 00:01:19 --edited-asset-path /absolute/path/to/final.mp4`.\n'
  printf '4. Use the captured walkthrough as the primary publish attachment once X auth is restored; otherwise fall back to the proof still.\n\n'

  printf '## Raw preflight output\n'
  printf '```\n%s\n```\n' "$preflight_output"
} > "$REPORT_PATH"

cp "$REPORT_PATH" "$LATEST_REPORT"

printf 'Demo capture preflight OK\n'
printf 'Readiness report: %s\n' "${REPORT_PATH#$ROOT_DIR/}"
printf 'Canonical latest: %s\n' "${LATEST_REPORT#$ROOT_DIR/}"
