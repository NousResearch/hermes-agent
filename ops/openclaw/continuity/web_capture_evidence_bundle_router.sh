#!/usr/bin/bash
set -euo pipefail

# Web Capture Evidence Bundle Router
# Operator-facing entry point for web capture → bundle → validation → triage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Defaults
INDEX_PATH=""
OUTPUT_DIR=""
FORCE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --index)
      INDEX_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 --index PATH --output-dir DIR [--force] [--verbose]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "${INDEX_PATH}" ]]; then echo "ERROR: --index required" >&2; exit 1; fi
if [[ -z "${OUTPUT_DIR}" ]]; then echo "ERROR: --output-dir required" >&2; exit 1; fi
if [[ ! -f "${INDEX_PATH}" ]]; then echo "ERROR: Index not found: ${INDEX_PATH}" >&2; exit 1; fi

mkdir -p "${OUTPUT_DIR}" || exit 1
if [[ ! -w "${OUTPUT_DIR}" ]]; then echo "ERROR: Output dir not writable" >&2; exit 1; fi

# Source venv
source "${REPO_ROOT}/.venv/bin/activate" || { echo "ERROR: venv activation failed" >&2; exit 1; }

# Generate IDs
INDEX_NAME="$(basename "${INDEX_PATH}" .json)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%SZ)"
BUNDLE_ID="b8ev_webcap_${INDEX_NAME}_${TIMESTAMP}"
TRIAGE_ID="tri_wb_evidence_${TIMESTAMP}"
FINAL_BUNDLE_PATH="${OUTPUT_DIR}/${BUNDLE_ID}.json"
TRIAGE_PATH="${OUTPUT_DIR}/${TRIAGE_ID}.json"

# Working directory
WORK_DIR="$(mktemp -d)"
trap "rm -rf '${WORK_DIR}'" EXIT

# Step 1: Validate artifacts exist
if [[ "${VERBOSE}" == true ]]; then echo "[1/5] Validating artifacts..." >&2; fi

python3 "${REPO_ROOT}/ops/openclaw/continuity/validate_web_capture_artifacts.py" \
  --index "${INDEX_PATH}" || exit 1

# Step 2: Generate bundle
if [[ "${VERBOSE}" == true ]]; then echo "[2/5] Generating bundle..." >&2; fi

BUNDLE_PATH="${WORK_DIR}/${BUNDLE_ID}.json"
python3 "${REPO_ROOT}/scripts/web_capture_ui_evidence_pack_bridge.py" \
  --index "${INDEX_PATH}" --out "${BUNDLE_PATH}" --bundle-id "${BUNDLE_ID}" --json \
  > "${WORK_DIR}/bridge.json" || {
  echo "ERROR: Bundle generation failed" >&2
  cat "${WORK_DIR}/bridge.json" >&2
  exit 1
}

# Step 3: Validate bundle
if [[ "${VERBOSE}" == true ]]; then echo "[3/5] Validating bundle..." >&2; fi

SCHEMA_PATH="${REPO_ROOT}/docs/ops/schemas/b8_ui_evidence_bundle.schema.json"
python3 "${REPO_ROOT}/scripts/b8_ui_evidence_bundle_validate.py" \
  --schema "${SCHEMA_PATH}" --bundle "${BUNDLE_PATH}" --pretty \
  > "${WORK_DIR}/validation.json" || {
  echo "ERROR: Validation failed" >&2
  cat "${WORK_DIR}/validation.json" >&2
  exit 1
}

# Check validation
python3 -c "import json, sys; d=json.load(sys.stdin); sys.exit(0 if d.get('ok') else 1)" < "${WORK_DIR}/validation.json" || {
  echo "ERROR: Validation failed" >&2
  cat "${WORK_DIR}/validation.json" >&2
  exit 1
}

# Step 4: Copy bundle
if [[ "${VERBOSE}" == true ]]; then echo "[4/5] Copying bundle..." >&2; fi

if [[ -f "${FINAL_BUNDLE_PATH}" && "${FORCE}" != true ]]; then
  echo "ERROR: Bundle exists (use --force): ${FINAL_BUNDLE_PATH}" >&2
  exit 1
fi

cp "${BUNDLE_PATH}" "${FINAL_BUNDLE_PATH}" || {
  echo "ERROR: Failed to copy bundle" >&2
  exit 1
}

# Step 5: Generate triage
if [[ "${VERBOSE}" == true ]]; then echo "[5/5] Generating triage record..." >&2; fi

cat > "${TRIAGE_PATH}" <<EOF
{
  "triage_record_id": "${TRIAGE_ID}",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "severity": "observation",
  "category": "web_capture_evidence",
  "bundle_id": "${BUNDLE_ID}",
  "bundle_path": "${FINAL_BUNDLE_PATH}",
  "validation_gate_log": [
    {"gate": "screenshot_exists", "status": "passed"},
    {"gate": "state_snapshot_fresh", "status": "passed"},
    {"gate": "schema_valid", "status": "passed"},
    {"gate": "semantic_valid", "status": "passed"}
  ],
  "operator_actions": [
    {"action": "view_bundle", "command": "cat ${FINAL_BUNDLE_PATH}"},
    {"action": "validate_bundle", "command": "python3 ${REPO_ROOT}/scripts/b8_ui_evidence_bundle_validate.py --schema ${SCHEMA_PATH} --bundle ${FINAL_BUNDLE_PATH}"}
  ],
  "source_index": "${INDEX_PATH}",
  "provenance": {
    "router_version": "2026-04-05",
    "bridge_script": "scripts/web_capture_ui_evidence_pack_bridge.py",
    "validator_script": "scripts/b8_ui_evidence_bundle_validate.py"
  }
}
EOF

# Success output
echo '{"ok": true, "bundle_path": "'"${FINAL_BUNDLE_PATH}"'", "triage_path": "'"${TRIAGE_PATH}"'", "bundle_id": "'"${BUNDLE_ID}"'", "triage_id": "'"${TRIAGE_ID}"'"}'