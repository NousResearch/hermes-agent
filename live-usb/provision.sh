#!/usr/bin/env bash
# =============================================================================
# Hermes AgentCyber — USB Provisioning Script
# =============================================================================
# Injects a pre-configured ~/.hermes directory into a written USB drive's
# config partition (partition 3) so the first-boot wizard is skipped and
# the gateway starts automatically with your credentials.
#
# This is the "fleet" workflow: build one ISO, provision N USBs with
# different configs without rebuilding.
#
# Usage:
#   sudo ./provision.sh --usb /dev/sdb --config /path/to/dot-hermes-dir
#   sudo ./provision.sh --usb /dev/sdb --config /path/to/hermes-config.tar.gz
#   sudo ./provision.sh --usb /dev/sdb --env-file /path/to/.env \
#                       --telegram-token "xxx" --allowed-users "123,456" \
#                       --model-key "sk-ant-..."
# =============================================================================

set -euo pipefail

DEVICE=""
CONFIG_DIR=""
CONFIG_TARBALL=""
ENV_FILE=""
TG_TOKEN=""
TG_USERS=""
MODEL_KEY=""
MODEL_PROVIDER="anthropic"
MODEL_NAME="claude-opus-4-7-20251101"
AUDIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --usb)              DEVICE="$2";         shift 2 ;;
    --config)
      if [[ -d "$2" ]]; then CONFIG_DIR="$2";
      else                  CONFIG_TARBALL="$2"; fi
      shift 2 ;;
    --env-file)         ENV_FILE="$2";       shift 2 ;;
    --telegram-token)   TG_TOKEN="$2";       shift 2 ;;
    --allowed-users)    TG_USERS="$2";       shift 2 ;;
    --model-key)        MODEL_KEY="$2";      shift 2 ;;
    --model-provider)   MODEL_PROVIDER="$2"; shift 2 ;;
    --model-name)       MODEL_NAME="$2";     shift 2 ;;
    --audit)            AUDIT=true;          shift   ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

[[ -z "$DEVICE" ]]    && { echo "❌  --usb required";   exit 1; }
[[ $EUID -ne 0 ]]     && { echo "❌  Run as root";      exit 1; }
[[ ! -b "$DEVICE" ]]  && { echo "❌  Not a block device: $DEVICE"; exit 1; }

PROVISION_PART="${DEVICE}3"
if [[ ! -b "$PROVISION_PART" ]]; then
  echo "❌  Config partition ${PROVISION_PART} not found."
  echo "    Write the ISO first (./write_usb.sh) — it creates a config partition."
  exit 1
fi

MNT=$(mktemp -d)
mount "$PROVISION_PART" "$MNT"
cleanup() { umount "$MNT" 2>/dev/null || true; rm -rf "$MNT"; }
trap cleanup EXIT

# ---- Write config -----------------------------------------------------------
if [[ -n "$CONFIG_DIR" ]]; then
  tar czf "${MNT}/hermes-config.tar.gz" \
    -C "$(dirname "$CONFIG_DIR")" "$(basename "$CONFIG_DIR")"
  echo "✓  Config dir packed from ${CONFIG_DIR}"

elif [[ -n "$CONFIG_TARBALL" ]]; then
  cp "$CONFIG_TARBALL" "${MNT}/hermes-config.tar.gz"
  echo "✓  Config tarball copied"

else
  # Build a minimal config on the fly from CLI args
  TMP_CFG=$(mktemp -d)
  mkdir -p "${TMP_CFG}/.hermes"

  cat > "${TMP_CFG}/.hermes/config.yaml" << YAML
# Provisioned by provision.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
model:
  default: "${MODEL_PROVIDER}/${MODEL_NAME}"
  provider: "${MODEL_PROVIDER}"
  api_key: "${MODEL_KEY}"

toolsets:
  enabled: [cyber, web, terminal, file, delegation, todo, memory]
YAML

  if [[ -n "$TG_TOKEN" ]]; then
    cat >> "${TMP_CFG}/.hermes/config.yaml" << YAML

telegram:
  token: "${TG_TOKEN}"
  allowed_users: [${TG_USERS}]
  busy_input_mode: interrupt
YAML
  fi

  # .env
  {
    echo "HERMES_LIVE_MODE=gateway"
    [[ "$AUDIT" == "true" ]] && echo "HERMES_CYBER_AUDIT=true"
    [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]] && cat "$ENV_FILE"
  } > "${TMP_CFG}/.hermes/.env"

  chmod 600 "${TMP_CFG}/.hermes/config.yaml" "${TMP_CFG}/.hermes/.env"

  tar czf "${MNT}/hermes-config.tar.gz" -C "${TMP_CFG}" ".hermes"
  rm -rf "${TMP_CFG}"
  echo "✓  Config built and packed"
fi

sync
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅  USB provisioned successfully."
echo "  Plug in and boot — wizard will be skipped."
echo "═══════════════════════════════════════════════"
