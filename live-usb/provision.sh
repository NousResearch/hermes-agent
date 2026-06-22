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
#
# High-consequence gate:
#   build/write/provision scripts require root plus exact operator approval.
#   Set HERMES_AGENTCYBER_LIVE_USB_APPROVAL for the approved maintenance
#   session, then pass --operator-approval with the exact same value, pass
#   --operator-approval-stdin and write the token to stdin, or omit both in an
#   interactive terminal to enter a silent prompt.
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
OPERATOR_APPROVAL=""
OPERATOR_APPROVAL_PROVIDED=false
OPERATOR_APPROVAL_STDIN=false

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
    --operator-approval)
      if [[ $# -lt 2 ]]; then
        echo "❌  --operator-approval requires a value" >&2
        exit 1
      fi
      OPERATOR_APPROVAL="$2"
      OPERATOR_APPROVAL_PROVIDED=true
      shift 2 ;;
    --operator-approval-stdin)
      OPERATOR_APPROVAL_STDIN=true
      shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

require_operator_approval() {
  local action="$1"
  if [[ -z "${HERMES_AGENTCYBER_LIVE_USB_APPROVAL:-}" ]]; then
    echo "❌  ${action} requires exact operator approval." >&2
    echo "    Set HERMES_AGENTCYBER_LIVE_USB_APPROVAL for the approved maintenance session." >&2
    echo "    Pass --operator-approval with the exact same value or enter it at the silent prompt." >&2
    echo "    Root/sudo alone is not sufficient." >&2
    return 1
  fi

  if [[ "$OPERATOR_APPROVAL_PROVIDED" != "true" ]]; then
    if [[ "$OPERATOR_APPROVAL_STDIN" == "true" ]]; then
      IFS= read -r OPERATOR_APPROVAL || true
      OPERATOR_APPROVAL_PROVIDED=true
    elif [[ -t 0 ]]; then
      printf "Operator approval token: " >&2
      IFS= read -r -s OPERATOR_APPROVAL
      printf "\n" >&2
      OPERATOR_APPROVAL_PROVIDED=true
    else
      echo "❌  ${action} requires --operator-approval or --operator-approval-stdin in non-interactive mode." >&2
      echo "    It must exactly match HERMES_AGENTCYBER_LIVE_USB_APPROVAL; no trimming or case normalization is applied." >&2
      echo "    Root/sudo alone is not sufficient." >&2
      return 1
    fi
  fi

  if [[ "$OPERATOR_APPROVAL" != "$HERMES_AGENTCYBER_LIVE_USB_APPROVAL" ]]; then
    echo "❌  Operator approval did not match exactly for ${action}." >&2
    echo "    No trimming, case normalization, or aliases are accepted." >&2
    echo "    Root/sudo alone is not sufficient." >&2
    return 1
  fi

  unset HERMES_AGENTCYBER_LIVE_USB_APPROVAL
  OPERATOR_APPROVAL=""
  OPERATOR_APPROVAL_PROVIDED=false
}

_canonical_removable_device() {
  local input="$1"
  local canonical=""
  canonical="$(readlink -f -- "$input" 2>/dev/null || true)"

  if [[ -z "$canonical" || "$canonical" != /dev/* ]]; then
    echo "❌  Target must resolve to a canonical /dev/... block device: $input" >&2
    return 1
  fi
  if [[ ! -b "$canonical" ]]; then
    echo "❌  Not a block device: $canonical" >&2
    return 1
  fi

  local device_type=""
  device_type="$(lsblk -dn -o TYPE -- "$canonical" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$device_type" != "disk" ]]; then
    echo "❌  Target must be a whole removable disk, not a partition or mapper: $canonical" >&2
    return 1
  fi

  local base="$(basename "$canonical")"
  local removable_path="/sys/class/block/${base}/removable"
  local removable=""
  if [[ -r "$removable_path" ]]; then
    removable="$(tr -d '[:space:]' < "$removable_path")"
  fi
  if [[ "$removable" != "1" ]]; then
    echo "❌  Refusing to provision: Linux does not verify ${canonical} as removable media." >&2
    echo "    Expected ${removable_path} to contain 1; got '${removable:-unreadable}'." >&2
    echo "    Root/operator approval is not enough without verifiable removable-media metadata." >&2
    return 1
  fi

  printf '%s\n' "$canonical"
}

_partition_path() {
  local disk="$1"
  local number="$2"
  if [[ "$disk" =~ [0-9]$ ]]; then
    printf '%sp%s\n' "$disk" "$number"
  else
    printf '%s%s\n' "$disk" "$number"
  fi
}

[[ -z "$DEVICE" ]]    && { echo "❌  --usb required";   exit 1; }
require_operator_approval "provision" || exit 1
if [[ $EUID -ne 0 ]]; then
  echo "❌  Run as root."
  echo "    Root/sudo alone is not sufficient; exact operator approval plus a canonical whole removable /dev disk with removable=1 are also required."
  exit 1
fi
DEVICE="$(_canonical_removable_device "$DEVICE")" || exit 1

PROVISION_PART="$(_partition_path "$DEVICE" 3)"
if [[ ! -b "$PROVISION_PART" ]]; then
  echo "❌  Config partition ${PROVISION_PART} not found."
  echo "    Write the ISO first (./write_usb.sh) — it creates a config partition."
  exit 1
fi

MNT=$(mktemp -d)
TMP_CFG=""
mount "$PROVISION_PART" "$MNT"
cleanup() {
  umount "$MNT" 2>/dev/null || true
  rm -rf "$MNT"
  if [[ -n "${TMP_CFG:-}" ]]; then
    rm -rf "$TMP_CFG"
  fi
}
trap cleanup EXIT

# ---- Write config -----------------------------------------------------------
if [[ -n "$CONFIG_DIR" ]]; then
  TMP_CFG=$(mktemp -d)
  mkdir -p "${TMP_CFG}/.hermes"
  tar cf - -C "$CONFIG_DIR" . | tar xf - -C "${TMP_CFG}/.hermes"
  tar czf "${MNT}/hermes-config.tar.gz" \
    -C "${TMP_CFG}" ".hermes"
  rm -rf "${TMP_CFG}"
  TMP_CFG=""
  echo "✓  Config dir packed from ${CONFIG_DIR} as .hermes"

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
