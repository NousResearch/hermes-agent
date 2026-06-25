#!/usr/bin/env bash
set -euo pipefail

TOKEN="${1:-}"
BASE_URL="${SECOND_BRAIN_BASE_URL:-__PUBLIC_BASE_URL__}"
PLACEHOLDER_BASE_URL="__PUBLIC""_BASE_URL__"
ASSET_VERSION="${SECOND_BRAIN_INSTALL_VERSION:-20260624-generic}"
HERMES_SKILL_PARENT="${HERMES_HOME:-$HOME/.hermes}/skills/productivity"
CODEX_SKILL_PARENT="${CODEX_HOME:-$HOME/.codex}/skills/productivity"
SKILL_ROOT="$HERMES_SKILL_PARENT/company-second-brain"
START_SKILL_ROOT="$HERMES_SKILL_PARENT/company-second-brain-start"
CODEX_SKILL_ROOT="$CODEX_SKILL_PARENT/company-second-brain"
CODEX_START_SKILL_ROOT="$CODEX_SKILL_PARENT/company-second-brain-start"
BIN_DIR="${SECOND_BRAIN_BIN_DIR:-$HOME/.local/bin}"
WORK_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if [[ "$BASE_URL" == "$PLACEHOLDER_BASE_URL" ]]; then
  echo "SECOND_BRAIN_BASE_URL is required when running this installer outside a deployed gateway." >&2
  echo "Example: SECOND_BRAIN_BASE_URL=https://second-brain.example.com bash install-company-second-brain-skill.sh" >&2
  exit 1
fi

install_pack_from_bundle() {
  local bundle_path="$1"
  local skill_parent="$2"
  mkdir -p "$skill_parent"
  tar -xzf "$bundle_path" -C "$skill_parent"
}

install_pack_from_source() {
  local skill_parent="$1"
  mkdir -p "$skill_parent/company-second-brain" "$skill_parent/company-second-brain-start"
  cp -R skills/productivity/company-second-brain/. "$skill_parent/company-second-brain/"
  cp -R skills/productivity/company-second-brain-start/. "$skill_parent/company-second-brain-start/"
}

parents=("$HERMES_SKILL_PARENT")
if [[ "$CODEX_SKILL_PARENT" != "$HERMES_SKILL_PARENT" ]]; then
  parents+=("$CODEX_SKILL_PARENT")
fi

BUNDLE_PATH=""
if [[ -f "company-second-brain-skill.tar.gz" ]]; then
  BUNDLE_PATH="company-second-brain-skill.tar.gz"
elif [[ -d "skills/productivity/company-second-brain" && -d "skills/productivity/company-second-brain-start" ]]; then
  for skill_parent in "${parents[@]}"; do
    install_pack_from_source "$skill_parent"
  done
else
  echo "Downloading company-second-brain skills from $BASE_URL ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$BASE_URL/download/company-second-brain-skill.tar.gz?v=$ASSET_VERSION" -o "$WORK_DIR/company-second-brain-skill.tar.gz"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$WORK_DIR/company-second-brain-skill.tar.gz" "$BASE_URL/download/company-second-brain-skill.tar.gz?v=$ASSET_VERSION"
  else
    echo "curl or wget is required to download the skill bundle" >&2
    exit 1
  fi
  BUNDLE_PATH="$WORK_DIR/company-second-brain-skill.tar.gz"
fi

if [[ -n "$BUNDLE_PATH" ]]; then
  for skill_parent in "${parents[@]}"; do
    install_pack_from_bundle "$BUNDLE_PATH" "$skill_parent"
  done
fi

chmod +x "$SKILL_ROOT/scripts/second-brain"
if [[ -f "$CODEX_SKILL_ROOT/scripts/second-brain" ]]; then
  chmod +x "$CODEX_SKILL_ROOT/scripts/second-brain"
fi

for template_file in \
  "$SKILL_ROOT/SKILL.md" \
  "$SKILL_ROOT/scripts/second-brain" \
  "$START_SKILL_ROOT/SKILL.md" \
  "$CODEX_SKILL_ROOT/SKILL.md" \
  "$CODEX_SKILL_ROOT/scripts/second-brain" \
  "$CODEX_START_SKILL_ROOT/SKILL.md"; do
  if [[ -f "$template_file" ]]; then
    sed -i.bak "s#$PLACEHOLDER_BASE_URL#$BASE_URL#g" "$template_file"
    rm -f "$template_file.bak"
  fi
done

mkdir -p "$BIN_DIR"
if ln -sf "$SKILL_ROOT/scripts/second-brain" "$BIN_DIR/second-brain"; then
  CLI_COMMAND="$BIN_DIR/second-brain"
else
  CLI_COMMAND="$SKILL_ROOT/scripts/second-brain"
fi

if [[ -z "$TOKEN" ]]; then
  printf "Paste your Company Second Brain token: "
  IFS= read -r TOKEN
fi

if [[ -z "$TOKEN" ]]; then
  echo "Token is required to connect." >&2
  exit 1
fi

TOKEN="$(printf '%s' "$TOKEN" | sed -E 's/^[[:space:]]*(Authorization:[[:space:]]*)?[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+//; s/[[:space:]]+$//')"

if [[ -z "$TOKEN" ]]; then
  echo "Token is required to connect." >&2
  exit 1
fi

"$SKILL_ROOT/scripts/second-brain" connect --base-url "$BASE_URL" --token "$TOKEN"

cat <<EOF

Installed company-second-brain skills to:
$SKILL_ROOT
$START_SKILL_ROOT
$CODEX_SKILL_ROOT
$CODEX_START_SKILL_ROOT

CLI command:
$CLI_COMMAND

Quick test:
$CLI_COMMAND query "toi co the truy cap workspace nao?"
EOF
