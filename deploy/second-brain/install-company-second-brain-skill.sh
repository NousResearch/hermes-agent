#!/usr/bin/env bash
set -euo pipefail

TOKEN="${1:-}"
BASE_URL="${SECOND_BRAIN_BASE_URL:-__PUBLIC_BASE_URL__}"
PLACEHOLDER_BASE_URL="__PUBLIC""_BASE_URL__"
ASSET_VERSION="${SECOND_BRAIN_INSTALL_VERSION:-20260624-generic}"
SKILL_ROOT="${HERMES_HOME:-$HOME/.hermes}/skills/productivity/company-second-brain"
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

mkdir -p "$SKILL_ROOT"

if [[ -f "company-second-brain-skill.tar.gz" ]]; then
  tar -xzf company-second-brain-skill.tar.gz -C "$SKILL_ROOT" --strip-components=1
elif [[ -d "skills/productivity/company-second-brain" ]]; then
  cp -R skills/productivity/company-second-brain/. "$SKILL_ROOT/"
else
  echo "Downloading company-second-brain skill from $BASE_URL ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$BASE_URL/download/company-second-brain-skill.tar.gz?v=$ASSET_VERSION" -o "$WORK_DIR/company-second-brain-skill.tar.gz"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$WORK_DIR/company-second-brain-skill.tar.gz" "$BASE_URL/download/company-second-brain-skill.tar.gz?v=$ASSET_VERSION"
  else
    echo "curl or wget is required to download the skill bundle" >&2
    exit 1
  fi
  tar -xzf "$WORK_DIR/company-second-brain-skill.tar.gz" -C "$SKILL_ROOT" --strip-components=1
fi

chmod +x "$SKILL_ROOT/scripts/second-brain"

for template_file in "$SKILL_ROOT/SKILL.md" "$SKILL_ROOT/scripts/second-brain"; do
  if [[ -f "$template_file" ]]; then
    sed -i.bak "s#$PLACEHOLDER_BASE_URL#$BASE_URL#g" "$template_file"
    rm -f "$template_file.bak"
  fi
done

if [[ -z "$TOKEN" ]]; then
  printf "Paste your Company Second Brain token: "
  IFS= read -r TOKEN
fi

if [[ -z "$TOKEN" ]]; then
  echo "Token is required to connect." >&2
  exit 1
fi

"$SKILL_ROOT/scripts/second-brain" connect --base-url "$BASE_URL" --token "$TOKEN"

cat <<EOF

Installed company-second-brain skill to:
$SKILL_ROOT

Quick test:
$SKILL_ROOT/scripts/second-brain query "toi co the truy cap workspace nao?"
EOF
