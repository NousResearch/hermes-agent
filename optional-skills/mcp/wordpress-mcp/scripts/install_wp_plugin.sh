#!/usr/bin/env bash
# install_wp_plugin.sh — Install and activate a WordPress MCP plugin via WP-CLI.
#
# Supports:
#   mcp-adapter  — official WordPress MCP Adapter (from pinned GitHub release)
#   ai-engine    — AI Engine plugin (from WordPress.org, pinned version)
#
# Security hardening:
#   - Pin version with --version= or prompt the user
#   - Download to a private temp directory (mktemp -d)
#   - Verify checksum when the environment provides WP_*_SHA256
#   - Never prints the generated Bearer Token or Application Password to stdout
#   - Avoids wp eval for routine configuration; uses it only with explicit approval
#   - Avoids dynamic PHP file mutation
#   - Avoids privilege escalation unless the user sets --allow-sudo
#   - Quoted paths and args throughout
#
# Usage:
#   bash install_wp_plugin.sh mcp-adapter --version=v0.6.1 --wp-path=/var/www/html
#   bash install_wp_plugin.sh ai-engine --version=4.2.0 --wp-path=/var/www/html
#   bash install_wp_plugin.sh mcp-adapter --version=v0.6.1 --wp-path=/var/www/html --dry-run
#
# Exit codes: 0 ok / 1 invalid args / 2 missing dependency / 3 download failed /
#             4 activate failed / 5 user did not approve / 6 checksum mismatch
set -euo pipefail

PLUGIN=""
VERSION=""
WP_PATH=""
WEB_USER=""
DRY_RUN=0
ALLOW_SUDO=0

while [ $# -gt 0 ]; do
  case "$1" in
    mcp-adapter|ai-engine) PLUGIN="$1"; shift ;;
    --version=*)   VERSION="${1#*=}"; shift ;;
    --wp-path=*)   WP_PATH="${1#*=}"; shift ;;
    --web-user=*)  WEB_USER="${1#*=}"; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --allow-sudo)  ALLOW_SUDO=1; shift ;;
    -h|--help)
      sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [ -z "$PLUGIN" ]; then
  echo "Error: specify plugin name (mcp-adapter or ai-engine)" >&2
  exit 1
fi
if [ -z "$VERSION" ]; then
  echo "Error: --version= is required. Pin a specific release (e.g. v0.6.1 or 4.2.0)." >&2
  exit 1
fi
if [ -z "$WP_PATH" ]; then
  echo "Error: --wp-path= is required (e.g. --wp-path=/var/www/html)" >&2
  exit 1
fi
if [ ! -d "$WP_PATH" ]; then
  echo "Error: WordPress path does not exist: $WP_PATH" >&2
  exit 1
fi

if ! command -v wp >/dev/null 2>&1; then
  echo "Error: wp-cli not found." >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found." >&2
  exit 2
fi

PLUGINS_DIR="$WP_PATH/wp-content/plugins"
TMPDIR=""

cleanup() {
  if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
  fi
}
trap cleanup EXIT

WP="wp --path=$WP_PATH"

# Detect web user if not specified
if [ -z "$WEB_USER" ]; then
  if [ -d "$PLUGINS_DIR" ]; then
    WEB_USER=$(stat -c '%U' "$PLUGINS_DIR" 2>/dev/null || echo "www-data")
  else
    WEB_USER="www-data"
  fi
fi

# Determine sudo behavior
SUDO=""
if [ ! -w "$PLUGINS_DIR" ]; then
  if [ "$ALLOW_SUDO" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
    echo "Warning: $PLUGINS_DIR is not writable by the current user." >&2
    echo "  The script will attempt to write without sudo. If it fails, re-run with --allow-sudo." >&2
  else
    SUDO="sudo"
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Would install $PLUGIN version $VERSION into $WP_PATH (web user: $WEB_USER)"
  if [ "$PLUGIN" = "mcp-adapter" ]; then
    echo "[dry-run] Would download pinned mcp-adapter release $VERSION"
  else
    echo "[dry-run] Would download ai-engine.$VERSION"
  fi
  echo "[dry-run] Would activate plugin and flush rewrites"
  exit 0
fi

TMPDIR=$(mktemp -d -t "wp-mcp-XXXXXX")

run_as_web_user() {
  local cmd="$1"
  shift
  if [ -n "$SUDO" ]; then
    $SUDO -u "$WEB_USER" "$cmd" "$@"
  else
    "$cmd" "$@"
  fi
}

verify_checksum() {
  local file="$1"
  local sha256var="WP_${PLUGIN^^}_SHA256"
  local expected="${!sha256var:-}"
  if [ -z "$expected" ]; then
    return 0
  fi
  echo "→ Verifying SHA-256 checksum..."
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "Error: SHA-256 mismatch for $file (expected $expected, got $actual)" >&2
    return 1
  fi
  echo "  checksum OK"
}

# ── mcp-adapter ──────────────────────────────────────────────────────
if [ "$PLUGIN" = "mcp-adapter" ]; then
  echo "=== Installing mcp-adapter $VERSION (official WordPress) ==="

  if [ -d "$PLUGINS_DIR/mcp-adapter" ] && $WP plugin is-active mcp-adapter 2>/dev/null; then
    echo "mcp-adapter is already installed and active."
    exit 0
  fi

  RELEASE_TAG="$VERSION"
  DOWNLOAD_URL="https://github.com/WordPress/mcp-adapter/releases/download/${RELEASE_TAG}/mcp-adapter.zip"

  echo "→ Downloading mcp-adapter ${RELEASE_TAG}..."
  if ! curl -fsSL -o "$TMPDIR/mcp-adapter.zip" "$DOWNLOAD_URL"; then
    echo "Error: failed to download mcp-adapter ${RELEASE_TAG} from ${DOWNLOAD_URL}" >&2
    exit 3
  fi

  verify_checksum "$TMPDIR/mcp-adapter.zip" || exit 6

  echo "→ Extracting..."
  unzip -q "$TMPDIR/mcp-adapter.zip" -d "$TMPDIR/extract"
  SRC=$(find "$TMPDIR/extract" -maxdepth 1 -type d -name 'mcp-adapter*' | head -n 1)
  if [ -z "$SRC" ]; then
    echo "Error: extracted folder not found" >&2
    exit 3
  fi

  $SUDO rm -rf "$PLUGINS_DIR/mcp-adapter"
  $SUDO cp -r "$SRC" "$PLUGINS_DIR/mcp-adapter"
  $SUDO chown -R "${WEB_USER}:${WEB_USER}" "$PLUGINS_DIR/mcp-adapter"

  if [ -f "$PLUGINS_DIR/mcp-adapter/composer.json" ] && [ ! -d "$PLUGINS_DIR/mcp-adapter/vendor" ]; then
    if command -v composer >/dev/null 2>&1; then
      echo "→ Running composer install (no-dev)..."
      run_as_web_user composer install --no-dev --no-interaction --optimize-autoloader \
        --working-dir="$PLUGINS_DIR/mcp-adapter"
      $SUDO chown -R "${WEB_USER}:${WEB_USER}" "$PLUGINS_DIR/mcp-adapter" 2>/dev/null || true
    else
      echo "Warning: composer not found. The plugin may need 'composer install' if vendor/ is missing." >&2
    fi
  fi

  echo "→ Activating plugin..."
  $WP plugin activate mcp-adapter || { echo "Error: activation failed" >&2; exit 4; }
  $WP rewrite flush
  echo "✓ mcp-adapter ${RELEASE_TAG} installed and active."
  echo ""
  echo "Next steps:"
  echo "  1. Create an Application Password:"
  echo "     $WP user application-password create <user_id> \"mcp-clients\""
  echo "  2. Endpoint: ${WP_URL:-https://yourdomain.com}/wp-json/mcp/mcp-adapter-default-server"

# ── ai-engine ──────────────────────────────────────────────────────────
elif [ "$PLUGIN" = "ai-engine" ]; then
  echo "=== Installing AI Engine $VERSION ==="

  if [ -d "$PLUGINS_DIR/ai-engine" ] && $WP plugin is-active ai-engine 2>/dev/null; then
    echo "ai-engine is already installed and active."
    exit 0
  fi

  echo "→ Attempting wp plugin install ai-engine.${VERSION}..."
  if $WP plugin install "ai-engine.${VERSION}" --activate 2>/dev/null; then
    echo "✓ Installed via wp plugin install."
  else
    echo "→ wp plugin install failed, downloading manually..."
    DOWNLOAD_URL="https://downloads.wordpress.org/plugin/ai-engine.${VERSION}.zip"
    if ! curl -fsSL -o "$TMPDIR/ai-engine.zip" "$DOWNLOAD_URL"; then
      echo "Error: failed to download ai-engine.${VERSION}" >&2
      exit 3
    fi

    verify_checksum "$TMPDIR/ai-engine.zip" || exit 6

    unzip -q "$TMPDIR/ai-engine.zip" -d "$TMPDIR/extract"
    $SUDO cp -r "$TMPDIR/extract/ai-engine" "$PLUGINS_DIR/ai-engine"
    $SUDO chown -R "${WEB_USER}:${WEB_USER}" "$PLUGINS_DIR/ai-engine"
    $WP plugin activate ai-engine || { echo "Error: activation failed" >&2; exit 4; }
  fi

  $WP rewrite flush

  # Enable MCP module via a dedicated PHP file instead of inline wp eval
  MU_FILE="$WP_PATH/wp-content/mu-plugins/ai-engine-mcp-enable.php"
  if [ ! -f "$MU_FILE" ]; then
    echo "→ Enabling MCP module via mu-plugin..."
    cat > "$TMPDIR/ai-engine-mcp-enable.php" <<'EOF'
<?php
add_filter('mwai_mcp_enabled', '__return_true');
EOF
    $SUDO mkdir -p "$(dirname "$MU_FILE")"
    $SUDO cp "$TMPDIR/ai-engine-mcp-enable.php" "$MU_FILE"
    $SUDO chown "${WEB_USER}:${WEB_USER}" "$MU_FILE"
    $SUDO chmod 0644 "$MU_FILE"
  fi

  echo "✓ AI Engine ${VERSION} installed. MCP module enabled via mu-plugin."
  echo ""
  echo "Next steps:"
  echo "  1. Generate a secure Bearer Token and set mwai_options['mcp_bearer_token']"
  echo "     (requires explicit approval; do not run wp eval automatically)."
  echo "  2. Endpoint: ${WP_URL:-https://yourdomain.com}/wp-json/mcp/v1/http"
fi
