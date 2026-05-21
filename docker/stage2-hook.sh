#!/bin/sh
# s6-overlay stage2 hook — runs as root after the supervision tree is
# up but before user services start. Handles UID/GID remap, volume
# chown, config seeding, and skills sync.
#
# Per-service privilege drop happens inside each service's `run` script
# (and in main-wrapper.sh) via s6-setuidgid, not here.
#
# Wired into the image as /etc/cont-init.d/01-hermes-setup by the
# Dockerfile. The shim at docker/entrypoint.sh forwards to this script
# so external references to docker/entrypoint.sh still work.
#
# NB: cont-init.d scripts run with no arguments — the user's CMD args
# are NOT visible here. That's fine: we use Architecture B (s6-overlay
# main-program model), so main-wrapper.sh runs the CMD with full
# stdin/stdout/stderr access and handles arg parsing there.

set -eu

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

# --- UID/GID remap ---
if [ -n "${HERMES_UID:-}" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
    echo "[stage2] Changing hermes UID to $HERMES_UID"
    usermod -u "$HERMES_UID" hermes
fi
if [ -n "${HERMES_GID:-}" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
    echo "[stage2] Changing hermes GID to $HERMES_GID"
    # -o allows non-unique GID (e.g. macOS GID 20 "staff" may already
    # exist as "dialout" in the Debian-based container image).
    groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
fi

# --- Fix ownership of data volume ---
actual_hermes_uid=$(id -u hermes)
needs_chown=false
if [ -n "${HERMES_UID:-}" ] && [ "$HERMES_UID" != "10000" ]; then
    needs_chown=true
elif [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
    needs_chown=true
fi
if [ "$needs_chown" = true ]; then
    echo "[stage2] Fixing ownership of $HERMES_HOME to hermes ($actual_hermes_uid)"
    # In rootless Podman the container's "root" is mapped to an
    # unprivileged host UID — chown will fail. That's fine: the volume
    # is already owned by the mapped user on the host side.
    chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || \
        echo "[stage2] Warning: chown failed (rootless container?) — continuing"
    # The .venv must also be re-chowned when UID is remapped, otherwise
    # lazy_deps.py cannot install platform packages (discord.py, etc.).
    chown -R hermes:hermes "$INSTALL_DIR/.venv" 2>/dev/null || \
        echo "[stage2] Warning: chown .venv failed (rootless container?) — continuing"
fi

# --- config.yaml permissions ---
# Ensure config.yaml is readable by the hermes runtime user even if it
# was edited on the host after initial ownership setup.
if [ -f "$HERMES_HOME/config.yaml" ]; then
    chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
    chmod 640 "$HERMES_HOME/config.yaml" 2>/dev/null || true
fi

# --- Seed directory structure as hermes user ---
# Run as hermes via s6-setuidgid so dirs end up owned correctly (matters
# under rootless Podman where chown back to root would fail).
s6-setuidgid hermes sh -c "mkdir -p \"$HERMES_HOME\"/cron \
    \"$HERMES_HOME\"/sessions \"$HERMES_HOME\"/logs \"$HERMES_HOME\"/hooks \
    \"$HERMES_HOME\"/memories \"$HERMES_HOME\"/skills \"$HERMES_HOME\"/skins \
    \"$HERMES_HOME\"/plans \"$HERMES_HOME\"/workspace \"$HERMES_HOME\"/home"

# --- Install-method stamp (read by detect_install_method() in hermes status) ---
# Preserved from the tini-era entrypoint (PR #27843). Must be written as
# the hermes user so ownership matches the file's documented owner.
s6-setuidgid hermes sh -c "echo docker > \"$HERMES_HOME/.install_method\"" 2>/dev/null || true

# --- Seed config files (only on first boot) ---
seed_one() {
    dest=$1
    src=$2
    if [ ! -f "$HERMES_HOME/$dest" ] && [ -f "$INSTALL_DIR/$src" ]; then
        s6-setuidgid hermes cp "$INSTALL_DIR/$src" "$HERMES_HOME/$dest"
    fi
}
seed_one ".env" ".env.example"
seed_one "config.yaml" "cli-config.yaml.example"
seed_one "SOUL.md" "docker/SOUL.md"

# auth.json: bootstrap from env on first boot only. Same semantics as the
# pre-s6 entrypoint — the [ ! -f ] guard is critical to avoid clobbering
# rotated refresh tokens on container restart.
if [ ! -f "$HERMES_HOME/auth.json" ] && [ -n "${HERMES_AUTH_JSON_BOOTSTRAP:-}" ]; then
    printf '%s' "$HERMES_AUTH_JSON_BOOTSTRAP" > "$HERMES_HOME/auth.json"
    chown hermes:hermes "$HERMES_HOME/auth.json" 2>/dev/null || true
    chmod 600 "$HERMES_HOME/auth.json"
fi

# --- Sync bundled skills ---
if [ -d "$INSTALL_DIR/skills" ]; then
    s6-setuidgid hermes sh -c \
        ". $INSTALL_DIR/.venv/bin/activate && python3 $INSTALL_DIR/tools/skills_sync.py" \
        || echo "[stage2] Warning: skills_sync.py failed; continuing"
fi

echo "[stage2] Setup complete; starting user services"
