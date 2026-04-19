#!/bin/bash
# Docker/Podman entrypoint: bootstrap config files into the mounted volume, then run hermes.
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

# --- Privilege dropping via gosu ---
# When started as root (the default for Docker, or fakeroot in rootless Podman),
# optionally remap the hermes user/group to match host-side ownership, fix volume
# permissions, then re-exec as hermes.
if [ "$(id -u)" = "0" ]; then
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
        echo "Changing hermes UID to $HERMES_UID"
        usermod -u "$HERMES_UID" hermes
    fi

    if [ -n "$HERMES_GID" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
        echo "Changing hermes GID to $HERMES_GID"
        # -o allows non-unique GID (e.g. macOS GID 20 "staff" may already exist
        # as "dialout" in the Debian-based container image)
        groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
    fi

    actual_hermes_uid=$(id -u hermes)
    if [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
        echo "$HERMES_HOME is not owned by $actual_hermes_uid, fixing"
        # In rootless Podman the container's "root" is mapped to an unprivileged
        # host UID — chown will fail.  That's fine: the volume is already owned
        # by the mapped user on the host side.
        chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || \
            echo "Warning: chown failed (rootless container?) — continuing anyway"
    fi

    echo "Dropping root privileges"
    exec gosu hermes "$0" "$@"
fi

# --- Running as hermes from here ---
echo "Running as $(id -u):$(id -g) ($(id -un):$(id -gn))"
# if SYNC_CODE_DIR is set to an existing target path that we can write to, copy the code in /opt/hermes as set in INSTALL_DIR (only the code) so that external tools can use hermes-agent and install it in their own venv.  
# This is useful for development with bind mounts, but also for production if someone wants to use the image as a base for their own image and install hermes-agent into their own venv instead of using the bundled one.
# check SYNC_CODE_DIR exist
if [ ! -z "${SYNC_CODE_DIR+x}" ]; then
    if [ -d "$SYNC_CODE_DIR" ] && [ -w "$SYNC_CODE_DIR" ]; then
        it="${SYNC_CODE_DIR}/.testfile"; rm -f "$it"; touch "$it" 
        if [ -f $it ]; then
            rm -f "$it" || echo "Test file created successfully in SYNC_CODE_DIR directory, but deletion failed, it is likely not fully writable by the hermes user. We will attempt the code copy anyway."
            echo "SYNC_CODE_DIR is set to $SYNC_CODE_DIR, copying code there"
            rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' --exclude='.git' --exclude ".playwright" ${INSTALL_DIR}/ "$SYNC_CODE_DIR"/ --delete || echo "Failed to copy all files from ${INSTALL_DIR} to ${SYNC_CODE_DIR} directory, will continue anyway."
            # The delete option ensures that if files are removed from the source (INSTALL_DIR), they are also removed from the target (SYNC_CODE_DIR) on subsequent runs, keeping them in sync.
            echo "Code copied to SYNC_CODE_DIR successfully (this oes not invalidted any previous message, but the attempt was made), you can use the SYNC_CODE_DIR directory to access the code and install hermes-agent in your own venv if needed."
        else
            echo "Failed to write a test file to SYNC_CODE_DIR directory as the hermes user, we will skip code copy to avoid potential issues with file permissions. Please ensure that the SYNC_CODE_DIR directory is writable by the hermes user."
        fi
    else
        echo "SYNC_CODE_DIR is not set to a writable directory, skipping code copy"
    fi
fi

source "${INSTALL_DIR}/.venv/bin/activate"

# Create essential directory structure.  Cache and platform directories
# (cache/images, cache/audio, platforms/whatsapp, etc.) are created on
# demand by the application — don't pre-create them here so new installs
# get the consolidated layout from get_hermes_dir().
# The "home/" subdirectory is a per-profile HOME for subprocesses (git,
# ssh, gh, npm …).  Without it those tools write to /root which is
# ephemeral and shared across profiles.  See issue #4426.
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# .env
if [ ! -f "$HERMES_HOME/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

# config.yaml
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
fi

# SOUL.md
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"
fi

# Sync bundled skills (manifest-based so user edits are preserved)
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

exec hermes "$@"
