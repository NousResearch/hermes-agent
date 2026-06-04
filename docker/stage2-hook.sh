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

# Drop to hermes via s6-setuidgid, but skip it when already non-root.
as_hermes() { [ "$(id -u)" = 0 ] || { "$@"; return; }; s6-setuidgid hermes "$@"; }

# --- Reject the unsupported `docker run --user <uid>:<gid>` start ---
# Detect the case where the container was launched with `--user` pinned to an
# arbitrary host UID (the classic `--user $(id -u):$(id -g)` invocation people
# used in the tini era to make container-written files match their host user).
#
# Under s6-overlay this no longer works: the bootstrap (UID remap, volume +
# build-tree chown, config seeding) all require root, and they're skipped when
# the container starts non-root. The baked image trees (/opt/data, /opt/hermes/
# .venv, ui-tui, node_modules) stay owned by the hermes build UID (10000), so an
# arbitrary `--user` UID can't write them — the runtime then fails with EACCES
# on a bind mount, or hard-crashes on a named volume (Docker initialises the
# volume from the image as UID 10000, and the non-root start can't even `cd`
# into $HERMES_HOME). See #34837 for the supervision-tree side of this.
#
# The supported way to match host-side ownership is to start as root (the image
# default) and pass HERMES_UID/HERMES_GID — or the PUID/PGID aliases — which the
# remap block below consumes via usermod/groupmod + targeted chown. That gives
# the exact same outcome (files owned by your host UID) without breaking s6.
#
# preinit runs setuid-root (euid=0) but cont-init.d hooks run with the real UID
# the container was started as, so `id -u` here is the host UID (e.g. 1000), and
# `id -u hermes` is the unremapped build UID (10000) because no root-only remap
# could run. root starts (id -u = 0) and the normal supervised drop to the
# hermes UID are both unaffected.
cur_uid="$(id -u)"
if [ "$cur_uid" != 0 ] && [ "$cur_uid" != "$(id -u hermes)" ]; then
    cat >&2 <<EOF
[stage2] ERROR: container started with --user $cur_uid (an arbitrary, non-hermes UID).

This is not supported under the s6-overlay image. The container bootstrap
(UID remap, volume ownership, dependency installs) needs to start as root,
and the baked image directories are owned by the hermes user (UID $(id -u hermes)),
so a pinned --user UID cannot write them — startup will fail.

To make container-written files match your HOST user, DON'T use --user.
Start the container as root (the default) and pass your host UID/GID instead:

    docker run -e HERMES_UID=\$(id -u) -e HERMES_GID=\$(id -g) ...

NAS users (Synology / unRAID / UGOS) can use the PUID/PGID aliases:

    docker run -e PUID=\$(id -u) -e PGID=\$(id -g) ...

The image remaps the hermes user to that UID/GID at boot and chowns the data
volume accordingly, so files land owned by your host user — the same outcome
--user was being used for, without breaking the supervision tree.
EOF
    exit 1
fi

# --- Bootstrap HERMES_HOME as root ---
# Create the directory (and any missing parents) while we still have root
# privileges so the chown checks below see real metadata and the later
# `s6-setuidgid hermes mkdir -p` block doesn't EACCES on root-owned
# ancestors. Without this, custom HERMES_HOME paths whose parents only
# root can create (e.g. `HERMES_HOME=/home/hermes/.hermes` in a Compose
# file, or any path under a fresh / not pre-populated by the image)
# fail on first boot with `mkdir: cannot create directory '/...': Permission
# denied` and the cont-init hook exits non-zero. Idempotent — `mkdir -p`
# is a no-op if the dir already exists. (#18482, salvages #18488)
mkdir -p "$HERMES_HOME"

# Numeric UID/GID validation: must be digits only, 1000-65534
validate_uid_gid() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -ge 1000 ] && [ "$1" -le 65534 ] ;;
    esac
}

# --- UID/GID remap ---
# Accept PUID/PGID as aliases for HERMES_UID/HERMES_GID.  NAS users (UGOS,
# Synology, unRAID) expect the LinuxServer.io PUID/PGID convention and
# bind-mount /opt/data from a host directory owned by their own UID; without
# this alias those vars are silently ignored and the s6-setuidgid drop to
# UID 10000 leaves the runtime unable to read the volume.  HERMES_UID/
# HERMES_GID still win when both are set.  See #15290, salvages #25872.
HERMES_UID="${HERMES_UID:-${PUID:-}}"
HERMES_GID="${HERMES_GID:-${PGID:-}}"

if [ -n "${HERMES_UID:-}" ] && validate_uid_gid "$HERMES_UID" && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
    echo "[stage2] Changing hermes UID to $HERMES_UID"
    usermod -u "$HERMES_UID" hermes
fi
if [ -n "${HERMES_GID:-}" ] && validate_uid_gid "$HERMES_GID" && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
    echo "[stage2] Changing hermes GID to $HERMES_GID"
    # -o allows non-unique GID (e.g. macOS GID 20 "staff" may already
    # exist as "dialout" in the Debian-based container image).
    groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
fi

# --- Docker socket group membership (docker-in-docker / DooD) ---
# When the user bind-mounts the host Docker daemon socket
# (`-v /var/run/docker.sock:/var/run/docker.sock`) to use the `docker`
# terminal backend from inside the container, the socket is owned by the
# host's `docker` group (or root). The supervised hermes user (UID 10000)
# is not a member of any group that matches the socket's GID, so every
# `docker` invocation EACCES'es and `check_terminal_requirements()` fails.
# See #16703.
#
# Granting the supp group via `docker run --group-add <gid>` alone is
# NOT sufficient with our s6-setuidgid privilege drop: s6-setuidgid (and
# gosu, the older shim) calls initgroups() for the target user, which
# rebuilds the supplementary group list from /etc/group. Without an
# /etc/group entry whose GID matches the socket, the kernel-granted
# supp group is silently wiped between PID 1 and the dropped process.
# Confirmed empirically: `--group-add 998` alone leaves the dropped
# hermes process with `Groups: 10000` (998 gone); after this hook adds
# the entry, the dropped process has `Groups: 998 10000` as expected.
#
# Fix: detect the socket's GID at boot and ensure /etc/group has a
# matching entry that includes hermes. Idempotent across container
# restarts. Skipped silently when no socket is bind-mounted.
#
# Handles the awkward corner cases:
#   - socket owned by GID 0 (root) — some Podman setups; usermod -aG root
#   - socket GID already used by a known container group (e.g. tty=5):
#     reuse that group's name rather than creating a duplicate
#   - hermes is already a member of the right group (idempotent restart)
#   - chown/groupadd failures under rootless containers — non-fatal
for sock in /var/run/docker.sock /run/docker.sock; do
    [ -S "$sock" ] || continue
    sock_gid=$(stat -c '%g' "$sock" 2>/dev/null) || continue
    [ -n "$sock_gid" ] || continue
    # Already a member? Nothing to do.
    if id -G hermes 2>/dev/null | tr ' ' '\n' | grep -qx "$sock_gid"; then
        echo "[stage2] hermes already in group $sock_gid for $sock"
        break
    fi
    # Resolve or create a group name for this GID.
    sock_group=$(getent group "$sock_gid" 2>/dev/null | cut -d: -f1)
    if [ -z "$sock_group" ]; then
        sock_group="hostdocker"
        if ! groupadd -g "$sock_gid" "$sock_group" 2>/dev/null; then
            echo "[stage2] Warning: groupadd -g $sock_gid $sock_group failed; skipping docker socket group setup"
            break
        fi
        echo "[stage2] Created group $sock_group (GID $sock_gid) for Docker socket"
    fi
    if usermod -aG "$sock_group" hermes 2>/dev/null; then
        echo "[stage2] Added hermes to group $sock_group (GID $sock_gid) for $sock"
    else
        echo "[stage2] Warning: usermod -aG $sock_group hermes failed; docker backend may fail with EACCES"
    fi
    break
done

# --- Fix ownership of data volume ---
# When HERMES_UID is remapped or the top-level $HERMES_HOME isn't owned by
# the runtime hermes UID, restore ownership to hermes — but ONLY for the
# directories hermes actually writes to. The full $HERMES_HOME may be a
# host-mounted bind containing unrelated user files; `chown -R` would
# silently destroy host ownership of those (see issue #19788).
#
# The canonical list of hermes-owned subdirs is the same one the s6-setuidgid
# mkdir -p block below seeds. Keep them in sync if the seed list changes.
actual_hermes_uid=$(id -u hermes)
needs_chown=false
if [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
    needs_chown=true
fi
if [ "$needs_chown" = true ]; then
    echo "[stage2] Fixing ownership of $HERMES_HOME (targeted) to hermes ($actual_hermes_uid)"
    # In rootless Podman the container's "root" is mapped to an
    # unprivileged host UID — chown will fail. That's fine: the volume
    # is already owned by the mapped user on the host side.
    #
    # Top-level $HERMES_HOME: chown the directory itself (not its contents)
    # so hermes can mkdir new subdirs but bind-mounted host files keep
    # their existing ownership.
    chown hermes:hermes "$HERMES_HOME" 2>/dev/null || \
        echo "[stage2] Warning: chown $HERMES_HOME failed (rootless container?) — continuing"
    # Hermes-owned subdirs: recursive chown is safe here because these are
    # created and managed exclusively by hermes (see the s6-setuidgid mkdir
    # -p block below for the canonical list).
    for sub in cron sessions logs hooks memories skills skins plans workspace home profiles; do
        if [ -e "$HERMES_HOME/$sub" ]; then
            chown -R hermes:hermes "$HERMES_HOME/$sub" 2>/dev/null || \
                echo "[stage2] Warning: chown $HERMES_HOME/$sub failed (rootless container?) — continuing"
        fi
    done
fi

# --- Fix ownership of build trees under $INSTALL_DIR ---
# Hermes-owned trees under $INSTALL_DIR must be re-chowned whenever the
# runtime hermes UID no longer owns them — otherwise:
#   - .venv: lazy_deps.py cannot install platform packages (discord.py,
#     telegram, slack, etc.) with EACCES (#15012, #21100)
#   - ui-tui: esbuild rebuilds dist/entry.js on every TUI launch (when
#     the source mtime is newer than dist/ or when HERMES_TUI_FORCE_BUILD
#     is set) and writes to ui-tui/dist/. Without this chown the new
#     hermes UID can't write the build output (#28851).
#   - node_modules: root-level dependencies (puppeteer, web tooling)
#     that runtime code may walk/update.
# The set mirrors the build-time `chown -R hermes:hermes` line in the
# Dockerfile — keep them in sync if the Dockerfile chown set changes.
# These are under $INSTALL_DIR (not $HERMES_HOME), so the bind-mount
# concern doesn't apply — recursive is fine.
#
# This MUST be gated independently of the $HERMES_HOME ownership check
# above. `usermod -u <new> hermes` re-chowns the hermes home dir
# ($HERMES_HOME == /opt/data) to the new UID as a side effect, so after a
# HERMES_UID/PUID remap `stat $HERMES_HOME` always already matches the new
# UID and `needs_chown` is false — but the build trees under /opt/hermes
# are NOT touched by usermod and remain owned by the build-time UID
# (10000). Gating them on $HERMES_HOME ownership (as #35027 did) silently
# skipped this chown on the common PUID/NAS path, regressing lazy installs
# and TUI rebuilds. Probe the build trees directly instead: chown only
# when the venv is not already owned by the runtime hermes UID. Idempotent
# and skips the expensive recursive chown on every restart once ownership
# is settled.
venv_owner=$(stat -c %u "$INSTALL_DIR/.venv" 2>/dev/null || echo "")
if [ -n "$venv_owner" ] && [ "$venv_owner" != "$actual_hermes_uid" ]; then
    echo "[stage2] Fixing ownership of build trees under $INSTALL_DIR to hermes ($actual_hermes_uid)"
    chown -R hermes:hermes \
        "$INSTALL_DIR/.venv" \
        "$INSTALL_DIR/ui-tui" \
        "$INSTALL_DIR/node_modules" \
        2>/dev/null || \
        echo "[stage2] Warning: chown of build trees failed (rootless container?) — continuing"
fi

# Always reset ownership of $HERMES_HOME/profiles to hermes on every
# boot. Profile dirs and files can land owned by root when commands
# are invoked via `docker exec <container> hermes …` (which defaults
# to root unless `-u` is passed), and that breaks the cont-init
# reconciler (02-reconcile-profiles) which runs as hermes and walks
# the profiles dir. Idempotent; skipped on rootless containers where
# chown would fail.
if [ -d "$HERMES_HOME/profiles" ]; then
    chown -R hermes:hermes "$HERMES_HOME/profiles" 2>/dev/null || true
fi

# Reset ownership of hermes-owned top-level state files on every boot.
# The targeted data-volume chown above only covers hermes-owned
# *subdirectories*; loose state files living directly under $HERMES_HOME
# are missed. When those files are created or rewritten by
# `docker exec <container> hermes …` (root unless `-u` is passed) they
# land root-owned, and the unprivileged hermes runtime then hits
# PermissionError on next startup (e.g. gateway.lock / state.db /
# auth.json), producing a gateway restart loop.
#
# We use an explicit allowlist rather than a blanket `find -user root`
# sweep so host-owned files in a bind-mounted $HERMES_HOME are never
# touched — same targeted-ownership contract as the subdir chown above
# (issue #19788, PR #19795). The list mirrors the top-level *file*
# entries of hermes_cli.profile_distribution.USER_OWNED_EXCLUDE plus the
# runtime lock files; keep them in sync if that set changes.
for f in \
    auth.json auth.lock .env \
    state.db state.db-shm state.db-wal \
    hermes_state.db \
    response_store.db response_store.db-shm response_store.db-wal \
    gateway.pid gateway.lock gateway_state.json processes.json \
    active_profile; do
    if [ -e "$HERMES_HOME/$f" ]; then
        chown hermes:hermes "$HERMES_HOME/$f" 2>/dev/null || true
    fi
done

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
#
# Use direct `mkdir -p` invocation (no `sh -c "..."` wrapper) so the
# shell isn't a second interpreter — defends against $HERMES_HOME values
# containing shell metacharacters. PR #30136 review item O2.
as_hermes mkdir -p \
    "$HERMES_HOME/cron" \
    "$HERMES_HOME/sessions" \
    "$HERMES_HOME/logs" \
    "$HERMES_HOME/hooks" \
    "$HERMES_HOME/memories" \
    "$HERMES_HOME/skills" \
    "$HERMES_HOME/skins" \
    "$HERMES_HOME/plans" \
    "$HERMES_HOME/workspace" \
    "$HERMES_HOME/home"

# --- Install-method stamp (read by detect_install_method() in hermes status) ---
# Preserved from the tini-era entrypoint (PR #27843). Must be written as
# the hermes user so ownership matches the file's documented owner.
# tee is invoked directly via s6-setuidgid (no `sh -c` wrapper) for the
# same shell-metacharacter safety described above.
printf 'docker\n' | as_hermes tee "$HERMES_HOME/.install_method" >/dev/null \
    || true

# --- Seed config files (only on first boot) ---
seed_one() {
    dest=$1
    src=$2
    if [ ! -f "$HERMES_HOME/$dest" ] && [ -f "$INSTALL_DIR/$src" ]; then
        as_hermes cp "$INSTALL_DIR/$src" "$HERMES_HOME/$dest"
    fi
}
seed_one ".env" ".env.example"
seed_one "config.yaml" "cli-config.yaml.example"
seed_one "SOUL.md" "docker/SOUL.md"

# .env holds API keys and secrets — restrict to owner-only access. Applied
# unconditionally (not only on first-seed) so a host-mounted .env that was
# created with a permissive umask gets tightened on every container start.
if [ -f "$HERMES_HOME/.env" ]; then
    chown hermes:hermes "$HERMES_HOME/.env" 2>/dev/null || true
    chmod 600 "$HERMES_HOME/.env" 2>/dev/null || true
fi

# --- Migrate persisted config schema ---
# Docker image upgrades replace the code under $INSTALL_DIR but preserve
# $HERMES_HOME on the mounted volume. Run the same safe, non-interactive
# config-schema migrations that `hermes update` runs for non-Docker installs,
# after first-boot seeding and before supervised gateway services start.
# Set HERMES_SKIP_CONFIG_MIGRATION=1 for controlled/manual migrations.
if [ -f "$HERMES_HOME/config.yaml" ]; then
    s6-setuidgid hermes "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/scripts/docker_config_migrate.py" \
        || echo "[stage2] Warning: docker_config_migrate.py failed; continuing"
fi

# auth.json: bootstrap from env on first boot only. Same semantics as the
# pre-s6 entrypoint — the [ ! -f ] guard is critical to avoid clobbering
# rotated refresh tokens on container restart.
if [ ! -f "$HERMES_HOME/auth.json" ] && [ -n "${HERMES_AUTH_JSON_BOOTSTRAP:-}" ]; then
    printf '%s' "$HERMES_AUTH_JSON_BOOTSTRAP" > "$HERMES_HOME/auth.json"
    chown hermes:hermes "$HERMES_HOME/auth.json" 2>/dev/null || true
    chmod 600 "$HERMES_HOME/auth.json"
fi

# gateway_state.json: declare the gateway's INITIAL supervised state on a
# fresh volume. Same first-boot-only env-seed pattern as auth.json above.
#
# On a blank volume there is no gateway_state.json, so the boot reconciler
# (cont-init.d/02-reconcile-profiles → container_boot.reconcile_profile_gateways)
# registers the gateway-default s6 slot but leaves it DOWN — it only
# auto-starts when the last recorded state was "running". That means a
# freshly-provisioned container comes up with the gateway down until
# someone starts it (e.g. from the dashboard). An orchestrator that
# provisions a fresh volume and wants the gateway running from first boot
# can set HERMES_GATEWAY_BOOTSTRAP_STATE=running; we seed the state file
# here, BEFORE 02-reconcile-profiles runs (cont-init.d scripts run in
# lexicographic order), so the reconciler sees prior_state=running and
# brings the supervised slot up on the very first boot.
#
# This is a generic container contract, not specific to any host: it seeds
# the SAME gateway_state.json the reconciler already consults, exactly as
# HERMES_AUTH_JSON_BOOTSTRAP seeds auth.json. The [ ! -f ] guard is the
# load-bearing part — on every subsequent boot the persisted state wins,
# so a gateway the operator deliberately stopped stays stopped across
# restarts and we never clobber real runtime state.
#
# Only a literal "running" is honoured (the sole value in the reconciler's
# _AUTOSTART_STATES); any other value is ignored so a typo can't write a
# bogus state the reconciler would treat as "no prior state" anyway.
if [ ! -f "$HERMES_HOME/gateway_state.json" ] && \
        [ "${HERMES_GATEWAY_BOOTSTRAP_STATE:-}" = "running" ]; then
    printf '{"gateway_state":"running"}\n' > "$HERMES_HOME/gateway_state.json"
    chown hermes:hermes "$HERMES_HOME/gateway_state.json" 2>/dev/null || true
    chmod 644 "$HERMES_HOME/gateway_state.json"
fi

# --- Sync bundled skills ---
# Invoke the venv's python by absolute path so we don't need a `sh -c`
# wrapper to source the activate script. This is safe because
# skills_sync.py doesn't depend on any environment exports beyond what
# the python binary's own bin-stub already sets up (sys.path is rooted
# at the venv's site-packages by virtue of running .venv/bin/python).
if [ -d "$INSTALL_DIR/skills" ]; then
    as_hermes "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/tools/skills_sync.py" \
        || echo "[stage2] Warning: skills_sync.py failed; continuing"
fi

# --- Discover agent-browser's Chromium binary ---
# The image's Dockerfile runs `npx playwright install chromium`, which
# populates ``$PLAYWRIGHT_BROWSERS_PATH`` (=/opt/hermes/.playwright) with
# a ``chromium_headless_shell-<build>/chrome-headless-shell-linux64/``
# directory. agent-browser (the runtime CLI Hermes spawns for the
# browser tool) doesn't recognise this layout in its own cache scan and
# fails with "Auto-launch failed: Chrome not found" — even though the
# binary is right there (#15697).
#
# Fix: locate the binary at boot and export ``AGENT_BROWSER_EXECUTABLE_PATH``
# via /run/s6/container_environment so the `with-contenv` shebang on
# main-wrapper.sh propagates it into the supervised ``hermes`` process
# and thence to agent-browser subprocesses.
#
# - Skipped when the user has already set ``AGENT_BROWSER_EXECUTABLE_PATH``
#   (lets users override with a system Chrome install).
# - Filename-matched (not path-matched): the chromium dir contains many
#   shared libraries (libGLESv2.so, libEGL.so, ...) which inherit the
#   executable bit from Playwright's tarball but are NOT browser binaries.
#   We only accept files whose basename is chrome / chromium /
#   chrome-headless-shell / headless_shell / chromium-browser. Compare
#   PR #18635's earlier ``find | grep -Ei 'chrome|chromium'`` which would
#   match the path ``.../chrome-headless-shell-linux64/libGLESv2.so`` and
#   pick a .so.
# - Quietly skipped when $PLAYWRIGHT_BROWSERS_PATH doesn't exist (e.g.
#   custom builds that strip Playwright).
if [ -z "${AGENT_BROWSER_EXECUTABLE_PATH:-}" ] && \
        [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && \
        [ -d "$PLAYWRIGHT_BROWSERS_PATH" ]; then
    browser_bin=$(find "$PLAYWRIGHT_BROWSERS_PATH" -type f -executable \
        \( -name 'chrome' -o -name 'chromium' \
           -o -name 'chrome-headless-shell' -o -name 'headless_shell' \
           -o -name 'chromium-browser' \) \
        2>/dev/null | head -n 1)
    if [ -n "$browser_bin" ]; then
        echo "[stage2] Found agent-browser Chromium binary: $browser_bin"
        # Write to s6's container_environment so with-contenv picks it
        # up for all supervised services (main-hermes, dashboard, etc.).
        # Idempotent: each boot overwrites with the current path.
        # Some container runtimes / s6-overlay versions do not create the
        # envdir before cont-init hooks run, so create it defensively.
        mkdir -p /run/s6/container_environment
        printf '%s' "$browser_bin" > /run/s6/container_environment/AGENT_BROWSER_EXECUTABLE_PATH
    else
        echo "[stage2] Warning: no Chromium binary under $PLAYWRIGHT_BROWSERS_PATH; browser tool may fail"
    fi
fi

echo "[stage2] Setup complete; starting user services"
