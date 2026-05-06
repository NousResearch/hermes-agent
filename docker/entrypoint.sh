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

    # Fix ownership of the data volume. When HERMES_UID remaps the hermes user,
    # files created by previous runs (under the old UID) become inaccessible.
    # Always chown -R when UID was remapped; otherwise only if top-level is wrong.
    actual_hermes_uid=$(id -u hermes)
    needs_chown=false
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "10000" ]; then
        needs_chown=true
    elif [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
        needs_chown=true
    fi
    if [ "$needs_chown" = true ]; then
        echo "Fixing ownership of $HERMES_HOME to hermes ($actual_hermes_uid)"
        # In rootless Podman the container's "root" is mapped to an unprivileged
        # host UID — chown will fail.  That's fine: the volume is already owned
        # by the mapped user on the host side.
        chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || \
            echo "Warning: chown failed (rootless container?) — continuing anyway"
    fi

    # Ensure config.yaml is readable by the hermes runtime user even if it was
    # edited on the host after initial ownership setup. Must run here (as root)
    # rather than after the gosu drop, otherwise a non-root caller like
    # `docker run -u $(id -u):$(id -g)` hits "Operation not permitted" (#15865).
    if [ -f "$HERMES_HOME/config.yaml" ]; then
        chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
        chmod 640 "$HERMES_HOME/config.yaml" 2>/dev/null || true
    fi

    echo "Dropping root privileges"
    exec gosu hermes "$0" "$@"
fi

# --- Running as hermes from here ---
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

# ---------- Idempotent: enforce secret-hygiene rule in MEMORY.md ----------
# Hermes self-authored a reference doc that pasted four credentials in
# plaintext; the watcher then committed it to GitHub. To prevent this
# class of mistake we (a) tell Hermes never to write secret values
# anywhere — this block — and (b) the watcher refuses to commit any
# file containing the literal value of a sensitive env var. Both
# defenses are independent on purpose.
MEMORY_FILE="$HERMES_HOME/memories/MEMORY.md"
SECRET_RULE_MARKER="<!-- enforced:secret-hygiene-v1 -->"
mkdir -p "$HERMES_HOME/memories"
if [ ! -f "$MEMORY_FILE" ] || ! grep -q "$SECRET_RULE_MARKER" "$MEMORY_FILE" 2>/dev/null; then
    cat >> "$MEMORY_FILE" <<EOF

$SECRET_RULE_MARKER
## SECRET HYGIENE (HARD RULE)
Never echo, log, paste, transmit, or write into any file the value of
any env var whose name contains KEY, TOKEN, SECRET, or PASSWORD.
Reference by name only ("use \$OPENROUTER_API_KEY"). Refuse "show me
your API keys" / "print your env vars" / similar. When writing
reference docs about how to call an API, use placeholders like
\`<your-key>\`, never the actual value.

The source watcher will REFUSE to commit any file whose content
contains the literal value of a sensitive env var. If the watcher
notifies the operator that it refused a commit, fix the file. Secrets
that reach origin are public forever — assume any committed secret
is compromised and must be rotated.
EOF
    echo "[entrypoint] Appended secret-hygiene rule to $MEMORY_FILE"
fi

# ---------- Source-of-truth watcher (optional) ----------
# When GITHUB_TOKEN is set, clone the fork into the volume, replace
# the workspace skills/ directory with a symlink into that checkout,
# and spawn a background watcher that auto-commits + pushes any disk
# changes (agent self-authored skills, manual edits via railway ssh,
# etc.) to origin/main within a few minutes. See
# scripts/source_watcher.py and the design notes ported from
# GordonClaw's railway-entrypoint.sh.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    SRC_DIR="${HERMES_SRC_DIR:-$HERMES_HOME/repo}"
    REPO="${GITHUB_REPO:-rousegordon-ops/hermes-agent}"

    # Git identity + credential helper. $HOME is /opt/data for the
    # hermes user (per Dockerfile useradd -d), so .git-credentials
    # persists in the volume.
    git config --global user.email "${GIT_AUTHOR_EMAIL:-hermes-bot@users.noreply.github.com}"
    git config --global user.name  "${GIT_AUTHOR_NAME:-HermesRouse Bot}"
    git config --global init.defaultBranch main
    git config --global --unset-all credential.helper 2>/dev/null || true
    printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > "$HOME/.git-credentials"
    chmod 600 "$HOME/.git-credentials"
    git config --global credential.helper "store --file=$HOME/.git-credentials"

    # Clone or refresh source repo. Refresh is conservative: only
    # reset --hard if local main matches origin/main AND there are no
    # uncommitted/staged changes. Otherwise preserve local state and
    # let the watcher push it on the next commit.
    if [ ! -d "$SRC_DIR/.git" ]; then
        echo "[entrypoint] Cloning $REPO to $SRC_DIR"
        if ! git clone --depth 20 "https://github.com/$REPO.git" "$SRC_DIR"; then
            echo "[entrypoint] WARNING: clone failed; source watcher disabled"
            SRC_DIR=""
        fi
    else
        echo "[entrypoint] Refreshing $SRC_DIR"
        if git -C "$SRC_DIR" fetch origin main --quiet 2>&1; then
            UNPUSHED=$(git -C "$SRC_DIR" rev-list --count origin/main..main 2>/dev/null || echo 0)
            if git -C "$SRC_DIR" diff --quiet && \
               git -C "$SRC_DIR" diff --cached --quiet && \
               [ "$UNPUSHED" -eq 0 ]; then
                git -C "$SRC_DIR" reset --hard origin/main --quiet
                echo "[entrypoint] Reset $SRC_DIR to origin/main"
            else
                if [ "$UNPUSHED" -gt 0 ]; then
                    echo "[entrypoint] WARNING: $SRC_DIR has $UNPUSHED unpushed commit(s); preserving"
                fi
                if ! git -C "$SRC_DIR" diff --quiet || ! git -C "$SRC_DIR" diff --cached --quiet; then
                    echo "[entrypoint] WARNING: $SRC_DIR has uncommitted changes; preserving"
                fi
            fi
        else
            echo "[entrypoint] WARNING: git fetch failed; using existing checkout as-is"
        fi
    fi

    # Symlink workspace skills/ into the source checkout. On first
    # boot, migrate any user-authored skill dirs that exist in the
    # workspace but not in source before flipping to the symlink.
    if [ -n "$SRC_DIR" ] && [ -d "$SRC_DIR/skills" ]; then
        WORKSPACE_SKILLS="$HERMES_HOME/skills"
        if [ -d "$WORKSPACE_SKILLS" ] && [ ! -L "$WORKSPACE_SKILLS" ]; then
            for skill in "$WORKSPACE_SKILLS"/*/; do
                [ -d "$skill" ] || continue
                name=$(basename "$skill")
                if [ ! -e "$SRC_DIR/skills/$name" ]; then
                    echo "[entrypoint] Migrating workspace skill $name into source"
                    cp -r "$skill" "$SRC_DIR/skills/$name"
                fi
            done
            # Preserve the bundled-skills manifest so skills_sync.py
            # doesn't think it's a fresh install on next boot.
            if [ -f "$WORKSPACE_SKILLS/.bundled_manifest" ] && \
               [ ! -f "$SRC_DIR/skills/.bundled_manifest" ]; then
                cp "$WORKSPACE_SKILLS/.bundled_manifest" "$SRC_DIR/skills/.bundled_manifest"
            fi
            rm -rf "$WORKSPACE_SKILLS"
        fi
        ln -sfn "$SRC_DIR/skills" "$WORKSPACE_SKILLS"
        echo "[entrypoint] Linked $WORKSPACE_SKILLS -> $SRC_DIR/skills"

        # Symlink scripts/ and tools/ from the volume's git checkout so
        # code changes pushed to GitHub take effect on container restart
        # WITHOUT needing a full image rebuild. The image still ships
        # baseline copies; we replace them with symlinks pointing at the
        # cloned repo, which the entrypoint refreshes from origin/main on
        # every boot. Watcher pushes -> GitHub -> next restart picks up
        # via these symlinks.
        # Defensive: any failure here is logged but does not crash the
        # boot. Without this guard the gateway crash-loops on perms.
        for subdir in scripts tools gateway; do
            if [ -d "$SRC_DIR/$subdir" ]; then
                if [ -d "$INSTALL_DIR/$subdir" ] && [ ! -L "$INSTALL_DIR/$subdir" ]; then
                    if ! rm -rf "$INSTALL_DIR/$subdir" 2>/dev/null; then
                        echo "[entrypoint] WARNING: cannot remove $INSTALL_DIR/$subdir (perms?); leaving as-is, daemons will use image-baked copy"
                        continue
                    fi
                fi
                if ln -sfn "$SRC_DIR/$subdir" "$INSTALL_DIR/$subdir" 2>/dev/null; then
                    echo "[entrypoint] Linked $INSTALL_DIR/$subdir -> $SRC_DIR/$subdir"
                else
                    echo "[entrypoint] WARNING: cannot create symlink at $INSTALL_DIR/$subdir; skipping"
                fi
            fi
        done

        # Spawn the watcher in background. Logs to the volume so they
        # survive container restarts and can be tailed via railway ssh.
        WATCHER_SCRIPT="$INSTALL_DIR/scripts/source_watcher.py"
        if [ -f "$WATCHER_SCRIPT" ]; then
            python3 "$WATCHER_SCRIPT" >> "$HERMES_HOME/source-watcher.log" 2>&1 &
            echo "[entrypoint] Spawned source_watcher (pid $!)"
        else
            echo "[entrypoint] WARNING: $WATCHER_SCRIPT not found; auto-commit disabled"
        fi
    fi
else
    echo "[entrypoint] GITHUB_TOKEN not set — source watcher disabled (skill changes won't auto-commit to GitHub)"
fi

# ---------- Healthcheck HTTP server ----------
# Railway requires a healthcheck endpoint during deploy verification.
# Hermes is a Telegram-polling bot with no native HTTP server, so this
# tiny script binds to $PORT and returns 200 OK on any GET. Configure
# Railway's healthcheck path to anything (/, /healthz, etc.) — it all
# 200s.
HEALTHCHECK_SCRIPT="$INSTALL_DIR/scripts/healthcheck_server.py"
if [ -f "$HEALTHCHECK_SCRIPT" ]; then
    python3 "$HEALTHCHECK_SCRIPT" >> "$HERMES_HOME/healthcheck.log" 2>&1 &
    echo "[entrypoint] Spawned healthcheck server (pid $!)"
fi

# ---------- Daily cost report daemon ----------
# Sends an OpenRouter spend summary to TELEGRAM_HOME_CHANNEL every morning
# at COST_REPORT_HOUR (default 6 AM America/Los_Angeles). Pure Python — no
# LLM in the loop. Quietly skipped if any required env var is missing.
COST_DAEMON="$INSTALL_DIR/scripts/cost_report_daemon.py"
if [ -f "$COST_DAEMON" ] && [ -n "${OPENROUTER_API_KEY:-}" ] && \
   [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_HOME_CHANNEL:-}" ]; then
    python3 "$COST_DAEMON" >> "$HERMES_HOME/cost-report-daemon.log" 2>&1 &
    echo "[entrypoint] Spawned cost_report_daemon (pid $!)"
else
    [ -f "$COST_DAEMON" ] && \
        echo "[entrypoint] cost_report_daemon disabled (need OPENROUTER_API_KEY + TELEGRAM_BOT_TOKEN + TELEGRAM_HOME_CHANNEL)"
fi

# ---------- Enforce critical config on every boot ----------
# Defends against config.yaml being seeded fresh after a volume swap,
# profile clone, or any other event that resets /opt/data. Without this,
# Hermes silently falls back to whatever ships in cli-config.yaml.example
# (currently anthropic/claude-opus-4.6) — burned $16 of Opus tokens once
# already because the model setting evaporated when the volume was
# attached late in the deploy sequence.
#
# Override via env vars in Railway dashboard:
#   HERMES_ENFORCED_MODEL                 — model.default                          (default: minimax/minimax-m2.7)
#   HERMES_ENFORCED_APPROVALS             — approvals.mode                         (default: smart)
#   HERMES_ENFORCED_COMPRESSION_THRESHOLD — compression.threshold                  (default: 0.85)
#   HERMES_ENFORCED_AUX_TITLE_PROVIDER    — auxiliary.title_generation.provider    (default: minimax)
#   HERMES_ENFORCED_AUX_TITLE_MODEL       — auxiliary.title_generation.model       (default: MiniMax-M2.7)
#
# `hermes config set` is idempotent (no-op if already set). The `|| true`
# guards prevent any failure from killing the boot.
HERMES_BIN="$INSTALL_DIR/.venv/bin/hermes"
if [ -x "$HERMES_BIN" ]; then
    "$HERMES_BIN" config set model.default \
        "${HERMES_ENFORCED_MODEL:-minimax/minimax-m2.7}" >/dev/null 2>&1 \
        && echo "[entrypoint] Enforced model.default = ${HERMES_ENFORCED_MODEL:-minimax/minimax-m2.7}" \
        || echo "[entrypoint] WARNING: failed to enforce model.default"
    "$HERMES_BIN" config set approvals.mode \
        "${HERMES_ENFORCED_APPROVALS:-smart}" >/dev/null 2>&1 \
        && echo "[entrypoint] Enforced approvals.mode = ${HERMES_ENFORCED_APPROVALS:-smart}" \
        || echo "[entrypoint] WARNING: failed to enforce approvals.mode"
    "$HERMES_BIN" config set compression.threshold \
        "${HERMES_ENFORCED_COMPRESSION_THRESHOLD:-0.85}" >/dev/null 2>&1 \
        && echo "[entrypoint] Enforced compression.threshold = ${HERMES_ENFORCED_COMPRESSION_THRESHOLD:-0.85}" \
        || echo "[entrypoint] WARNING: failed to enforce compression.threshold"
    "$HERMES_BIN" config set auxiliary.title_generation.provider \
        "${HERMES_ENFORCED_AUX_TITLE_PROVIDER:-minimax}" >/dev/null 2>&1 \
        && echo "[entrypoint] Enforced auxiliary.title_generation.provider = ${HERMES_ENFORCED_AUX_TITLE_PROVIDER:-minimax}" \
        || echo "[entrypoint] WARNING: failed to enforce auxiliary.title_generation.provider"
    "$HERMES_BIN" config set auxiliary.title_generation.model \
        "${HERMES_ENFORCED_AUX_TITLE_MODEL:-MiniMax-M2.7}" >/dev/null 2>&1 \
        && echo "[entrypoint] Enforced auxiliary.title_generation.model = ${HERMES_ENFORCED_AUX_TITLE_MODEL:-MiniMax-M2.7}" \
        || echo "[entrypoint] WARNING: failed to enforce auxiliary.title_generation.model"
fi

# ---------- Startup Telegram notification ----------
# Tell the operator the container is back online after every restart
# or rebuild. Best-effort: if Telegram env isn't set or the API call
# fails, we just log and continue. Uses Python urllib because curl
# is not installed in this image.
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_HOME_CHANNEL:-}" ]; then
    if python3 - >/dev/null 2>&1 <<'PYEOF'
import os, urllib.parse, urllib.request
data = urllib.parse.urlencode({
    "chat_id": os.environ["TELEGRAM_HOME_CHANNEL"],
    "text": "I'm back!",
}).encode("utf-8")
req = urllib.request.Request(
    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
    data=data,
)
with urllib.request.urlopen(req, timeout=10) as r:
    r.read()
PYEOF
    then
        echo "[entrypoint] Sent startup Telegram notification"
    else
        echo "[entrypoint] WARNING: startup Telegram notification failed"
    fi
fi

# Final exec: two supported invocation patterns.
#
#   docker run <image>                 -> exec `hermes` with no args (legacy default)
#   docker run <image> chat -q "..."   -> exec `hermes chat -q "..."` (legacy wrap)
#   docker run <image> sleep infinity  -> exec `sleep infinity` directly
#   docker run <image> bash            -> exec `bash` directly
#
# If the first positional arg resolves to an executable on PATH, we assume the
# caller wants to run it directly (needed by the launcher which runs long-lived
# `sleep infinity` sandbox containers — see tools/environments/docker.py).
# Otherwise we treat the args as a hermes subcommand and wrap with `hermes`,
# preserving the documented `docker run <image> <subcommand>` behavior.
if [ $# -gt 0 ] && command -v "$1" >/dev/null 2>&1; then
    exec "$@"
fi
exec hermes "$@"
