FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM debian:13.4

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Store Playwright browsers outside the volume mount so the build-time
# install survives the /opt/data volume overlay at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# Install system dependencies in one layer, clear APT cache.
# tini was previously PID 1 to reap orphaned zombie processes (MCP stdio
# subprocesses, git, bun, etc.) that would otherwise accumulate when hermes
# ran as PID 1. See #15012. Phase 2 of the s6-overlay supervision plan
# replaces tini with s6-overlay's /init (PID 1 = s6-svscan), which reaps
# zombies non-blockingly on SIGCHLD and additionally supervises the main
# hermes process, the dashboard, and per-profile gateways.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev procps git openssh-client docker-cli xz-utils && \
    rm -rf /var/lib/apt/lists/*

# ---------- s6-overlay install ----------
# s6-overlay provides supervision for the main hermes process, the dashboard,
# and per-profile gateways. /init becomes PID 1 below — see ENTRYPOINT.
#
# Multi-arch: BuildKit auto-populates TARGETARCH (amd64 / arm64). s6-overlay
# uses tarball names keyed on the kernel arch string (x86_64 / aarch64), so
# we map between them inline. The noarch + symlinks tarballs are
# architecture-independent and reused as-is.
#
# We use `curl` instead of `ADD` for the per-arch tarball because `ADD`
# evaluates its URL at parse time, before any ARG / TARGETARCH substitution
# — splitting one URL per arch into two ADDs would download both on every
# build and leave dead bytes in the cache. A single curl + arch-keyed URL
# is simpler and cache-friendlier.
#
# Supply-chain integrity: every tarball is checksum-verified against the
# upstream-published SHA256. To bump S6_OVERLAY_VERSION, fetch the four
# `.sha256` files from the corresponding release and update the ARGs. The
# checksum lookup happens during build, so a compromised release artifact
# fails the build loudly instead of silently producing a tampered image.
ARG TARGETARCH
ARG S6_OVERLAY_VERSION=3.2.3.0
ARG S6_OVERLAY_NOARCH_SHA256=b720f9d9340efc8bb07528b9743813c836e4b02f8693d90241f047998b4c53cf
ARG S6_OVERLAY_X86_64_SHA256=a93f02882c6ed46b21e7adb5c0add86154f01236c93cd82c7d682722e8840563
ARG S6_OVERLAY_AARCH64_SHA256=0952056ff913482163cc30e35b2e944b507ba1025d78f5becbb89367bf344581
ARG S6_OVERLAY_SYMLINKS_SHA256=a60dc5235de3ecbcf874b9c1f18d73263ab99b289b9329aa950e8729c4789f0e
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp/
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz /tmp/
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) s6_arch="x86_64"; s6_arch_sha="${S6_OVERLAY_X86_64_SHA256}" ;; \
        arm64) s6_arch="aarch64"; s6_arch_sha="${S6_OVERLAY_AARCH64_SHA256}" ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH} for s6-overlay" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 -o /tmp/s6-overlay-arch.tar.xz \
        "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${s6_arch}.tar.xz"; \
    { \
        printf '%s  %s\n' "${S6_OVERLAY_NOARCH_SHA256}" /tmp/s6-overlay-noarch.tar.xz; \
        printf '%s  %s\n' "${s6_arch_sha}" /tmp/s6-overlay-arch.tar.xz; \
        printf '%s  %s\n' "${S6_OVERLAY_SYMLINKS_SHA256}" /tmp/s6-overlay-symlinks-noarch.tar.xz; \
    } > /tmp/s6-overlay.sha256; \
    sha256sum -c /tmp/s6-overlay.sha256; \
    tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz; \
    tar -C / -Jxpf /tmp/s6-overlay-arch.tar.xz; \
    tar -C / -Jxpf /tmp/s6-overlay-symlinks-noarch.tar.xz; \
    rm /tmp/s6-overlay-*.tar.xz /tmp/s6-overlay.sha256

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /opt/hermes

# ---------- Layer-cached dependency install ----------
# Copy only package manifests first so npm install + Playwright are cached
# unless the lockfiles themselves change.
#
# ui-tui/packages/hermes-ink/ is copied IN FULL (not just its manifests)
# because it is referenced as a `file:` workspace dependency from
# ui-tui/package.json.  Copying the tree up front lets npm resolve the
# workspace to real content instead of stopping at a bare package.json.
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/
COPY ui-tui/package.json ui-tui/package-lock.json ui-tui/
COPY ui-tui/packages/hermes-ink/ ui-tui/packages/hermes-ink/

# `npm_config_install_links=false` forces npm to install `file:` deps as
# symlinks (the npm 10+ default) even on Debian's older bundled npm 9.x,
# which defaults to `install-links=true` and installs file deps as *copies*.
# The host-side package-lock.json is generated with a newer npm that uses
# symlinks, so an install-as-copy produces a hidden node_modules/.package-lock.json
# that permanently disagrees with the root lock on the @hermes/ink entry.
# That disagreement trips the TUI launcher's `_tui_need_npm_install()`
# check on every startup and triggers a runtime `npm install` that then
# fails with EACCES (node_modules/ is root-owned from build time).
ENV npm_config_install_links=false

RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    (cd web && npm install --prefer-offline --no-audit) && \
    (cd ui-tui && npm install --prefer-offline --no-audit) && \
    npm cache clean --force

# ---------- Layer-cached Python dependency install ----------
# Copy only pyproject.toml + uv.lock so the Python dep resolve + wheel
# download + native-extension compile layer is cached unless those inputs
# change.  Before this split the Python install sat after `COPY . .`, so
# every source-only commit re-did ~4-5 min of dep work on cold builds.
#
# README.md is referenced by pyproject.toml's `readme =` field, but it's
# excluded from the build context by .dockerignore's `*.md`.  uv's build
# frontend stats the readme path during dep resolution, so we `touch` an
# empty placeholder — the real README is restored by `COPY . .` below.
#
# `uv sync --frozen --no-install-project --extra all --extra messaging`
# installs the deps reachable through the composite `[all]` extra
# (handpicked set intended for the production image), plus gateway
# messaging adapters that should work in the published image without a
# first-boot lazy install.  We do NOT use `--all-extras`:
# that would pull in `[rl]` (atroposlib + tinker + torch + wandb from
# git), `[yc-bench]` (another git dep), and `[termux-all]` (Android
# redundancy), none of which belong in the published container.
#
# The editable link is created after the source copy below.
COPY pyproject.toml uv.lock ./
RUN touch ./README.md
RUN uv sync --frozen --no-install-project --extra all --extra messaging

# ---------- Source code ----------
# .dockerignore excludes node_modules, so the installs above survive.
COPY --chown=hermes:hermes . .

# Build browser dashboard and terminal UI assets.
RUN cd web && npm run build && \
    cd ../ui-tui && npm run build

# ---------- Permissions ----------
# Make install dir world-readable so any HERMES_UID can read it at runtime.
# The venv needs to be traversable too.
# node_modules trees additionally need to be writable by the hermes user
# so the runtime `npm install` triggered by _tui_need_npm_install() in
# hermes_cli/main.py succeeds (see #18800). /opt/hermes/web is build-time
# only (HERMES_WEB_DIST points at hermes_cli/web_dist) and is intentionally
# not chowned here.
# The .venv MUST remain hermes-writable so lazy_deps.py can install
# remaining optional platform packages and future pin bumps at first use.
# Without this, `uv pip install` fails with EACCES and adapters silently
# fail to load.  See tools/lazy_deps.py.
USER root
RUN chmod -R a+rX /opt/hermes && \
    chown -R hermes:hermes /opt/hermes/.venv /opt/hermes/ui-tui /opt/hermes/node_modules
# Start as root so the s6-overlay stage2 hook can usermod/groupmod and chown
# the data volume. Each supervised service then drops to the hermes user via
# `s6-setuidgid hermes` in its run script. If HERMES_UID is unset, services
# run as the default hermes user (UID 10000).

# ---------- Link hermes-agent itself (editable) ----------
# Deps are already installed in the cached layer above; `--no-deps` makes
# this a fast (~1s) egg-link creation with no resolution or downloads.
RUN uv pip install --no-cache-dir --no-deps -e "."

# ---------- s6-overlay service wiring ----------
# Static services declared at build time: main-hermes + dashboard.
# Per-profile gateway services are registered dynamically at runtime by
# the profile create/delete hooks (Phase 4); they live under
# /run/service/ (tmpfs) and are reconciled on container restart by
# /etc/cont-init.d/02-reconcile-profiles (Phase 4 Task 4.0).
COPY docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/

# stage2-hook handles UID/GID remap, volume chown, config seeding,
# skills sync — all the work the old entrypoint.sh did before
# `exec hermes`. Wired in as cont-init.d/01- so it
# runs before user services start.
#
# 02-reconcile-profiles re-creates per-profile gateway s6 service
# slots from $HERMES_HOME/profiles/<name>/ after a container restart
# (the /run/service/ scandir is tmpfs and wiped on restart). Phase 4.
RUN mkdir -p /etc/cont-init.d && \
    printf '#!/bin/sh\nexec /opt/hermes/docker/stage2-hook.sh\n' \
        > /etc/cont-init.d/01-hermes-setup && \
    chmod +x /etc/cont-init.d/01-hermes-setup
COPY --chmod=0755 docker/cont-init.d/015-supervise-perms /etc/cont-init.d/015-supervise-perms
COPY --chmod=0755 docker/cont-init.d/02-reconcile-profiles /etc/cont-init.d/02-reconcile-profiles

# ---------- Runtime ----------
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
# Pre-s6 entrypoint.sh did `source .venv/bin/activate` which exported
# the venv bin onto PATH; Architecture B's main-wrapper.sh does the
# same for the container's main process, but `docker exec` and our
# cont-init.d scripts don't pass through the wrapper. Expose the venv
# bin globally so `docker exec <container> hermes ...` and any
# subprocess that doesn't activate the venv first still find hermes.
ENV PATH="/opt/hermes/.venv/bin:/opt/data/.local/bin:${PATH}"
RUN mkdir -p /opt/data
VOLUME [ "/opt/data" ]

# s6-overlay's /init is PID 1. It sets up the supervision tree, runs
# /etc/cont-init.d/* (our stage2 hook), starts s6-rc services
# declared in /etc/s6-overlay/s6-rc.d/, then exec's its remaining
# argv as the container's "main program" with stdin/stdout/stderr
# inherited (this is what makes interactive --tui work). When the
# main program exits, /init begins stage 3 shutdown and the container
# exits with the program's exit code. Replaces tini — see Phase 2 of
# docs/plans/2026-05-07-s6-overlay-dynamic-subagent-gateways.md.
#
# We use the ENTRYPOINT+CMD split rather than CMD alone so the
# wrapper is prepended to user-supplied args automatically:
#
#   docker run <image>                  → /init main-wrapper.sh   (CMD default)
#   docker run <image> chat -q "hi"     → /init main-wrapper.sh chat -q hi
#   docker run <image> sleep infinity   → /init main-wrapper.sh sleep infinity
#   docker run <image> --tui            → /init main-wrapper.sh --tui
#
# main-wrapper.sh handles arg routing (bare-exec vs. hermes
# subcommand vs. no-args), drops to the hermes user via s6-setuidgid,
# and exec's the final program so its exit code becomes the container
# exit code. Without the wrapper-as-ENTRYPOINT, leading-dash args
# like `--version` would be intercepted by /init's POSIX shell.
ENTRYPOINT [ "/init", "/opt/hermes/docker/main-wrapper.sh" ]
CMD [ ]
