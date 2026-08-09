# Debian 13 still ships SQLite 3.46.1, which contains the upstream WAL-reset
# corruption bug. Build a pinned shared library for the runtime image instead
# of relying on a distro backport that trixie does not currently provide.
# See #70480 and https://sqlite.org/wal.html#walresetbug.
FROM debian:13.4 AS sqlite_build
ARG SQLITE_AUTOCONF_VERSION=3530400
ARG SQLITE_SHA256=0e9483900e92cd5de8fd48d16bf9200145a61f7fd5be542a5ac81d8a9516eb9c
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    (curl -fsSL --retry 1 --retry-all-errors --connect-timeout 15 --max-time 60 \
        -o /tmp/sqlite.tar.gz \
        "https://sqlite.org/2026/sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}.tar.gz" || \
     curl -fsSL --retry 3 --retry-all-errors --connect-timeout 15 --max-time 120 \
        -o /tmp/sqlite.tar.gz \
        "https://sources.buildroot.net/sqlite/sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}.tar.gz") && \
    printf '%s  %s\n' "${SQLITE_SHA256}" /tmp/sqlite.tar.gz > /tmp/sqlite.sha256 && \
    sha256sum -c /tmp/sqlite.sha256 && \
    tar -xzf /tmp/sqlite.tar.gz -C /tmp && \
    cd "/tmp/sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}" && \
    CFLAGS="-O2 \
        -DSQLITE_ENABLE_FTS3 \
        -DSQLITE_ENABLE_FTS3_PARENTHESIS \
        -DSQLITE_ENABLE_FTS4 \
        -DSQLITE_ENABLE_FTS5 \
        -DSQLITE_ENABLE_RTREE \
        -DSQLITE_ENABLE_GEOPOLY \
        -DSQLITE_ENABLE_COLUMN_METADATA \
        -DSQLITE_ENABLE_UNLOCK_NOTIFY \
        -DSQLITE_ENABLE_DBSTAT_VTAB \
        -DSQLITE_ENABLE_DBPAGE_VTAB \
        -DSQLITE_ENABLE_MATH_FUNCTIONS \
        -DSQLITE_ENABLE_PREUPDATE_HOOK \
        -DSQLITE_ENABLE_SESSION \
        -DSQLITE_SECURE_DELETE \
        -DSQLITE_THREADSAFE=1 \
        -DSQLITE_MAX_VARIABLE_NUMBER=250000" \
        ./configure --prefix=/opt/sqlite-fixed --disable-static && \
    make -j"$(nproc)" && \
    make install

FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
# Node 26 source stage. Debian trixie's bundled nodejs is pinned to 20.x
# which reached EOL in April 2026 — we copy node + npm from the upstream
# node:26 image instead (Hermes pins its toolchain to Node 26 everywhere).
# Bookworm-based slim image used so the produced binary links
# against glibc 2.36, which runs cleanly on our Debian 13 (trixie, glibc
# 2.41) runtime.  Bumping to a new Node major is a one-line ARG change; see
# #4977.
FROM node:26-bookworm-slim@sha256:9e6f9357d371591e32ab6f2d8a26d63bdd0d17c29eee3f4f3e7e454d9634bf73 AS node_source
FROM debian:13.4

# Disable Python stdout buffering to ensure logs are printed immediately.
# Do not write .pyc files at runtime: /opt/hermes is immutable in the
# published container and writable state belongs under /opt/data.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    ca-certificates curl iputils-ping python3 python-is-python3 ripgrep ffmpeg gcc g++ make cmake python3-dev python3-venv libffi-dev libolm-dev libatomic1 procps git openssh-client docker-cli xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Prefer the fixed SQLite over Debian's vulnerable libsqlite3.so.0. Keep the
# public library name stable so both the system interpreter and the uv-created
# venv resolve the replacement without changing Python import paths.
COPY --from=sqlite_build /opt/sqlite-fixed/lib/libsqlite3.so.3.53.4 /usr/local/lib/
RUN ln -sf libsqlite3.so.3.53.4 /usr/local/lib/libsqlite3.so.0 && \
    ln -sf libsqlite3.so.3.53.4 /usr/local/lib/libsqlite3.so && \
    printf '/usr/local/lib\n' > /etc/ld.so.conf.d/000-sqlite-fixed.conf && \
    ldconfig && \
    python3 -c "import sqlite3, sys; \
v = sqlite3.sqlite_version_info; \
sys.exit(f'linked SQLite {sqlite3.sqlite_version} still has the WAL-reset bug') if v < (3, 51, 3) else None; \
db = sqlite3.connect(':memory:'); \
db.execute(\"CREATE VIRTUAL TABLE docs USING fts5(content, tokenize='trigram')\"); \
db.execute(\"INSERT INTO docs VALUES ('hermes')\"); \
sys.exit('SQLite FTS5 trigram self-test failed') if db.execute(\"SELECT count(*) FROM docs WHERE docs MATCH 'erm'\").fetchone()[0] != 1 else None; \
db.close()"

# ---------- s6-overlay install ----------
# s6-overlay provides supervision for the main hermes process, the dashboard,
# and per-profile gateways. /init becomes PID 1 below — see ENTRYPOINT.
#
# Multi-arch: BuildKit auto-populates TARGETARCH (amd64 / arm64). s6-overlay
# uses tarball names keyed on the kernel arch string (x86_64 / aarch64), so
# we map between them inline. The noarch + symlinks tarballs are
# architecture-independent and reused as-is.
#
# We use `curl` instead of `ADD` for ALL three tarballs: `ADD` evaluates its
# URL at parse time (no ARG / TARGETARCH substitution) and — critically for
# CI reliability — cannot retry, so a single GitHub-release CDN blip fails
# the whole 15-45 min build. curl -fsSL --retry 3 self-heals those blips,
# and every tarball is still checksum-verified below before extraction.
ARG TARGETARCH
ARG S6_OVERLAY_VERSION=3.2.3.0
ARG S6_OVERLAY_NOARCH_SHA256=b720f9d9340efc8bb07528b9743813c836e4b02f8693d90241f047998b4c53cf
ARG S6_OVERLAY_X86_64_SHA256=a93f02882c6ed46b21e7adb5c0add86154f01236c93cd82c7d682722e8840563
ARG S6_OVERLAY_AARCH64_SHA256=0952056ff913482163cc30e35b2e944b507ba1025d78f5becbb89367bf344581
ARG S6_OVERLAY_SYMLINKS_SHA256=a60dc5235de3ecbcf874b9c1f18d73263ab99b289b9329aa950e8729c4789f0e
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) s6_arch="x86_64"; s6_arch_sha="${S6_OVERLAY_X86_64_SHA256}" ;; \
        arm64) s6_arch="aarch64"; s6_arch_sha="${S6_OVERLAY_AARCH64_SHA256}" ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH} for s6-overlay" >&2; exit 1 ;; \
    esac; \
    base="https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}"; \
    curl -fsSL --retry 3 -o /tmp/s6-overlay-noarch.tar.xz \
        "${base}/s6-overlay-noarch.tar.xz"; \
    curl -fsSL --retry 3 -o /tmp/s6-overlay-symlinks-noarch.tar.xz \
        "${base}/s6-overlay-symlinks-noarch.tar.xz"; \
    curl -fsSL --retry 3 -o /tmp/s6-overlay-arch.tar.xz \
        "${base}/s6-overlay-${s6_arch}.tar.xz"; \
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

# #34192 / #66679: backward-compat shim for orchestration templates that
# still reference the legacy /usr/bin/tini entrypoint (Hostinger's
# 'Hermes WebUI' catalog, NAS compose projects that preserve an old
# entrypoint on image update, etc.). A plain symlink to /init made the
# path exist, but forwarded tini flags like `-g` into s6-overlay's
# rc.init as the container CMD (`rc.init: 91: -g: not found`) and
# boot-looped any `restart: unless-stopped` deploy. The shim strips the
# tini CLI surface, then exec's /init + main-wrapper — see
# docker/tini-shim.sh. Safe to drop once the affected catalogs are
# updated.
COPY --chmod=0755 docker/tini-shim.sh /usr/bin/tini

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# Node 26: copy the node binary plus the bundled npm JS install from the
# upstream image.  npm and npx are recreated as symlinks because they're
# symlinks in the source image (and need to live on PATH).
#
# No corepack: Node unbundled it upstream, so node:26 ships only npm in
# /usr/local/lib/node_modules.  Nothing here needs it — no package.json
# declares a `packageManager`, and no build step shells out to yarn or pnpm.
#
# See node_source stage at the top of the file for the version-bump
# rationale (#4977).
COPY --chmod=0755 --from=node_source /usr/local/bin/node /usr/local/bin/
COPY --from=node_source /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

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
COPY web/package.json web/
COPY ui-tui/package.json ui-tui/
COPY ui-tui/packages/hermes-ink/ ui-tui/packages/hermes-ink/
# apps/shared/ is copied IN FULL because web/package.json references it as a
# `file:` workspace dependency (same pattern as hermes-ink above).
COPY apps/shared/ apps/shared/

# `npm_config_install_links=false` forces npm to install `file:` deps as
# symlinks instead of copies.  This is the default since npm 10+, which is
# what the image ships now (via the node:22 source stage).  We set it
# explicitly anyway as defense-in-depth: the previous Debian-bundled npm
# 9.x defaulted to install-as-copy, which produced a hidden
# node_modules/.package-lock.json that permanently disagreed with the root
# lock on the @hermes/ink entry, tripped the TUI launcher's
# `_tui_need_npm_install()` check on every startup, and triggered a
# runtime `npm install` that then failed with EACCES.  Keeping the env
# guards against a future regression if the source npm version changes.
ENV npm_config_install_links=false

RUN npm install --prefer-offline --no-audit --fetch-retries=5 && \
    for i in 1 2 3; do \
        npx playwright install --with-deps chromium --only-shell && break || \
        { [ "$i" = 3 ] && exit 1; echo "playwright install failed (attempt $i); retrying in 10s"; sleep 10; }; \
    done && \
    npm cache clean --force

# ---------- Photon iMessage sidecar deps (baked, NS-606) ----------
# The photon plugin's Node sidecar needs its own node_modules
# (spectrum-ts). The install tree is immutable at runtime, so a lazy
# `npm ci` on first connect would hit EROFS — bake the deps here instead
# (deterministic installs, NS-559). The patch script is copied alongside
# the manifests because package.json's postinstall runs it, which also
# means the spectrum-ts patch is applied at build time. Layer-cached:
# only re-runs when the sidecar manifests/patch change.
COPY plugins/platforms/photon/sidecar/package.json \
     plugins/platforms/photon/sidecar/package-lock.json \
     plugins/platforms/photon/sidecar/patch-spectrum-mixed-attachments.mjs \
     plugins/platforms/photon/sidecar/
RUN cd plugins/platforms/photon/sidecar && \
    npm ci --no-audit --fetch-retries=5 && \
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
# `uv sync --frozen --no-install-project --extra all --extra messaging --extra otlp`
# installs the deps reachable through the composite `[all]` extra
# (handpicked set intended for the production image — excludes `[dev]`),
# plus gateway messaging adapters that should work in the published image
# without a first-boot lazy install.  We do NOT use `--all-extras`:
# that would pull in `[rl]` (atroposlib + tinker + torch + wandb from
# git), `[yc-bench]` (another git dep), and `[termux-all]` (Android
# redundancy), none of which belong in the published container.
#
# Provider packages (anthropic, bedrock, azure-identity) are included
# so Docker users can use these providers without requiring runtime
# lazy-install access to PyPI (often blocked in containerized envs).
#
# The [otlp] extra contains the SDK/exporter imported by Hermes when Gateway
# Health export is enabled. Collector and observability-backend dependencies
# remain external and are not part of the Hermes production image.
#
# The hindsight memory provider's client (hindsight-client) is baked in
# for the same reason: it lazy-installs into /opt/hermes/.venv at first
# use, which lives inside the (immutable) image layer rather than the
# mounted /opt/data volume, so it is lost on every container recreate /
# image update and recall/retain then fails with
# `ModuleNotFoundError: No module named 'hindsight_client'` (#38128).
#
# The Matrix gateway's deps ([matrix] extra) are baked in because
# python-olm (transitive via mautrix[encryption]) builds from source on
# Python/image combinations without usable wheels.  The Docker image is
# Linux-only, so keeping the native libolm/build-toolchain packages here
# avoids the cross-platform failures that kept [matrix] out of [all]
# while still making Matrix work in the published container. Fixes #30399.
#
# The editable link is created after the source copy below.
COPY pyproject.toml uv.lock ./
RUN touch ./README.md
# `kubernetes` joins the other extras that are baked but deliberately outside
# [all] (anthropic, bedrock, azure-identity, hindsight, matrix). The terminal
# tool is gated on the client being importable, and the lazy-install path is
# not reachable in a locked-down cluster deployment: /opt/hermes is root-owned
# and immutable at runtime, and a session-pod NetworkPolicy will not permit
# egress to PyPI. Baking it keeps `terminal.backend: kubernetes` usable in the
# published image.
RUN uv sync --frozen --no-install-project --extra all --extra messaging --extra otlp --extra anthropic --extra bedrock --extra azure-identity --extra hindsight --extra matrix --extra kubernetes

# ---------- Frontend build (cached independently from Python source) ----------
# Copy only the frontend source trees first so that Python-only changes don't
# invalidate the (relatively slow) web + ui-tui build layer.
COPY web/ web/
COPY ui-tui/ ui-tui/
COPY apps/shared/ apps/shared/
RUN cd web && npm run build && \
    cd ../ui-tui && npm run build

# ---------- Source code ----------
# .dockerignore excludes node_modules, so the installs above survive.
# --link decouples this layer from parents for cache purposes; --chmod bakes
# the final read-only permissions at copy time so we skip the separate
# `chmod -R` pass that previously walked ~30k files across the venv +
# node_modules + source (21s amd64 / 222s arm64 — #49113).  `a+rX,go-w`
# gives the non-root hermes user read + traverse but no write; root retains
# write so the build steps below don't need chmod u+w dances.
COPY --link --chmod=a+rX,go-w . .

# ---------- Permissions ----------
# Link hermes-agent itself (editable). Deps are already installed in the
# cached layer above; `--no-deps` makes this a fast egg-link creation with no
# resolution or downloads.
RUN uv pip install --no-cache-dir --no-deps -e "."

# Wire the exec shim and install-method stamp.  Files under /opt/hermes are
# already root-owned (COPY, uv sync, npm install all run as root) and
# read-only for the hermes user (go-w from the --chmod above).

USER root
RUN mkdir -p /opt/hermes/bin && \
    cp /opt/hermes/docker/hermes-exec-shim.sh /opt/hermes/bin/hermes && \
    chmod 0755 /opt/hermes/bin/hermes && \
    printf 'docker\n' > /opt/hermes/.install_method
# The ``.install_method`` stamp is baked next to the running code (the install
# tree), NOT into $HERMES_HOME. $HERMES_HOME (/opt/data) is a shared data
# volume that is commonly bind-mounted from the host and even shared with a
# host-side Desktop/CLI install; stamping it at boot used to clobber that
# host install's marker and wrongly block its ``hermes update``. A code-scoped
# stamp is read first by detect_install_method() and is immune to the share.
# Start as root so the s6-overlay stage2 hook can usermod/groupmod and chown
# the data volume. Each supervised service then drops to the hermes user via
# `s6-setuidgid hermes` in its run script. If HERMES_UID is unset, services
# run as the default hermes user (UID 10000).

# ---------- Bake build-time git revision ----------
# .dockerignore excludes .git, so `git rev-parse HEAD` from inside the
# container always returns nothing — meaning `hermes dump` reports
# "(unknown)" and the startup banner drops its `· upstream <sha>` suffix.
# That makes support triage from container bug reports impossible:
# we can't tell which commit the user is actually running.
#
# Fix: write the commit SHA passed via the HERMES_GIT_SHA build-arg to
# /opt/hermes/.hermes_build_sha at build time, and have
# hermes_cli/build_info.py read it at runtime.  Both `hermes dump` and
# banner.get_git_banner_state() try the baked SHA first, then fall back
# to live `git rev-parse` for source installs (unchanged behaviour).
#
# The arg is optional — local `docker build` without --build-arg simply
# omits the file, and the runtime falls back to live-git lookup.  CI
# (.github/workflows/docker.yml) passes ${{ github.sha }} so
# every published image has it.
ARG HERMES_GIT_SHA=
RUN if [ -n "${HERMES_GIT_SHA}" ]; then \
        printf '%s\n' "${HERMES_GIT_SHA}" > /opt/hermes/.hermes_build_sha; \
    fi

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
    printf '#!/command/with-contenv sh\nexec /opt/hermes/docker/stage2-hook.sh\n' \
        > /etc/cont-init.d/01-hermes-setup && \
    chmod +x /etc/cont-init.d/01-hermes-setup
COPY --chmod=0755 docker/cont-init.d/015-supervise-perms /etc/cont-init.d/015-supervise-perms
COPY --chmod=0755 docker/cont-init.d/02-reconcile-profiles /etc/cont-init.d/02-reconcile-profiles

# ---------- Runtime ----------
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
# Point the TUI launcher at the prebuilt bundle baked at build time (Layer 8:
# `ui-tui && npm run build`). This makes _make_tui_argv take the prebuilt-bundle
# fast path (`node --expose-gc /opt/hermes/ui-tui/dist/entry.js`) and skip the
# _tui_need_npm_install / runtime `npm install` branch entirely — exactly the
# nix/packaged-release path the launcher was designed for.
#
# Why this is required (not just an optimization): the root package-lock.json
# describes the WHOLE monorepo workspace set (root + web + ui-tui + apps/*),
# but the image only installs root/web/ui-tui (apps/* — the desktop app — is
# never `npm install`ed here). So the actualized node_modules permanently
# disagrees with the canonical lock, _tui_need_npm_install() returns True on
# every launch, and the runtime `npm install` it triggers (a) can never
# converge against the partial monorepo and (b) races itself across concurrent
# embedded-chat (/api/pty) connections → ENOTEMPTY → the chat tab dies with a
# 502 / "[session ended]". Pointing at the prebuilt bundle sidesteps the whole
# check. (A separate launcher hardening is tracked independently.)
ENV HERMES_TUI_DIR=/opt/hermes/ui-tui
ENV HERMES_HOME=/opt/data
ENV HERMES_WRITE_SAFE_ROOT=/opt/data
ENV HERMES_DISABLE_LAZY_INSTALLS=1
# The published image seals /opt/hermes (root-owned, read-only) so a runtime
# lazy install can't mutate the agent's own venv and brick it. But opt-in
# backends (Firecrawl web search, Exa, Feishu, …) keep their SDKs in
# tools/lazy_deps.py — deliberately NOT baked into [all] (see pyproject.toml
# policy 2026-05-12: one quarantined release must not break every install).
# Redirect those lazy installs to a writable dir on the durable data volume.
# lazy_deps appends this dir to the END of sys.path, so a package installed
# here can only ADD modules — it can never shadow or downgrade a core module,
# so the sealed-venv guarantee holds even with installs re-enabled. The dir
# is seeded + chowned to the hermes user by docker/stage2-hook.sh and lives
# on the /opt/data volume, so it persists across container recreates / image
# updates (an ABI stamp invalidates it if a rebuild bumps the interpreter).
ENV HERMES_LAZY_INSTALL_TARGET=/opt/data/lazy-packages

# `docker exec` privilege-drop shim. When operators run
# `docker exec <c> hermes ...` they default to root, and any file the
# command writes under $HERMES_HOME (auth.json, .env, config.yaml) ends
# up root-owned and unreadable to the supervised gateway (UID 10000).
# The shim lives at /opt/hermes/bin/hermes, sits earliest on PATH, and
# transparently re-exec's the real venv binary via `s6-setuidgid hermes`
# when invoked as root. Non-root callers (supervised processes,
# `--user hermes`, etc.) hit the short-circuit path with no overhead.
# Recursion is impossible because the shim exec's the venv binary by
# absolute path (/opt/hermes/.venv/bin/hermes). See the shim source for
# the opt-out env var (HERMES_DOCKER_EXEC_AS_ROOT=1).
COPY --chmod=0755 docker/hermes-exec-shim.sh /opt/hermes/bin/hermes
COPY --chmod=0755 docker/entrypoint-dispatch.sh /opt/hermes/docker/entrypoint-dispatch.sh

# Pre-s6 entrypoint.sh did `source .venv/bin/activate` which exported
# the venv bin onto PATH; Architecture B's main-wrapper.sh does the
# same for the container's main process, but `docker exec` and our
# cont-init.d scripts don't pass through the wrapper. Expose the venv
# bin globally so `docker exec <container> hermes ...` and any
# subprocess that doesn't activate the venv first still find hermes.
#
# /opt/hermes/bin is prepended ahead of the venv so the privilege-drop
# shim wins PATH resolution. The shim's last act is to exec the venv
# binary by absolute path, so this PATH ordering is transparent to
# every other consumer.
ENV PATH="/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin:${PATH}"
RUN mkdir -p /opt/data
VOLUME [ "/opt/data" ]

# The image ENTRYPOINT is a tiny dispatcher rather than `/init` directly.
# When the image really owns PID 1 (normal Docker / Podman), the dispatcher
# execs `/init` and preserves the full s6 supervision tree. When a platform
# wraps the image entrypoint under its own PID-1 init (Fly Machines,
# `docker run --init`, some schedulers), `/init` would abort with
# `can only run as pid 1`; in that case the dispatcher falls back to
# `stage2-hook.sh` + `main-wrapper.sh` directly so foreground commands still
# work. See #38349.
#
# On the PID-1 path, s6-overlay's /init sets up the supervision tree, runs
# /etc/cont-init.d/* (our stage2 hook), starts s6-rc services declared in
# /etc/s6-overlay/s6-rc.d/, then exec's its remaining argv as the container's
# "main program" with stdin/stdout/stderr inherited (this is what makes
# interactive --tui work). When the main program exits, /init begins stage 3
# shutdown and the container exits with the program's exit code. Replaces
# tini — see Phase 2 of docs/plans/2026-05-07-s6-overlay-dynamic-subagent-gateways.md.
#
# We use the ENTRYPOINT+CMD split rather than CMD alone so the
# wrapper is prepended to user-supplied args automatically:
#
#   docker run <image>                  → entrypoint-dispatch.sh   (CMD default)
#   docker run <image> chat -q "hi"     → entrypoint-dispatch.sh chat -q hi
#   docker run <image> sleep infinity   → entrypoint-dispatch.sh sleep infinity
#   docker run <image> --tui            → entrypoint-dispatch.sh --tui
#
# main-wrapper.sh handles arg routing (bare-exec vs. hermes
# subcommand vs. no-args), drops to the hermes user via s6-setuidgid,
# and exec's the final program so its exit code becomes the container
# exit code. The dispatcher preserves that contract across both the
# supervised PID-1 path and the non-PID-1 fallback path. Without the
# wrapper-as-ENTRYPOINT, leading-dash args like `--version` would be
# intercepted by /init's POSIX shell.
ENTRYPOINT [ "/opt/hermes/docker/entrypoint-dispatch.sh" ]
CMD [ ]
