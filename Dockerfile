# syntax=docker/dockerfile:1
#
# Multi-arch Hermes image: linux/amd64, linux/arm64, linux/riscv64.
#
# amd64/arm64 are UNCHANGED from before: uv and Node are copied from the
# upstream images. riscv64 has no manifest for those images and no manylinux
# riscv64 wheels for the native Python deps, so it takes alternate paths gated
# on TARGETARCH, each marked `# riscv64:`:
#   - uv and Node come from checksum-pinned riscv64 tarballs, selected via the
#     `FROM <name>_${TARGETARCH}` per-arch-stage pattern below. BuildKit prunes
#     the unreferenced (image-based) stages, so the riscv64-less uv/node images
#     are never pulled on a riscv64 build.
#   - a rustup toolchain + a few build deps are added so uv compiles the native
#     wheels from sdist (no-op on amd64/arm64, which use prebuilt wheels).
#   - npm uses `npm ci` (npm 10.9.x `npm install` crashes reconciling optional
#     deps on a platform absent from the lockfile), and Playwright is skipped
#     (no riscv64 Chromium).
#   - the web dashboard (Tailwind v4 -> lightningcss + @tailwindcss/oxide, no
#     riscv64 napi binary) is built on the BUILD host by CI and injected via
#     the build context; only the TUI builds in-image on riscv64. See the
#     docker-publish.yml change that ships with this.

# TARGETARCH must be declared before the FROM stages that interpolate it.
ARG TARGETARCH

# ---------- uv source (per-arch) ----------
# amd64/arm64: the upstream uv image (its manifest covers both). riscv64:
# download Astral's standalone riscv64 binary into a debian stage. Only the
# stage named uv_${TARGETARCH} is referenced below, so on riscv64 the two
# image stages are pruned and the uv image (no riscv64 manifest) is not pulled.
FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_amd64
FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_arm64
FROM debian:13.4 AS uv_riscv64
ARG UV_VERSION=0.11.6
ARG UV_RISCV64_SHA256=0e3ead8667b51b07b5fb9d114bcd1914a5fe3159e6959a584dc2f89c6724e123
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL --retry 3 -o /tmp/uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-riscv64gc-unknown-linux-gnu.tar.gz"; \
    printf '%s  %s\n' "${UV_RISCV64_SHA256}" /tmp/uv.tar.gz | sha256sum -c; \
    tar -C /usr/local/bin --strip-components=1 -xzf /tmp/uv.tar.gz \
        uv-riscv64gc-unknown-linux-gnu/uv uv-riscv64gc-unknown-linux-gnu/uvx; \
    chmod 0755 /usr/local/bin/uv /usr/local/bin/uvx; \
    rm /tmp/uv.tar.gz
FROM uv_${TARGETARCH} AS uv_source

# ---------- Node 22 LTS source (per-arch) ----------
# Debian trixie's bundled nodejs is pinned to 20.x which reached EOL in April
# 2026 — we take node + npm + corepack from the upstream node:22 image instead
# so we can stay on a supported LTS without waiting for Debian 14 (forky,
# ~mid-2027). Bookworm-based slim image is used so the produced binary links
# against glibc 2.36, which runs cleanly on our Debian 13 (trixie, glibc 2.41)
# runtime. Bumping to a new Node major is a one-line ARG change; see #4977.
#
# riscv64: nodejs.org/dist ships no riscv64 build; the Node project's
# unofficial-builds sub-project cross-compiles it (Ubuntu 24.04, glibc 2.39,
# which also runs on trixie). Extracting the whole prefix lays down
# /usr/local/bin/node + lib/node_modules/{npm,corepack}, so the COPY --from
# below is identical across arches.
FROM node:22-bookworm-slim@sha256:7af03b14a13c8cdd38e45058fd957bf00a72bbe17feac43b1c15a689c029c732 AS node_amd64
FROM node:22-bookworm-slim@sha256:7af03b14a13c8cdd38e45058fd957bf00a72bbe17feac43b1c15a689c029c732 AS node_arm64
FROM debian:13.4 AS node_riscv64
ARG NODE_VERSION=22.22.3
ARG NODE_RISCV64_SHA256=7a2c78e87eb154d450c5cccc6f9a8b3de1a0854ee78f3ef92a2d7e4c71d4985f
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl xz-utils; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL --retry 3 -o /tmp/node.tar.xz \
        "https://unofficial-builds.nodejs.org/download/release/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-riscv64.tar.xz"; \
    printf '%s  %s\n' "${NODE_RISCV64_SHA256}" /tmp/node.tar.xz | sha256sum -c; \
    tar -C /usr/local --strip-components=1 \
        --exclude='*/CHANGELOG.md' --exclude='*/LICENSE' --exclude='*/README.md' \
        -xJf /tmp/node.tar.xz; \
    rm /tmp/node.tar.xz
FROM node_${TARGETARCH} AS node_source

FROM debian:13.4

# Re-declare in this stage so the conditional riscv64 RUN steps below can
# read it (ARG does not cross FROM boundaries).
ARG TARGETARCH

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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl iputils-ping python3 python-is-python3 ripgrep ffmpeg gcc g++ make cmake python3-dev python3-venv libffi-dev libolm-dev procps git openssh-client docker-cli xz-utils && \
    rm -rf /var/lib/apt/lists/*

# riscv64: there are no manylinux riscv64 wheels for the native Python
# extensions (cryptography, pydantic-core, jiter, pynacl, ...), so they compile
# from sdist during `uv sync` below. That needs a C/C++ toolchain plus Rust,
# and Debian trixie's rustc (1.85) is too old for some crates (davey/openmls
# needs the 1.87 `unsigned_is_multiple_of`), so install rustup. This block is a
# no-op on amd64/arm64 (their wheels are prebuilt), keeping those images lean.
# The RUSTUP/CARGO env is harmless on amd64/arm64 (the dirs just won't exist).
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}
ARG RUST_VERSION=1.96.0
RUN set -eu; \
    if [ "${TARGETARCH:-amd64}" = "riscv64" ]; then \
        apt-get update; \
        # libssl-dev/pkg-config: native Rust crates (davey/openmls). libjpeg-dev
        # + zlib1g-dev: Pillow compiles from source on riscv64 (no wheel) and is
        # pulled by the [matrix] extra via mautrix[encryption]; amd64/arm64 use
        # Pillow wheels so they need neither.
        apt-get install -y --no-install-recommends g++ make pkg-config libssl-dev libjpeg-dev zlib1g-dev; \
        rm -rf /var/lib/apt/lists/*; \
        curl --proto '=https' --tlsv1.2 -fsSL -o /tmp/rustup-init.sh https://sh.rustup.rs; \
        sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${RUST_VERSION}" --no-modify-path; \
        rm /tmp/rustup-init.sh; \
        chmod -R a+rX "${RUSTUP_HOME}" "${CARGO_HOME}"; \
        rustc --version; \
    fi

# ---------- s6-overlay install ----------
# s6-overlay provides supervision for the main hermes process, the dashboard,
# and per-profile gateways. /init becomes PID 1 below — see ENTRYPOINT.
#
# Multi-arch: BuildKit auto-populates TARGETARCH. s6-overlay uses tarball names
# keyed on the kernel arch string (x86_64 / aarch64 / riscv64), so we map
# between them inline. The noarch + symlinks tarballs are
# architecture-independent and reused as-is.
#
# We use `curl` instead of `ADD` for the per-arch tarball because `ADD`
# evaluates its URL at parse time, before any ARG / TARGETARCH substitution
# — splitting one URL per arch into multiple ADDs would download them all on
# every build and leave dead bytes in the cache. A single curl + arch-keyed
# URL is simpler and cache-friendlier.
#
# Supply-chain integrity: every tarball is checksum-verified against the
# upstream-published SHA256. To bump S6_OVERLAY_VERSION, fetch the per-arch
# `.sha256` files from the corresponding release and update the ARGs. The
# checksum lookup happens during build, so a compromised release artifact
# fails the build loudly instead of silently producing a tampered image.
ARG S6_OVERLAY_VERSION=3.2.3.0
ARG S6_OVERLAY_NOARCH_SHA256=b720f9d9340efc8bb07528b9743813c836e4b02f8693d90241f047998b4c53cf
ARG S6_OVERLAY_X86_64_SHA256=a93f02882c6ed46b21e7adb5c0add86154f01236c93cd82c7d682722e8840563
ARG S6_OVERLAY_AARCH64_SHA256=0952056ff913482163cc30e35b2e944b507ba1025d78f5becbb89367bf344581
# riscv64: s6-overlay publishes a first-class riscv64 tarball.
ARG S6_OVERLAY_RISCV64_SHA256=a4a4ed5eb17562879d07189cc4b3b9cd146c2d88855792fa95d99e0decbfcaf8
ARG S6_OVERLAY_SYMLINKS_SHA256=a60dc5235de3ecbcf874b9c1f18d73263ab99b289b9329aa950e8729c4789f0e
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp/
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz /tmp/
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) s6_arch="x86_64"; s6_arch_sha="${S6_OVERLAY_X86_64_SHA256}" ;; \
        arm64) s6_arch="aarch64"; s6_arch_sha="${S6_OVERLAY_AARCH64_SHA256}" ;; \
        riscv64) s6_arch="riscv64"; s6_arch_sha="${S6_OVERLAY_RISCV64_SHA256}" ;; \
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
    rm /tmp/s6-overlay-*.tar.xz /tmp/s6-overlay.sha256; \
    # #34192: backward-compat shim for orchestration templates that still\
    # reference the legacy /usr/bin/tini entrypoint (e.g. Hostinger's\
    # 'Hermes WebUI' catalog). The image has moved to s6-overlay /init\
    # as PID 1 (see ENTRYPOINT below + the migration comment at the top\
    # of this file), but external wrappers pinned to /usr/bin/tini will\
    # crash with 'tini: No such file or directory' on startup. The shim\
    # symlinks /usr/bin/tini -> /init so legacy wrappers exec the right\
    # PID-1 reaper without behavior change for users on the current\
    # ENTRYPOINT. Safe to drop once the affected catalogs are updated.\
    ln -sf /init /usr/bin/tini

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# Node 22 LTS: copy the node binary plus the bundled npm + corepack JS
# installs from the (per-arch) node_source stage.  npm and npx are recreated as
# symlinks because they're symlinks in the source image (and need to live on
# PATH).  See the node_source stages at the top of the file for the
# version-bump rationale (#4977).
COPY --chmod=0755 --from=node_source /usr/local/bin/node /usr/local/bin/
COPY --from=node_source /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=node_source /usr/local/lib/node_modules/corepack /usr/local/lib/node_modules/corepack
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    ln -sf /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

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

# amd64/arm64: `npm install` + bundle Playwright Chromium (browser tools).
# riscv64: `npm install` crashes reconciling optional deps that have no riscv64
# entry in the lockfile (npm 10.9.x: "Cannot read properties of undefined
# (reading 'os')"), so use `npm ci`, which installs straight from the lockfile.
# Playwright has no riscv64 Chromium, so it is skipped (browser-automation
# tools are unavailable on riscv64).
RUN set -eu; \
    if [ "${TARGETARCH:-amd64}" = "riscv64" ]; then \
        npm ci --no-audit; \
    else \
        npm install --prefer-offline --no-audit; \
        npx playwright install --with-deps chromium --only-shell; \
    fi; \
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
# riscv64 note: with no manylinux riscv64 wheels, every native extension
# compiles from sdist here against the rustup toolchain installed above, so
# this is the long pole on riscv64 (tens of minutes). riscv64 enumerates
# `[all]` MINUS `[dev]` (ty + ruff, two large dev-only Rust workspaces) to skip
# a needless long compile; amd64/arm64 keep `--extra all`. libolm-dev is
# installed above for every arch, so `--extra matrix` builds on riscv64 too and
# stays in dependency-set parity with amd64/arm64.
#
# The editable link is created after the source copy below.
COPY pyproject.toml uv.lock ./
RUN touch ./README.md
RUN set -eu; \
    if [ "${TARGETARCH:-amd64}" = "riscv64" ]; then \
        uv sync --frozen --no-install-project \
            --extra cron --extra cli --extra pty --extra mcp --extra homeassistant \
            --extra sms --extra acp --extra google --extra web --extra youtube \
            --extra messaging --extra anthropic --extra bedrock --extra azure-identity \
            --extra hindsight --extra matrix; \
    else \
        uv sync --frozen --no-install-project --extra all --extra messaging --extra anthropic --extra bedrock --extra azure-identity --extra hindsight --extra matrix; \
    fi

# ---------- Frontend build (cached independently from Python source) ----------
# Copy only the frontend source trees first so that Python-only changes don't
# invalidate the (relatively slow) web + ui-tui build layer.
COPY web/ web/
COPY ui-tui/ ui-tui/
COPY apps/shared/ apps/shared/
# riscv64: the web dashboard CANNOT be built in-image — Tailwind v4's Vite
# pipeline pulls lightningcss + @tailwindcss/oxide, Rust/napi modules with no
# riscv64 binary on npm, and there is no riscv64 `node` base image to build them
# under. Its output (hermes_cli/web_dist) is arch-independent, so CI cross-builds
# it on the host and injects it via the build context; the `COPY --link ... . .`
# below carries it into the image. Without it the riscv64 image ships
# dashboard-less (the dashboard service is gated off by default). See docker.yml.
# ui-tui (esbuild) builds on every arch.
RUN set -eu; \
    if [ "${TARGETARCH:-amd64}" != "riscv64" ]; then \
        ( cd web && npm run build ); \
    fi; \
    cd ui-tui && npm run build

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
