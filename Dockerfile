# ---------- CN mirror configuration (override with --build-arg to disable) ----------
# Registry mirrors (ghcr.io not reachable in CN; docker.m.daocloud.io preserves
# upstream digests so sha256 pinning still validates integrity).
ARG GHCR_MIRROR=ghcr.m.daocloud.io
ARG DOCKERHUB_MIRROR=docker.m.daocloud.io
# Package mirrors
ARG APT_MIRROR=mirrors.aliyun.com
ARG NPM_REGISTRY=https://registry.npmmirror.com
ARG PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com

FROM ${GHCR_MIRROR}/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM ${DOCKERHUB_MIRROR}/library/debian:13.4

# Re-declare ARGs that need to be visible in this stage (ARGs before FROM are
# only available to the FROM line itself).
ARG APT_MIRROR
ARG NPM_REGISTRY
ARG PLAYWRIGHT_DOWNLOAD_HOST
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Store Playwright browsers outside the volume mount so the build-time
# install survives the /opt/data volume overlay at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# Swap Debian sources to the CN mirror (trixie uses deb822-style sources).
# If APT_MIRROR is set to the upstream deb.debian.org this is a no-op.
RUN set -eux; \
    if [ "${APT_MIRROR}" != "deb.debian.org" ] && [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
            /etc/apt/sources.list.d/debian.sources; \
    fi

# Install system dependencies in one layer, clear APT cache.
# gosu is installed from Debian (1.17 in trixie) instead of the tianon/gosu
# image because CN registry mirrors (daocloud) do not whitelist that repo.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev procps git gosu openssh-client docker-cli && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /opt/hermes

# Configure npm + pip globally so any `npm install` / `pip install` picks up
# the CN mirrors automatically (also inherited by `npx` and uv's pip).
RUN npm config set registry "${NPM_REGISTRY}" && \
    npm config set fetch-timeout 600000 && \
    npm config set fetch-retries 5 && \
    mkdir -p /etc/pip && \
    printf '[global]\nindex-url = %s\ntrusted-host = %s\ntimeout = 120\n' \
        "${PIP_INDEX_URL}" "${PIP_TRUSTED_HOST}" > /etc/pip.conf

# ---------- Layer-cached dependency install ----------
# Copy only package manifests first so npm install + Playwright are cached
# unless the lockfiles themselves change.
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/

# Optional proxy for playwright only (npmmirror does not mirror the newer
# `builds/cft/` path used by chrome-headless-shell, so we fall back to upstream
# playwright.azureedge.net via the configured proxy when PLAYWRIGHT_PROXY is set).
ARG PLAYWRIGHT_PROXY=""

# npm install uses npmmirror (fast in CN). Skip playwright's postinstall browser
# download here; we do it in a separate RUN below so we can scope the proxy.
RUN export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 && \
    npm install --prefer-offline --no-audit && \
    (cd web && npm install --prefer-offline --no-audit)

# Playwright chromium-headless-shell falls back to upstream via PLAYWRIGHT_PROXY
# (npmmirror's mirror does not cover the newer builds/cft/ paths). Note: legacy
# docker builder exports ARGs as env vars, so we explicitly unset the ARG-leaked
# PLAYWRIGHT_DOWNLOAD_HOST before invoking playwright.
RUN unset PLAYWRIGHT_DOWNLOAD_HOST && \
    HTTPS_PROXY="${PLAYWRIGHT_PROXY}" HTTP_PROXY="${PLAYWRIGHT_PROXY}" \
    npx playwright install --with-deps chromium --only-shell && \
    npm cache clean --force

# ---------- Source code ----------
# .dockerignore excludes node_modules, so the installs above survive.
COPY --chown=hermes:hermes . .

# Build web dashboard (Vite outputs to hermes_cli/web_dist/)
RUN cd web && npm run build

# ---------- Python virtualenv ----------
RUN chown hermes:hermes /opt/hermes
USER hermes
ENV UV_INDEX_URL=${PIP_INDEX_URL}
RUN uv venv && \
    uv pip install --no-config --no-cache-dir -e ".[all]"

# ---------- Runtime ----------
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:${PATH}"
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
