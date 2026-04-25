# ============================================================
# STAGE 1: Build / Compile (heavy deps, discarded after build)
# ============================================================
FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM tianon/gosu:1.19-trixie@sha256:3b176695959c71e123eb390d427efc665eeb561b1540e82679c15e992006b8b9 AS gosu_source

FROM debian:13.4 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev procps git openssh-client docker-cli tini && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=gosu_source /gosu /usr/local/bin/
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /opt/hermes

# Layer-cached dependency install
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/

RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    (cd web && npm install --prefer-offline --no-audit) && \
    npm cache clean --force

# Source code
COPY --chown=hermes:hermes . .

# Build web dashboard
RUN cd web && npm run build

# Python virtualenv
RUN uv venv && \
    uv pip install --no-cache-dir -e ".[all]"

# Strip caches before final image
RUN rm -rf /root/.cache/uv /root/.cache/pip /root/.npm /root/.node-gyp \
    && find /opt/hermes -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# STAGE 2: Runtime (slim, no build deps)
# ============================================================
FROM debian:13.4-slim

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:/opt/hermes/.venv/bin:/usr/local/bin:$PATH"

# Runtime libs only (no build-essential, no gcc, no python3-dev)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nodejs tini procps \
        libasound2t64 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libcairo2 \
        libcups2 libdbus-1-3 libdrm2 libexpat1 libfontconfig1 libfreetype6 \
        libgbm1 libglib2.0-0t64 libgtk-3-0 libnss3 libpango-1.0-0 \
        libpangocairo-1.0-0 libpulse0 libxcomposite1 libxdamage1 libxfixes3 \
        libxkbcommon0 libxrandr2 libxtst6 libx11-6 libxcb1 libxext6 libxi6 \
        libwayland-client0 libvulkan1 libosmesa6 \
        openssh-client docker-cli git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=gosu_source /gosu /usr/local/bin/

# Copy only built artifacts from builder
COPY --from=builder --chown=root:root /opt/hermes /opt/hermes
RUN chmod -R a+rX /opt/hermes

VOLUME ["/opt/data"]
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh"]
