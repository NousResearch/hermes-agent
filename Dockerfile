# Dockerfile for Hermes Agent on Railway
# Uses minimax-oauth fix branch

FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM tianon/gosu:1.19-trixie@sha256:3b176695959c71e123eb390d427efc665eeb561b1540e82679c15e992006b8b9 AS gosu_source
FROM debian:13.4

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# Install all system deps in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev \
    procps git openssh-client docker-cli tini && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=gosu_source /gosu /usr/local/bin/
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /opt/hermes

# npm dependencies (cached layer)
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/
COPY ui-tui/package.json ui-tui/package-lock.json ui-tui/
COPY ui-tui/packages/hermes-ink/ ui-tui/packages/hermes-ink/
ENV npm_config_install_links=false

RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    (cd web && npm install --prefer-offline --no-audit) && \
    (cd ui-tui && npm install --prefer-offline --no-audit) && \
    npm cache clean --force

# Python deps (cached layer)
COPY pyproject.toml uv.lock ./
RUN touch ./README.md
RUN uv sync --frozen --no-install-project --extra all

# Source + build
COPY --chown=hermes:hermes . .
RUN cd web && npm run build && cd ../ui-tui && npm run build

# Permissions
USER root
RUN chmod -R a+rX /opt/hermes && \
    chown -R hermes:hermes /opt/hermes/ui-tui /opt/hermes/node_modules

# Editable install
RUN uv pip install --no-cache-dir --no-deps -e "."

# Environment
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:${PATH}"
ENV HERMES_UID=10000
ENV HERMES_GID=10000

# Custom entrypoint for Railway (no VOLUME - Railway handles persistence via dashboard)
COPY --chmod=0755 docker/entrypoint.railway.sh /usr/local/bin/hermes-entrypoint
ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/usr/local/bin/hermes-entrypoint" ]
CMD [ "gateway", "run" ]