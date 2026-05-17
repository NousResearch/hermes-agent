ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.13-trixie
FROM ${BUILD_FROM}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl nodejs npm python3-pip python3-dev \
    ripgrep ffmpeg gcc libffi-dev procps git openssh-client docker-cli tini && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist

WORKDIR /opt/hermes

# Layer-cached dependency install
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

# Python dependencies
COPY pyproject.toml uv.lock ./
RUN touch ./README.md
RUN uv sync --frozen --no-install-project --extra all

# Source code
COPY --chown=root:root . .

# Build assets
RUN cd web && npm run build && \
    cd ../ui-tui && npm run build

# Link hermes-agent
RUN uv pip install --no-cache-dir --no-deps -e "."

# Runtime PATH to include .venv
ENV PATH="/opt/hermes/.venv/bin:${PATH}"

# Copy add-on files
COPY hassio/hermes/run.sh /
RUN chmod a+x /run.sh

ARG BUILD_ARCH
LABEL \
    io.hass.name="Hermes Agent" \
    io.hass.description="The self-improving AI agent by Nous Research" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="app" \
    io.hass.version="0.13.0"

CMD [ "/run.sh" ]
