FROM astral-sh/uv:python3.14-trixie-slim

# Install system dependencies in one layer, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm ripgrep ffmpeg gcc python3-dev libffi-dev &&\
    rm -rf /var/lib/apt/lists/*

COPY . /opt/hermes
WORKDIR /opt/hermes

# Install Python and Node dependencies in one layer, no cache
RUN uv pip install --no-cache -e ".[all]" --system && \
    npm ci --omit=dev --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm ci --prefer-offline --no-audit && \
    npm cache clean --force && \
    rm -rf /var/lib/apt/lists/*

RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
