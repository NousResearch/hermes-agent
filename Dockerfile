FROM debian:13.4

# Install system dependencies in one layer, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 python3-pip ripgrep ffmpeg gcc python3-dev libffi-dev gosu && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the agent
RUN groupadd -r hermes && useradd -r -g hermes -d /opt/data hermes

COPY . /opt/hermes
WORKDIR /opt/hermes

# Install Python and Node dependencies in one layer, no cache
RUN pip install --no-cache-dir -e ".[all]" --break-system-packages && \
    npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force

WORKDIR /opt/hermes
RUN chmod +x /opt/hermes/docker/entrypoint.sh && \
    chown -R hermes:hermes /opt/hermes

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
# Entrypoint runs as root to bootstrap /opt/data, then drops to hermes via gosu
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
