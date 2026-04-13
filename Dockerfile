FROM python:3.12-slim

# Install system dependencies — no playwright, no nodejs, no ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        ripgrep \
        curl \
        git \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install filebrowser binary (static Go binary — no runtime deps)
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
        amd64) FB_ARCH="linux-amd64" ;; \
        arm64) FB_ARCH="linux-arm64" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/filebrowser/filebrowser/releases/download/v2.32.0/${FB_ARCH}-filebrowser.tar.gz" \
    | tar -xz -C /usr/local/bin filebrowser && \
    chmod +x /usr/local/bin/filebrowser

COPY . /opt/hermes
WORKDIR /opt/hermes

# Install hermes with only the extras needed for this deployment:
#   a2a — A2A SDK + uvicorn server
#   mcp — MCP tool support
# Playwright, nodejs, ffmpeg, homeassistant skipped to keep image lean (<1GB)
RUN pip install --no-cache-dir -e ".[a2a,mcp]"

# Supervisor config — manages hermes gateway + A2A server + filebrowser
COPY docker/supervisord.conf /etc/supervisor/conf.d/hermes.conf

# Entrypoint
RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME ["/opt/data"]

# hermes API (OpenAI-compatible) | A2A server | filebrowser UI
EXPOSE 8642 9000 8080

ENTRYPOINT ["/opt/hermes/docker/entrypoint.sh"]
