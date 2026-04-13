FROM debian:13.4

# Install system dependencies in one layer, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 python3-pip python3-venv ripgrep ffmpeg gcc python3-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/hermes
WORKDIR /opt/hermes

# F-019: install Python deps into an isolated venv rather than the system
# site-packages. --break-system-packages on the system interpreter mingled
# PEP 668-managed Debian packages with pip-resolved ones; a venv removes
# that risk. The venv is on PATH so `hermes`, `python`, `pip` all resolve
# inside it for the rest of the build and at runtime.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python and Node dependencies in one layer, no cache
RUN pip install --no-cache-dir -e ".[all]" && \
    npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force

WORKDIR /opt/hermes
RUN chmod +x /opt/hermes/docker/entrypoint.sh

# Run as non-root user for defense-in-depth
RUN groupadd -r hermes && useradd -r -g hermes -d /opt/data -s /bin/bash hermes && \
    mkdir -p /opt/data && chown -R hermes:hermes /opt/data /opt/hermes /opt/venv

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
USER hermes
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
