FROM debian:13.4

# Install system dependencies in one layer, including chromium dependencies, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 python3-pip python3-venv ripgrep ffmpeg gcc python3-dev libffi-dev git \
	libasound2t64 libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 libcairo2 libcups2t64 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0t64 libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 xvfb fonts-noto-color-emoji fonts-unifont libfontconfig1 libfreetype6 xfonts-scalable fonts-liberation fonts-ipafont-gothic fonts-wqy-zenhei fonts-tlwg-loma-otf fonts-freefont-ttf && \
    rm -rf /var/lib/apt/lists/*
 
# user IDs over 10000 are recommended for security; this can be overridden at runtime for easier permissions handling (running under the same UID as the computer user, for instance)
RUN useradd -u 10000 -m -d /opt/data hermes

USER hermes
COPY --chown=hermes:hermes . /opt/hermes
WORKDIR /opt/hermes

# Install Python and Node dependencies in one layer, no cache
RUN python3 -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -e ".[all]" && \
    .venv/bin/pip install --no-cache-dir -e ".[matrix]" && \
    npm install --prefer-offline --no-audit && \
    npx playwright install chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force
 
RUN chmod +x /opt/hermes/docker/entrypoint.sh
 
ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
