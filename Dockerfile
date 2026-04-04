FROM ghcr.io/astral-sh/uv:trixie-slim

WORKDIR /opt/hermes

# Install system packages required
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        git curl nodejs npm ripgrep ffmpeg gcc python3 python3-pip systemctl &&\
    rm -rf /var/lib/apt/lists/* &&\
    npm cache clean --force

# Install root Node.js dependencies.
COPY package*.json ./
RUN npm ci --omit=dev --no-audit &&\
    npm cache clean --force

# Install the shell chromium browser.
RUN npx playwright install chromium --with-deps --only-shell &&\
    rm -rf /var/lib/apt/lists/* &&\
    npm cache clean --force

# Install WhatsApp bridge dependencies.
RUN mkdir -p /opt/hermes/scripts/whatsapp-bridge 
COPY ./scripts/whatsapp-bridge/package*.json ./scripts/whatsapp-bridge
RUN cd /opt/hermes/scripts/whatsapp-bridge &&\
    npm ci --no-audit &&\
    npm cache clean --force

# Create and activate the Python virtual environment.
ENV VIRTUAL_ENV=/opt/hermes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV

# Configure optional Python extras to install.
ARG EXTRAS="messaging,cron,cli,modal,tts-premium,voice,pty,honcho,mcp,homeassistant,acp,slack"

# Copy the application source and install Hermes.
COPY . .
RUN uv pip install -e ".[$EXTRAS]" --no-cache &&\
    chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
