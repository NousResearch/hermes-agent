FROM node:lts-trixie-slim@sha256:18571e15a9668340dc5a48eefd93dddbbc8bee227cb68acf2b7158729130b4bf AS node_runtime
FROM ghcr.io/astral-sh/uv:latest@sha256:90bbb3c16635e9627f49eec6539f956d70746c409209041800a0280b93152823 AS uv_runtime
FROM python:slim-trixie@sha256:fb83750094b46fd6b8adaa80f66e2302ecbe45d513f6cece637a841e1025b4ca

WORKDIR /opt/hermes

# Install system packages required
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        git curl ripgrep ffmpeg systemctl &&\
    rm -rf /var/lib/apt/lists/*

# Copy Node.js and uv from upstream images instead of installing them with apt/curl.
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node_runtime /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm &&\
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY --from=uv_runtime /uv /usr/local/bin/uv
COPY --from=uv_runtime /uvx /usr/local/bin/uvx

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
