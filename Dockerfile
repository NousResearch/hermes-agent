FROM node:lts-trixie-slim@sha256:18571e15a9668340dc5a48eefd93dddbbc8bee227cb68acf2b7158729130b4bf AS node_runtime
FROM ghcr.io/astral-sh/uv:latest@sha256:90bbb3c16635e9627f49eec6539f956d70746c409209041800a0280b93152823 AS uv_runtime
FROM tianon/gosu:1.19-trixie@sha256:3b176695959c71e123eb390d427efc665eeb561b1540e82679c15e992006b8b9 AS gosu_runtime
FROM python:slim-trixie@sha256:fb83750094b46fd6b8adaa80f66e2302ecbe45d513f6cece637a841e1025b4ca

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Store Playwright browsers outside the volume mount so the build-time
# install survives the /opt/data volume overlay at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# Install system dependencies in one layer, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl ripgrep ffmpeg procps systemctl &&\
    rm -rf /var/lib/apt/lists/*

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data -s /bin/bash hermes

# Copy Node.js, uv, and gosu from upstream images instead of installing them in-place.
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node_runtime /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm &&\
ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY --from=uv_runtime /uv /usr/local/bin/uv
COPY --from=uv_runtime /uvx /usr/local/bin/uvx
COPY --from=gosu_runtime /gosu /usr/local/bin/

WORKDIR /opt/hermes

# Install top-level Node.js dependencies.
COPY package*.json ./
RUN npm ci --no-audit &&\
    npm cache clean --force &&\
    rm -rf /tmp/*

# Install the Playwright Chromium shell browser.
RUN npx playwright install chromium --with-deps --only-shell &&\
    rm -rf /var/lib/apt/lists/* &&\
    npm cache clean --force

# Install WhatsApp bridge dependencies.
RUN mkdir -p /opt/hermes/scripts/whatsapp-bridge 
COPY ./scripts/whatsapp-bridge/package*.json ./scripts/whatsapp-bridge
RUN cd /opt/hermes/scripts/whatsapp-bridge &&\
    npm ci --no-audit &&\
    npm cache clean --force

# Hand ownership to hermes user, then install Python deps in a virtualenv
RUN chown -R hermes:hermes /opt/hermes
USER hermes

# Configure the optional Python extras to install.
ARG EXTRAS="messaging,cron,cli,modal,tts-premium,voice,pty,honcho,mcp,homeassistant,acp,slack"
ENV VIRTUAL_ENV=/opt/hermes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Copy the application source as `hermes` and install Hermes into the virtual environment.
COPY . .
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache -e ".[$EXTRAS]"

USER root
RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
