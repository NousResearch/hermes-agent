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
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        git curl ripgrep ffmpeg procps openssh-client docker-cli tini \
        # Libraries required by Playwright’s headless Chromium
        libnss3 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
        libgbm1 libasound2 &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/locale

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data -s /bin/bash hermes

# Copy Node.js, uv, and gosu from upstream images instead of installing them in-place.
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node_runtime /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=uv_runtime /uv /usr/local/bin/uv
COPY --from=uv_runtime /uvx /usr/local/bin/uvx
COPY --from=gosu_runtime /gosu /usr/local/bin/

# Link npm and npx CLI scripts to /usr/local/bin
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm &&\
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Set working directory
WORKDIR /opt/hermes

# Install top-level Node.js dependencies.
COPY package*.json ./
RUN npm ci --no-audit &&\
    npm cache clean --force &&\
    rm -rf /tmp/* ~/.npm

# Install the Playwright Chromium shell browser.
RUN npx playwright install --with-deps chromium --only-shell &&\
    rm -rf /tmp/* ~/.npm /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/locale &&\
    npm cache clean --force

# Install WhatsApp bridge dependencies.
RUN mkdir -p /opt/hermes/scripts/whatsapp-bridge /opt/hermes/web/
COPY ./scripts/whatsapp-bridge/package*.json ./scripts/whatsapp-bridge/
RUN cd /opt/hermes/scripts/whatsapp-bridge &&\
    npm ci --no-audit &&\
    npm cache clean --force &&\
    rm -rf /tmp/* ~/.npm

COPY web/package*.json /opt/hermes/web/
RUN cd /opt/hermes/web &&\
    npm ci --no-audit &&\
    npm run build &&\
    npm cache clean --force &&\
    rm -rf /tmp/* ~/.npm /opt/hermes/web/node_modules

# Hand ownership to hermes user, then install Python deps in a virtualenv
RUN chown -R hermes:hermes /opt/hermes
USER hermes

# Create Python virtual environment
ARG EXTRAS="modal,daytona,messaging,cron,cli,tts-premium,slack,pty,honcho,mcp,homeassistant,sms,acp,voice,dingtalk,feishu,google,mistral,bedrock,web"
ENV VIRTUAL_ENV=/opt/hermes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the application source as `hermes` and install Hermes into the virtual environment.
COPY . .
RUN uv venv "$VIRTUAL_ENV" &&\
    uv pip install --no-cache -e ".[$EXTRAS]"

# ---------- Permissions ----------
# Make install dir world-readable so any HERMES_UID can read it at runtime.
# The venv needs to be traversable too.
USER root
RUN chmod -R a+rX /opt/hermes

# ---------- Runtime ----------
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:${PATH}"
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh" ]
