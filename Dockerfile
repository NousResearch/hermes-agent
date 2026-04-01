FROM astral/uv:trixie-slim

WORKDIR /opt/hermes

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        build-essential git curl nodejs npm ripgrep ffmpeg gcc python3 python3-pip python3-dev libffi-dev &&\
    npx playwright install chromium --with-deps --only-shell &&\
    rm -rf /var/lib/apt/lists/* &&\
    npm cache clean --force

COPY package*.json ./
RUN npm ci --omit=dev --no-audit &&\
    npm cache clean --force

RUN mkdir -p /opt/hermes/scripts/whatsapp-bridge 
COPY ./scripts/whatsapp-bridge/package*.json ./scripts/whatsapp-bridge
RUN cd /opt/hermes/scripts/whatsapp-bridge &&\
    npm ci --no-audit &&\
    npm cache clean --force

COPY . .
RUN uv pip install -e ".[all]" --no-cache --system --break-system-packages && \
    chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]
