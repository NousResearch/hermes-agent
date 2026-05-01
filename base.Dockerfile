FROM debian:13.4

# ---------- Mirror Configuration ----------
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources
ENV npm_config_registry=https://registry.npmmirror.com

# Store Playwright browsers outside the volume mount
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
        -o Acquire::Retries=3 \
        -o Acquire::http::Timeout=30 \
        build-essential nodejs npm python3 git || \
    (apt-get update && apt-get install -f -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 git) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hermes

# Copy only package manifests
COPY package.json package-lock.json ./

# Install npm dependencies and browsers in one layer
RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    npm cache clean --force

# agent-browser: install Chrome and move to persistent path
RUN npx agent-browser install && \
    rm -rf /root/.cache && \
    mv /root/.agent-browser /opt/hermes/.agent-browser
