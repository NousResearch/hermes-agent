# Dedicated local Gitea Actions job image for crypto_bot CI.
#
# The act_runner container is separate from this image. The runner advertises
# ubuntu-latest as docker://crypto-bot-ci-runner:python313-node20-go so JS
# actions such as actions/checkout can find Node on PATH inside each job.
FROM node:20-bookworm-slim AS node20

FROM python:3.13-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=node20 /usr/local/bin/node /usr/local/bin/node
COPY --from=node20 /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN set -eux; \
    ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm; \
    ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      golang-go \
      jq \
      openssh-client \
      ripgrep \
      rsync \
      unzip \
      xz-utils; \
    rm -rf /var/lib/apt/lists/*; \
    python_full_version="$(python -c 'import platform; print(platform.python_version())')"; \
    python_toolcache="/opt/hostedtoolcache/Python/${python_full_version}/arm64"; \
    mkdir -p "${python_toolcache}"; \
    cp -a /usr/local/. "${python_toolcache}/"; \
    touch "/opt/hostedtoolcache/Python/${python_full_version}/arm64.complete"; \
    python --version; \
    "${python_toolcache}/bin/python" --version; \
    node --version; \
    npm --version; \
    git --version; \
    rg --version; \
    go version

ENV PATH="/usr/local/bin:${PATH}" \
    AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache \
    RUNNER_TOOL_CACHE=/opt/hostedtoolcache \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
