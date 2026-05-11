FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget jq build-essential gcc g++ make \
    openssh-client ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install gh CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (no sudo access)
RUN useradd -m -u 1000 -s /bin/bash agent
RUN useradd -m -u 1001 -s /usr/sbin/nologin broker

# Create workspace root
RUN mkdir -p /workspaces && chown agent:agent /workspaces

# Create directory for hermes-agent clone (populated externally or at first boot)
RUN mkdir -p /opt/hermes-agent && chown agent:agent /opt/hermes-agent

# Git config for the agent user — set at SYSTEM level (/etc/gitconfig)
# because /home is mounted as tmpfs at runtime, wiping per-user configs.
RUN git config --system user.name "daimon[bot]" \
    && git config --system user.email "daimon[bot]@nousresearch.com" \
    && git config --system init.defaultBranch main

# Default working directory
WORKDIR /workspaces

FROM base AS agent

COPY gh_client.py /usr/local/bin/gh-client
RUN chmod 755 /usr/local/bin/gh-client \
    && rm -f /usr/bin/gh \
    && ln -s /usr/local/bin/gh-client /usr/bin/gh \
    && git config --system --unset-all credential.helper || true

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
USER agent

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]

FROM base AS broker

RUN mkdir -p /run/secrets
COPY gh_broker.py /usr/local/bin/gh-broker
COPY broker-entrypoint.sh /broker-entrypoint.sh
RUN chmod 755 /usr/local/bin/gh-broker /broker-entrypoint.sh

ENTRYPOINT ["/broker-entrypoint.sh"]
