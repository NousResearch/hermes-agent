FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    dnsutils \
    git \
    iproute2 \
    iptables \
    iputils-ping \
    jq \
    nano \
    net-tools \
    netcat-openbsd \
    nodejs \
    npm \
    openssh-client \
    openssh-sftp-server \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    ripgrep \
    rsync \
    sudo \
    traceroute \
    vim \
  && rm -rf /var/lib/apt/lists/*

# OpenShell's default static process policy runs as sandbox:sandbox.
RUN groupadd -r supervisor \
  && useradd -r -g supervisor -s /usr/sbin/nologin supervisor \
  && groupadd -r sandbox \
  && useradd -r -g sandbox -d /sandbox -s /bin/bash sandbox \
  && mkdir -p /sandbox /agent/.hermes/skills /agent/.hermes/credentials /agent/.hermes/cache \
  && chown -R sandbox:sandbox /sandbox /agent
WORKDIR /sandbox

COPY start.sh /usr/local/bin/hermes-openshell-start
RUN chmod +x /usr/local/bin/hermes-openshell-start

USER sandbox
ENTRYPOINT ["/bin/bash"]
