# Terminal-backend sandbox image WITH a desktop: the default sandbox base every
# docker/modal/daytona/singularity user already runs (nikolaik/python-nodejs), plus
# the tools that base was missing, plus the same display stack the -desktop Hermes
# image carries (TigerVNC + Xfce components + headed Chromium) and cua-driver, so
# Bot Screen, computer_use and the browser can live INSIDE the sandbox instead of
# on the gateway host. No Hermes runtime in here; the gateway shells in.
#
#   docker build -f docker/sandbox-desktop.Dockerfile -t nousresearch/hermes-sandbox:desktop .
#
# Published as nousresearch/hermes-sandbox:desktop by .github/workflows/sandbox-image.yml
# on releases and manual dispatch only: it carries no Hermes code, so it does not track main.
# The tag lives in the ARG so CI and a local build read one place; hadolint cannot
# see through the substitution, hence the inline ignore.
ARG SANDBOX_BASE=nikolaik/python-nodejs:python3.13-nodejs26
# hadolint ignore=DL3006
FROM ${SANDBOX_BASE}
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG CUA_DRIVER_VERSION=0.28.2
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    LANG=C.UTF-8

# Everyday tools the nikolaik base lacks (checked 2026-09: no jq, rg, fd, tmux,
# less, vim/nano, zip, rsync, tree, procps beyond ps). Kept to what agents reach
# for from a shell; language toolchains come from the base.
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        jq ripgrep fd-find tmux less nano vim-tiny zip rsync tree procps htop \
        file bsdextrautils ca-certificates locales sudo && \
    ln -sf /usr/bin/fdfind /usr/local/bin/fd && \
    rm -rf /var/lib/apt/lists/*

# Display stack: identical package set to the Hermes -desktop image (Dockerfile,
# HERMES_BOT_DESKTOP=1) so tools/bot_desktop/launcher.sh finds the same binaries.
# Components are launched individually by the launcher, never xfce4-session.
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        tigervnc-standalone-server xfce4-panel xfwm4 xfdesktop4 xfce4-settings xfce4-terminal \
        dbus-x11 x11-xserver-utils x11-utils x11-xkb-utils xauth fonts-dejavu-core \
        at-spi2-core libgtk-3-0 libnss3 libasound2 libxss1 && \
    rm -rf /var/lib/apt/lists/*

# Headed Chromium through Playwright (the same build the Hermes -desktop image
# uses), so agent-browser and the dock's Browser icon share one binary and one
# --user-data-dir. --with-deps pulls the Chromium runtime libraries. agent-browser
# itself is baked (the CLI the browser tools drive), pinned to the same range the
# gateway resolves (tools/browser_tool.py AGENT_BROWSER_NPX_SPEC), scripts off:
# the first browser_navigate in a fresh sandbox must not wait on an npm fetch.
RUN for i in 1 2 3; do \
        npx --yes playwright@1 install --with-deps chromium && break || \
        { [ "$i" = 3 ] && exit 1; echo "playwright chromium install failed (attempt $i); retrying in 10s"; sleep 10; }; \
    done && chmod -R a+rX /opt/playwright && \
    npm install -g --ignore-scripts --no-audit --fetch-retries=5 "agent-browser@^0.26.0" && \
    agent-browser --version

# cua-driver: computer_use's MCP driver. Pinned release tarball from the
# cua-driver-rs-v* tags (never releases/latest: prereleases publish 0 assets).
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) arch=x86_64 ;; \
        arm64) arch=arm64 ;; \
        *) echo "unsupported TARGETARCH ${TARGETARCH}"; exit 1 ;; \
    esac; \
    mkdir -p /opt/cua-driver; \
    curl -fsSL "https://github.com/trycua/cua/releases/download/cua-driver-rs-v${CUA_DRIVER_VERSION}/cua-driver-rs-${CUA_DRIVER_VERSION}-linux-${arch}-binary.tar.gz" \
        | tar -xz -C /opt/cua-driver; \
    ln -sf /opt/cua-driver/cua-driver /usr/local/bin/cua-driver; \
    /usr/local/bin/cua-driver --version

# The base's default user stays root, exactly like nikolaik today, so existing
# docker_image users see no ownership or PATH change. Desktop processes (Xvnc,
# Xfce, Chromium, cua-driver) are exec'd as the base's unprivileged `pn` (uid
# 1000): Chromium refuses to run as root without --no-sandbox and cua-driver's
# AT-SPI bus wants a real user session. `pn` may sudo for ad-hoc installs.
RUN echo "pn ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/pn && chmod 0440 /etc/sudoers.d/pn && \
    mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix && \
    mkdir -p /tmp/hermes-runtime && chown pn:pn /tmp/hermes-runtime && chmod 0700 /tmp/hermes-runtime

# Runtime dir for dbus/Xvnc; containers have no logind to create it. Both /tmp
# paths above are fixed by the X11 protocol / seeded per container, never shared.
ENV XDG_RUNTIME_DIR=/tmp/hermes-runtime
CMD ["sleep", "infinity"]
