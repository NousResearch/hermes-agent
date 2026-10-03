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
# see through the substitution, hence the inline ignore. The default is pinned to
# the multi-arch manifest digest so the base cannot drift under a rebuild: a tag
# alone resolves different content over time (the same discipline the runtime
# Dockerfile applies to debian:13.4). To move bases, re-resolve the index digest
# (docker buildx imagetools inspect) and paste it here.
ARG SANDBOX_BASE=nikolaik/python-nodejs:python3.13-nodejs26@sha256:2607a06ae1c2dc15d74329d913486e75df2c93584546f7c5a88547a597bd1642
# hadolint ignore=DL3006
FROM ${SANDBOX_BASE}
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG CUA_DRIVER_VERSION=0.28.2
# sha256 of the release asset cua-driver-rs-<version>-linux-<arch>-binary.tar.gz,
# re-verified on every build before extraction. To bump the driver, re-hash both
# assets (curl <url> | sha256sum) and paste them here.
ARG CUA_DRIVER_SHA256_x86_64=a1d99fd04bb4927ef5ffdbe60eb91ed8b51a2bab60e10fc604a75bd59ce69c3e
ARG CUA_DRIVER_SHA256_arm64=55e8a32839a4ac369a773df4dac87b345bd4567779221ade4a5e39223a45a2e8
# Exact npm package tarballs and SHA-512 integrity values, re-pinned on bump.
# npm otherwise resolves the registry response at build time, so identical
# Dockerfile inputs would not necessarily produce identical images.
ARG PLAYWRIGHT_VERSION=1.63.0
ARG AGENT_BROWSER_VERSION=0.26.0
ARG PLAYWRIGHT_SHA512=fbbce204b89d4b835a34276de7b4940d3fb062699de5f9a241e8d489cfe46fe62c61208fc8e384f6c79bccc8cd990aec9cda4326a778587bd5ffc95f29f50352
ARG AGENT_BROWSER_SHA512=a5da927e3c1b152a7eaa7c256f6836ddef705ef78839f322d7dc693c0f716546f3100529e96e180598fa61b8fccf833b5a471b1830d873ea032070edf51dc40d
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
RUN mkdir -p /tmp/npm-downloads; \
    curl -fsSL --retry 3 -o /tmp/npm-downloads/playwright.tgz \
        "https://registry.npmjs.org/playwright/-/playwright-${PLAYWRIGHT_VERSION}.tgz"; \
    echo "${PLAYWRIGHT_SHA512}  /tmp/npm-downloads/playwright.tgz" | sha512sum -c -; \
    curl -fsSL --retry 3 -o /tmp/npm-downloads/agent-browser.tgz \
        "https://registry.npmjs.org/agent-browser/-/agent-browser-${AGENT_BROWSER_VERSION}.tgz"; \
    echo "${AGENT_BROWSER_SHA512}  /tmp/npm-downloads/agent-browser.tgz" | sha512sum -c -; \
    npm install --global --ignore-scripts --no-audit --fetch-retries=5 \
        /tmp/npm-downloads/playwright.tgz /tmp/npm-downloads/agent-browser.tgz; \
    rm -rf /tmp/npm-downloads; \
    for i in 1 2 3; do \
        playwright install --with-deps chromium && break || \
        { [ "$i" = 3 ] && exit 1; echo "playwright chromium install failed (attempt $i); retrying in 10s"; sleep 10; }; \
    done && chmod -R a+rX /opt/playwright && \
    agent-browser --version

# cua-driver: computer_use's MCP driver. Pinned release tarball from the
# cua-driver-rs-v* tags (never releases/latest: prereleases publish 0 assets).
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) arch=x86_64 ;; \
        arm64) arch=arm64 ;; \
        *) echo "unsupported TARGETARCH ${TARGETARCH}"; exit 1 ;; \
    esac; \
    mkdir -p /opt/cua-driver /tmp/cua-driver-dl; \
    curl -fsSL --retry 3 -o /tmp/cua-driver-dl/cua-driver.tar.gz \
        "https://github.com/trycua/cua/releases/download/cua-driver-rs-v${CUA_DRIVER_VERSION}/cua-driver-rs-${CUA_DRIVER_VERSION}-linux-${arch}-binary.tar.gz"; \
    sha_var="CUA_DRIVER_SHA256_${arch}"; \
    echo "${!sha_var}  /tmp/cua-driver-dl/cua-driver.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/cua-driver-dl/cua-driver.tar.gz -C /opt/cua-driver; \
    rm -rf /tmp/cua-driver-dl; \
    ln -sf /opt/cua-driver/cua-driver /usr/local/bin/cua-driver; \
    /usr/local/bin/cua-driver --version

# Pillow: the Screen pane's thumbnail is grabbed INSIDE the sandbox (the X socket and its cookie live
# here, not on the gateway host). Also used by the base for ad-hoc image work.
RUN pip install --no-cache-dir "pillow>=10" && python3 -c "from PIL import ImageGrab"

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
# Dockerfile ENV reaches `docker exec` only. When this image is the target of the
# ssh backend (sshd added on top), a login session gets its environment from PAM,
# so the browser location must also live where pam_env reads it or agent-browser
# reports "Chrome not found" over ssh while working under docker exec.
RUN printf 'PLAYWRIGHT_BROWSERS_PATH=/opt/playwright\nXDG_RUNTIME_DIR=/tmp/hermes-runtime\n' >> /etc/environment
CMD ["sleep", "infinity"]
