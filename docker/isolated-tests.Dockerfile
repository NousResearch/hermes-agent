# Offline test image for scripts/isolated_tests/main.py (`run` / `prove-human`).
#
# Network is used only while this image is BUILT (a human-approved, one-time
# step); the result is pinned by its local image ID (`sha256:...`) and passed
# as `--image`. The runner starts it with `--pull=never --network=none`, so
# nothing below runs or downloads at test time.
#
# Pins (no new authority):
# - base: the same Debian digest the repository Dockerfile already uses.
# - Python 3.14.9 (python-build-standalone 20261021) and uv 0.13.5: URL and
#   sha256 copied from pm/lock.json `packages.python` / `packages.uv`.
#   uv.lock's `[tool.uv] environments` is python >= 3.14, so Debian's 3.13
#   cannot install the locked set.
# - test dependencies: `uv sync --locked` against uv.lock, CI's extras/groups
#   (.github/workflows/tests.yml). The Hermes project itself is NOT installed
#   (`--no-install-project`): the code under test is the read-only /src snapshot.
#
# Build context: this file, pyproject.toml and uv.lock only. No secrets or
# build-arg tokens. The final stage declares no VOLUME and no ENV but PATH
# (scripts/isolated_tests/image.py refuses anything else); it carries no
# compiler, curl, uv, git, SSH or credential helper. USER is left unset: the
# runner forces uid 65534, so everything below is world-readable.

FROM debian:trixie-20261020@sha256:4c25275446ca371766b8b093c696782eabce06263fccacb8d5d813b3dc44b308 AS fetch
ARG TARGETARCH
ARG PYTHON_BUILD=3.14.9+20261021
ARG PYTHON_RELEASE=20261021
ARG PYTHON_X86_64_SHA256=dfa4c5a40c47f338f41658017be550c72247c391e6bfb1fa4da03ebc8451abdf
ARG PYTHON_AARCH64_SHA256=847895ba727440b012207bb0192282d7e182eb991ae5c7bb5c35c6a12a99e732
ARG UV_VERSION=0.13.5
ARG UV_X86_64_SHA256=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
ARG UV_AARCH64_SHA256=4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
        amd64) arch="x86_64"; py_sha="${PYTHON_X86_64_SHA256}"; uv_sha="${UV_X86_64_SHA256}" ;; \
        arm64) arch="aarch64"; py_sha="${PYTHON_AARCH64_SHA256}"; uv_sha="${UV_AARCH64_SHA256}" ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 -o /tmp/python.tar.gz \
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_BUILD}-${arch}-unknown-linux-gnu-install_only.tar.gz"; \
    curl -fsSL --retry 3 -o /tmp/uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${arch}-unknown-linux-gnu.tar.gz"; \
    { printf '%s  %s\n' "${py_sha}" /tmp/python.tar.gz; \
      printf '%s  %s\n' "${uv_sha}" /tmp/uv.tar.gz; } > /tmp/pins.sha256; \
    sha256sum -c /tmp/pins.sha256; \
    mkdir /opt/isotest-python /tmp/uv; \
    tar -xzf /tmp/python.tar.gz -C /opt/isotest-python --strip-components=1; \
    tar -xzf /tmp/uv.tar.gz -C /tmp/uv --strip-components=1; \
    /opt/isotest-python/bin/python3.14 -c 'import sys; assert sys.version_info[:2] == (3, 14), sys.version'; \
    rm /tmp/python.tar.gz /tmp/uv.tar.gz /tmp/pins.sha256
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN set -eu; \
    UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/isotest-venv \
    /tmp/uv/uv sync --locked --no-install-project --compile-bytecode --link-mode=copy \
        --python /opt/isotest-python/bin/python3.14 \
        --extra all --group dev --extra anthropic --extra mistral --extra fal --extra modal \
        --extra daytona --extra parallel-web --extra keychain; \
    /opt/isotest-venv/bin/python -c 'import pytest, pytest_asyncio'; \
    if /opt/isotest-venv/bin/python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("hermes_cli") else 1)'; then \
        echo "hermes_cli importable from site-packages: project must not be installed" >&2; exit 1; \
    fi; \
    rm -rf /tmp/uv /build; \
    chmod -R a+rX,go-w /opt/isotest-python /opt/isotest-venv

FROM debian:trixie-20261020@sha256:4c25275446ca371766b8b093c696782eabce06263fccacb8d5d813b3dc44b308
# bash (run_tests.sh) is Essential in the base; nothing else is added.
COPY --from=fetch /opt/isotest-python /opt/isotest-python
COPY --from=fetch /opt/isotest-venv /opt/isotest-venv
ENV PATH=/opt/isotest-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
