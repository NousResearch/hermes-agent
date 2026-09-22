#!/usr/bin/env bash
# Build a pinned native release. /opt/data is never replaced by an upgrade.
set -euo pipefail
export PATH="/.sprite/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/opt/hermes-python
revision="$1"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2
release="/opt/hermes-releases/$revision"
install -d -m 755 /opt/hermes-releases
if [[ ! -f "$release/.sprites-ready" ]]; then
    if [[ ! -d "$release/.git" ]]; then
        git init -q "$release"
        git -C "$release" remote add origin https://github.com/NousResearch/hermes-agent.git
    fi
    git -C "$release" fetch --depth 1 origin "$revision"
    git -C "$release" checkout --detach FETCH_HEAD
    [[ "$(git -C "$release" rev-parse HEAD)" = "$revision" ]]
    # s6-setuidgid is reused by the common bootstrap hook, not as the supervisor.
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends s6 libolm-dev libffi-dev build-essential
    cd "$release"
    # Avoid the base image's pyenv shim and system Python 3.14 (unsupported by Hermes).
    uv sync --python 3.13.15 --managed-python --frozen --extra all --extra messaging --extra otlp
    .venv/bin/python -c 'import sqlite3; assert sqlite3.sqlite_version_info >= (3,51,3)'
    npm ci --no-audit --ignore-scripts
    npm --prefix web run build
    npm --prefix ui-tui run build
    touch .sprites-ready
fi
id hermes >/dev/null 2>&1 || useradd --uid 10000 --create-home --home-dir /opt/data --shell /bin/bash hermes
# No checkout or install command writes into the user home.
[[ "${2:-activate}" != "stage" ]] || exit 0
ln -sfn "$release" /opt/hermes.next
mv -Tf /opt/hermes.next /opt/hermes
ln -sfn /opt/hermes/.venv/bin/hermes /usr/local/bin/hermes
