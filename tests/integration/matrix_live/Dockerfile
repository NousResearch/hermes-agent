FROM debian:13.4@sha256:e2d08da6f42ef4b09b165d55528a12727aeed8240dc9edf888e3ec07e10ef9da

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HERMES_RUNTIME_DIR=/opt/hermes/tools \
    CC=gcc \
    CXX=g++

RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        ca-certificates python3 python3-venv gcc g++ make cmake \
        python3-dev libffi-dev libolm-dev ffmpeg openssl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -u 10000 -m -d /opt/data hermes

WORKDIR /opt/hermes

COPY pm/ pm/
COPY hermes_constants.py hermes_constants.py
COPY hermes_cli/__init__.py hermes_cli/runtime_state.py hermes_cli/
COPY scripts/bundles/payload.py scripts/bundles/payload.py
RUN python3 -c 'from pm import ensure; [ensure(name, explicit=True) for name in ("uv", "python")]' && \
    python3 -c 'from pathlib import Path; from pm import installed_package; Path("/usr/local/bin/python3").symlink_to(installed_package("python").binary)'

COPY pyproject.toml uv.lock ./
RUN touch README.md && \
    python3 -m pm.build_env --source /opt/hermes --python /usr/local/bin/python3 \
        --out /opt/hermes/.venv --no-install-project --sealed \
        --extra matrix --extra matrix-rtc --group test && \
    /opt/hermes/.venv/bin/python -c 'import olm; import mautrix.crypto; import nio'

COPY . .

ENV HERMES_HOME=/opt/data
