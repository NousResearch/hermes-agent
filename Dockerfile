# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HERMES_HOME=/data/hermes \
    API_SERVER_HOST=0.0.0.0 \
    API_SERVER_PORT=8642

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 hermes \
    && useradd --system --uid 10001 --gid hermes --home-dir /data/hermes --create-home hermes

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY acp_adapter ./acp_adapter
COPY cron ./cron
COPY gateway ./gateway
COPY hermes_cli ./hermes_cli
COPY plugins ./plugins
COPY skills ./skills
COPY tools ./tools
COPY tui_gateway ./tui_gateway
COPY *.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install '.[messaging]' \
    && mkdir -p /data/hermes/logs \
    && chown -R hermes:hermes /data/hermes /app

USER hermes
VOLUME ["/data/hermes"]
EXPOSE 8642

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${API_SERVER_PORT:-8642}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["hermes", "gateway", "run"]
