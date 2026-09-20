FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git patch \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 worker \
    && mkdir /workspace \
    && chown worker:worker /workspace

ENV HOME=/home/worker \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

USER worker
WORKDIR /workspace

CMD ["sleep", "infinity"]
