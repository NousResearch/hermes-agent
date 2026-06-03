FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BROWSER_TOOLS_ARTIFACT_DIR=/tmp/browser-tools-artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install fastapi 'uvicorn[standard]' pydantic 'scrapling[fetchers]'

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8790
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8790/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8790"]
