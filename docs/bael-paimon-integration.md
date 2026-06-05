# BAEL Paimon Integration

## Role Of This Fork

This fork exposes the Hermes agent as the backend for BAEL, a mobile-first assistant interface for non-technical users. BAEL should not depend on the Hermes dashboard or on raw workspace navigation. The mobile app talks to the Hermes API server through one authenticated port.

## BAEL API Contract

The API server must keep these endpoints available:

- `GET /health`
- `GET /v1/models`
- `GET /v1/capabilities`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions` with `stream: true` for SSE realtime mode
- `GET /api/artifacts`
- `GET /api/artifacts/{relative_path}`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`

All non-health endpoints use the same bearer token configured by `API_SERVER_KEY`.

## Artifact Rules

`/api/artifacts` is BAEL's mobile view into the agent workshop. It must expose deliverables, not internal infrastructure.

The endpoint:

- resolves configured roots from `API_SERVER_ARTIFACT_ROOTS`, `API_SERVER_ARTIFACT_ROOT`, or `HERMES_ARTIFACT_ROOT`;
- falls back to safe local Paimon output folders when no root is configured;
- supports multiple roots as virtual folders;
- returns BAEL-compatible objects with `artifactType`, `kind`, `mimeType`, `modifiedAt`, `name`, `path`, `preview`, and `size`;
- rejects path traversal;
- refuses secret/vendor/cache names such as `.env`, `.ssh`, `.git`, `.infisical`, `credentials`, `keys`, `secrets`, `node_modules`, and virtualenv folders;
- follows symlinks only if the resolved target stays inside the allowed root.

## Audio Rules

BAEL sends recorded audio as base64 to `/v1/audio/transcriptions`. The backend owns provider configuration and calls the configured Hermes STT tool.

BAEL asks for assistant speech through `/v1/audio/speech`. The backend owns provider configuration and calls the configured Hermes TTS tool.

Do not add provider keys, voice keys, or proxy credentials to BAEL. Those belong in the Paimon/Hermes runtime secret layer.

## Upstream Sync Guidance

When merging from the public Hermes upstream:

1. Preserve the BAEL API contract above unless BAEL has a newer ADR documenting a migration.
2. Prefer upstream security and runtime improvements, but keep BAEL auth, artifact path-safety, artifact response shape, and audio routes.
3. If `gateway/platforms/api_server.py` conflicts, resolve by keeping upstream internals and reapplying the BAEL handlers with tests.
4. Run the focused BAEL contract checks before pushing:

```bash
/home/magnus-vaos/paimon/venv/bin/python -m py_compile gateway/platforms/api_server.py
/home/magnus-vaos/paimon/venv/bin/python -m pytest tests/gateway/test_api_server_artifacts.py -q
/home/magnus-vaos/paimon/venv/bin/python -m pytest tests/gateway/test_api_server_multimodal.py -q
```

## Local Deployment

After changes to this fork:

```bash
/home/magnus-vaos/paimon/venv/bin/pip install -e .
systemctl --user restart paimon-gateway.service
curl -fsS http://100.122.118.25:8643/health
```

Then test authenticated capabilities and artifacts without printing the key value.
