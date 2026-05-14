# Matrix Synapse Gateway Integration Tests

This opt-in harness runs Hermes' Matrix adapter against a real local Synapse
homeserver. It covers the gateway parity path that fake clients cannot prove:
account registration, login/access tokens, DM-style private rooms, invite/join,
media upload/download, bot send/receive, and startup old-event filtering.

## Run

```bash
docker compose -f tests/e2e/matrix_synapse_gateway/docker-compose.yml up -d
HERMES_MATRIX_SYNAPSE_INTEGRATION=1 \
  scripts/run_tests.sh -m "integration and matrix_synapse" \
  tests/e2e/matrix_synapse_gateway/test_gateway.py
docker compose -f tests/e2e/matrix_synapse_gateway/docker-compose.yml down -v
```

The compose file binds Synapse to `127.0.0.1:28448` by default. Override the
host port with:

```bash
SYNAPSE_HOST_PORT=28449 docker compose -f tests/e2e/matrix_synapse_gateway/docker-compose.yml up -d
HERMES_MATRIX_SYNAPSE_URL=http://127.0.0.1:28449 HERMES_MATRIX_SYNAPSE_INTEGRATION=1 \
  scripts/run_tests.sh -m "integration and matrix_synapse" \
  tests/e2e/matrix_synapse_gateway/test_gateway.py
```

The registration shared secret is test-only and lives inside the local Docker
volume. `down -v` removes accounts, media, and room state.

## Optional E2EE Smoke

Encrypted-room coverage is marked separately:

```bash
HERMES_MATRIX_SYNAPSE_INTEGRATION=1 HERMES_MATRIX_SYNAPSE_E2EE=1 \
  scripts/run_tests.sh -m "integration and matrix_synapse and matrix_e2ee" \
  tests/e2e/matrix_synapse_gateway/test_gateway.py
```

The E2EE smoke is intentionally opt-in because it requires the Matrix extra
with crypto dependencies and tends to be slower on developer machines.
