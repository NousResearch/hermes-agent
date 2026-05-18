# NATS Gateway — Developer Pointers

Expose Hermes Agent as a NATS micro service using the **NATS Agent Protocol v0.2**. Callers publish prompts to `agents.hermes.<owner>.<name>` and iterate streamed responses back.

## Where to read next

- **User-facing setup guide:** [`website/docs/user-guide/messaging/nats.md`](../website/docs/user-guide/messaging/nats.md) — configuration, env vars, examples, security model, troubleshooting
- **Architectural reference:** [`docs/nats-gateway-design.md`](nats-gateway-design.md) — protocol↔adapter mapping, streaming model, session identity, lock scope, approval hook, failure modes, and §17 lessons learned
- **Implementation progress log:** [`docs/nats-gateway-progress.md`](nats-gateway-progress.md) — phase-by-phase checklist and decision log (primary source of truth for "where are we" across context-cleared sessions)
- **Protocol spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.2.0-draft)
- **Agent SDK:** `../nats-ai-pysdk` (package `natsagent`; install with `uv pip install --python venv/bin/python -e ../nats-ai-pysdk` until it ships on PyPI)

## Smoke-test recipe

One-time bootstrap on a fresh checkout:

```bash
./setup-hermes.sh
uv pip install --python venv/bin/python -e ../nats-ai-pysdk
```

Local broker + gateway + one-shot prompt:

```bash
# terminal 1 — broker
nats-server -p 4222 -a 127.0.0.1

# terminal 2 — gateway (uses config.yaml or env vars)
NATS_URL=nats://127.0.0.1:4222 HERMES_NATS_OWNER=dev HERMES_NATS_NAME=smoke \
  hermes gateway run

# terminal 3 — caller
cd ../nats-ai-pysdk
uv run python examples/02-prompt-text.py \
    --url nats://127.0.0.1:4222 \
    "what is 2+2? answer in one short sentence"
```

The user-facing doc has the full walkthrough including attachments, mid-stream approvals, and the discovery / heartbeat interop checks via `nats req '$SRV.INFO.agents'`.
