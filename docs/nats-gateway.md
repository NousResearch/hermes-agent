# NATS Gateway — Developer Pointers

Expose Hermes Agent as a NATS micro service using the **NATS Agent Protocol v0.3**. Callers publish prompts to `agents.prompt.hermes.<owner>.<session_name>` and iterate streamed responses back.

## Where to read next

- **User-facing setup guide:** [`website/docs/user-guide/messaging/nats.md`](../website/docs/user-guide/messaging/nats.md) — configuration, env vars, examples, security model, troubleshooting
- **Architectural reference:** [`docs/nats-gateway-design.md`](nats-gateway-design.md) — protocol↔adapter mapping, streaming model, session identity, lock scope, approval hook, failure modes, and §17 lessons learned
- **Implementation progress log:** [`docs/nats-gateway-progress.md`](nats-gateway-progress.md) — phase-by-phase checklist and decision log (primary source of truth for "where are we" across context-cleared sessions)
- **Protocol spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.3)
- **SDKs (split as of v0.5 client / v0.1 agent, 2026-04-30; on PyPI as of 2026-05-01):**
  - Client SDK — [`synadia-ai-agents`](https://pypi.org/project/synadia-ai-agents/) (import root `synadia_ai.agents`) — wire types: `Envelope`, `Attachment`, `ResponseChunk`, `StatusChunk`, errors, discovery, `load_context_options`.
  - Agent SDK — [`synadia-ai-agent-service`](https://pypi.org/project/synadia-ai-agent-service/) (import root `synadia_ai.agent_service`) — host surface: `AgentService`, `PromptStream`, `PromptHandler`.
  - Source lives in the [`synadia-ai/synadia-agents`](https://github.com/synadia-ai/synadia-agents) monorepo (`client-sdk/python`, `agent-sdk/python`, runnable `examples/` per SDK).

## Smoke-test recipe

One-time bootstrap on a fresh checkout — both SDKs come in via the `[nats]` extra:

```bash
./setup-hermes.sh
uv sync --all-extras --locked
```

Local broker + gateway + one-shot prompt:

```bash
# terminal 1 — broker
nats-server -p 4222 -a 127.0.0.1

# terminal 2 — gateway. Persist via `hermes setup gateway` (writes ~/.hermes/.env);
# inline-env override below is fine for one-shot smoke runs.
NATS_URL=nats://127.0.0.1:4222 HERMES_NATS_OWNER=dev HERMES_NATS_SESSION_NAME=smoke \
  hermes gateway run

# terminal 3 — caller (the wheel doesn't ship examples/, clone the monorepo for them)
git clone https://github.com/synadia-ai/synadia-agents.git
cd synadia-agents/client-sdk/python
uv run python examples/02-prompt-text.py \
    --url nats://127.0.0.1:4222 \
    --session smoke \
    "what is 2+2? answer in one short sentence"
```

The user-facing doc has the full walkthrough including attachments, mid-stream approvals, an inline-Python caller snippet that doesn't require cloning the monorepo, and the discovery / heartbeat interop checks via `nats micro list` and `nats req agents.status.hermes.<owner>.<session_name> ''`.
