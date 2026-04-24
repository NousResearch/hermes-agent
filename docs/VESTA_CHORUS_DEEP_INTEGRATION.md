# Vesta Chorus deep integration runbook

This records the Phase 1-4 activation path for Vesta von der Proto.

## Phase 1 — ambient Chorus memory

- Hermes memory provider: `plugins/memory/chorus`
- Shared JSON-RPC client: `plugins/chorus_common.py`
- Vesta profile uses `memory.provider: chorus`.
- Provider exposes compact tools:
  - `chorus_resume_context`
  - `chorus_memory_query`
  - `chorus_memory_store`
  - `chorus_emit_signal`
- Compression now threads `MemoryProvider.on_pre_compress()` return text into compression focus so provider preservation guidance reaches the summarizer.

## Phase 2 — lifecycle and gateway hooks

- Standalone plugin: `plugins/vesta-chorus`
- Registered hooks:
  - `on_session_start`
  - `on_session_end`
  - `on_session_finalize`
  - `pre_gateway_dispatch`
  - `pre_llm_call`
  - `post_tool_call`
- Gateway hook redacts and avoids storing credential-like inbound text.
- Risky tool audit redacts secret-like args before Chorus memory writes.

## Phase 3 — Vesta tools

Vesta plugin toolset: `vesta_chorus`.

Tools:
- `vesta_wake_briefing`
- `vesta_closeout`
- `vesta_worker_audit`
- `vesta_gate_check`
- `vesta_workstream_sweep`

Governance note: scoped `launch.agent_worker` is audit-only by Inu's explicit Vesta doctrine. Spend, production deploy, customer/legal/public, secret rotation, DNS change, open-source release, destructive memory/workstream operations remain approval-gated unless an active Circle authorizes them.

## Phase 4 — always-on posture

Vesta profile activation:
- `toolsets` includes `vesta_chorus`.
- Chorus MCP tree mode is `runtime`.
- Webhook platform enabled on local port `8644`.
- Dynamic Hermes webhook routes:
  - `chorus-alerts`
  - `chorus-briefings`
- Supervised WSL runner is tmux session `vesta-gateway`.

Commands:

```bash
# Start/restart supervised gateway
tmux kill-session -t vesta-gateway 2>/dev/null || true
tmux new-session -d -s vesta-gateway -x 160 -y 45 \
  'export HERMES_PROFILE=vesta HERMES_HOME=/home/inu/.hermes/profiles/vesta VESTA_CHORUS_HOOKS=1; cd /home/inu/agents-of-proto; vesta gateway run --accept-hooks'

# Health
curl -fsS http://127.0.0.1:8644/health
vesta webhook list

tmux capture-pane -t vesta-gateway -p -S -80 | tail -50
```

Cron jobs created:
- `vesta-chorus-watch-briefing` — daily 08:15
- `vesta-stale-workstream-sweep` — daily 08:30
- `vesta-approval-webhook-health-sweep` — daily 08:45

Chorus webhook registry remains intentionally unregistered until the Chorus signing secret can be connected to the Hermes route secret without exposing it in logs/transcripts. Hermes ingress is live and authenticated locally.

## Verification

Focused gates:

```bash
/home/inu/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/plugins/memory/test_chorus_provider.py \
  tests/plugins/test_vesta_chorus_plugin.py \
  tests/test_chorus_pre_compress_focus.py \
  -q -o 'addopts='

/home/inu/.hermes/hermes-agent/venv/bin/python -m py_compile \
  plugins/chorus_common.py \
  plugins/memory/chorus/__init__.py \
  plugins/vesta-chorus/__init__.py \
  run_agent.py
```

Observed focused result: `11 passed`.

Broader `tests/plugins` currently has one unrelated existing failure in Hindsight post-setup key preservation; Vesta/Chorus focused tests pass.
