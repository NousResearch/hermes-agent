# AgentCyber patchset notes

Base reviewed: local `main` at `fa5fd1973` with upstream `NousResearch/hermes-agent` at `c06898098` after fetch.

## Upstream sync review

The fork is 32 commits ahead and 1594 commits behind upstream. The downstream Cyber Edition lane overlaps upstream on these runtime files:

- `agent/conversation_loop.py`
- `agent/prompt_builder.py`
- `agent/system_prompt.py`
- `gateway/hooks.py`
- `gateway/run.py`
- `hermes_cli/main.py`
- `hermes_cli/models.py`
- `hermes_cli/status.py`
- `hermes_constants.py`
- `pyproject.toml`
- `toolsets.py`
- `uv.lock`

High-risk sync conflict areas are `agent/conversation_loop.py`, `toolsets.py`, `gateway/run.py`, and generated docs/site churn. Keep Cyber Edition changes isolated in new modules where possible and replay the small integration hooks after any upstream rebase.

## This lane adds

### Runtime routing guard

Files:

- `agent/cyber_routing.py`
- `agent/conversation_loop.py`

Sensitive routes now call `apply_agentcyber_route_guard()` before model invocation. The guard:

1. Keeps ordinary/general routes on the configured runtime.
2. Honors explicit hosted/Azure override when enabled.
3. Accepts already-local runtimes.
4. Switches to `agent_cyber.routing.local_open_weight` when configured.
5. Fails closed before hosted provider egress when no safe runtime is configured and `require_local_for_sensitive` is true.

The switch is transient: `restore_agentcyber_route_runtime()` restores the primary runtime at the next turn.

### Authorized asset registry and gates

Files:

- `agent/cyber_policy.py`
- `agent/agent_runtime_helpers.py`
- `agent/tool_executor.py`
- `hermes_cli/config.py`

The policy module loads built-in BC assets plus config/file/env registry entries. Tool calls are classified into S0-S5 and evaluated before execution.

S2/S3 actions require matching authorized assets. S5 destructive/high-impact actions are blocked from autonomous tool flow.

### Tests

Files:

- `tests/agent/test_agentcyber_routing_guard.py`
- updated `tests/run_agent/test_cyber_route_capture.py`

The tests cover:

- BC built-in registry matching.
- read-only S1 allow path.
- unknown external recon block.
- BC lab recon allow path.
- destructive S5 block.
- sensitive hosted route block without local runtime.
- transient local/open-weight runtime switch and restore.

### Docs

Files:

- `docs/CYBER_EDITION.md`
- `docs/CYBER_EDITION_PATCHSET.md`

## Acceptance commands

```bash
source .venv/bin/activate
python -m py_compile agent/cyber_routing.py agent/cyber_policy.py agent/conversation_loop.py agent/agent_runtime_helpers.py agent/tool_executor.py hermes_cli/config.py
python -m pytest tests/agent/test_cyber_routing.py tests/run_agent/test_cyber_route_capture.py tests/agent/test_agentcyber_routing_guard.py -q
```

## Remaining follow-ups

- Wire a human approval token/workflow for S4/S5 instead of hard-blocking all S5.
- Add live integration smoke test against the actual configured local model endpoint.
- Rebase/sync from upstream in a separate branch after preserving downstream Cyber Edition files.
- Add docs site sidebar entries only after sync, to avoid mixing generated docs churn with runtime policy changes.
