---
name: codex-bridge
description: Start and control local Codex tasks through Hermes Codex Bridge app-server integration.
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [codex, agent, bridge, app-server]
    category: software-development
---

# Codex Bridge

Use this skill when you need Hermes to start or steer a local Codex task through the Codex app-server protocol.

## CLI

Run the reference CLI from the repository root:

```bash
python skills/codex-bridge/references/cli.py start --prompt "Inspect this repository and summarize the test layout."
python skills/codex-bridge/references/cli.py status <task_id>
python skills/codex-bridge/references/cli.py list
python skills/codex-bridge/references/cli.py steer <task_id> --instruction "Focus only on tests."
python skills/codex-bridge/references/cli.py interrupt <task_id>
python skills/codex-bridge/references/cli.py respond <task_id> --request-id <request_id> --decision decline
python skills/codex-bridge/references/cli.py smoke-test --wait 10 --timeout 60
```

The CLI is a productized wrapper around `tools.codex_bridge_tool.codex_bridge`.
It does not implement the app-server protocol itself and does not use mailbox,
inbox, or outbox files.

## Safety Defaults

- Sandbox is limited to `read-only` or `workspace-write`.
- `danger-full-access` is rejected.
- Approval policy is limited to `untrusted` or `on-request`.
- `approval_policy=never` is rejected.
- `start` requires a non-empty prompt and an existing `cwd`.

## Output

Commands print JSON to stdout. Validation errors return:

```json
{"success": false, "error": "..."}
```

Successful `start` output is validated to ensure:

- `success` is `true`
- `protocol.mailbox` is `false`
- `protocol.transport` includes `app-server`
- task id, Codex thread id, and Codex turn id are present

The smoke test starts an async Codex task, polls `status`, and succeeds only
when the final task status is `completed` and `CODEX_ASYNC_OK` appears in
`recent_events` or `final_summary`.
