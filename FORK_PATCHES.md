# Fork Patch Manifest

Updated 2026-05-15 during the upstream merge tracked by HAD-1001.

The old sealed `hadto_patches/` compatibility domain was retired in favor of
native upstream runtime, sandboxing, approval, gateway, provider, plugin, and
Codex app-server support. The fork now carries only the small generic security
patches below.

| file | patch category | description | upstream-able? |
| --- | --- | --- | --- |
| `agent/trajectory.py` | security | Deep-copies and redacts trajectory content before writing JSONL trajectory artifacts. | yes |
| `gateway/platforms/discord.py` | robustness | Treats dummy/no-history Discord channels as empty context and avoids catching a mocked non-exception `discord.Forbidden` in tests. | yes |
| `gateway/platforms/slack.py` | messaging | Normalizes raw Slack member IDs like `@U...`/`@W...` into mrkdwn mention IDs so Slack notifications fire. | yes |
| `hermes_state.py` | security | Redacts tool message content before writing to `state.db`. | yes |
| `tools/delegate_tool.py` | security | Tells delegated child agents not to call Slack, Moshi, webhook URLs, or other external notification endpoints unless the task explicitly requires it. | yes |

## Retired In This Merge

- `hadto_patches/*` sealed compatibility wrappers.
- Doppler/env preload wrappers in core entrypoints and tools.
- ctx.rs task-binding wrappers in `hermes_cli/ctx_runtime.py` and terminal/file seams.
- Fork-local command approval and file/write sandbox policy, replaced by upstream `tools/approval.py`, `agent/file_safety.py`, `tools/terminal_tool.py`, and related tests/docs.
- Fork-local gateway/platform wrappers, replaced by upstream gateway platform registry and plugin surfaces.
- Fork-local De Novo book-study/webhook tests and code, which should live in the Hadto plugin or De Novo integration repos if still needed.
