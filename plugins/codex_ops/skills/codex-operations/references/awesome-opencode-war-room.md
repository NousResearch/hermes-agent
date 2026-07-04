# Awesome OpenCode → Hermes-Agent War-room Mapping

Source catalog: `awesome-opencode/awesome-opencode`.

Principle: Hermes should not embed OpenCode as a hidden second runtime. Port the best patterns into Hermes' native plugin, skill, CLI, and hook architecture.

## First wave implemented here

| Catalog pattern | Hermes-native integration |
|---|---|
| `opencode-telemetry`, token/cost/session observability entries | `codex-ops` local SQLite run ledger for Codex terminal runs. |
| `opencode-snip`, context-pruning entries | `codex-ops` terminal output compaction for long Codex/dev/test logs. |
| Secret redaction / log hygiene entries | High-confidence redaction of private keys, API keys, GitHub/OpenAI/Anthropic/AWS/Slack/JWT tokens. |
| `envsitter-guard`, `cc-safety-net` | `codex exec` no-sandbox guardrail and safe defaults in the skill. |
| Agent orchestration runbooks | `codex-operations` skill: bounded lanes, verification, evidence reporting. |

## Deferred wave

| Candidate | Why deferred |
|---|---|
| Multi-agent worktree runner | Needs UX and process lifecycle design; should be a CLI/kanban lane, not a model tool. |
| Browser/app automation adapters | Existing Hermes browser/computer-use tools cover this; avoid duplicate runtimes. |
| GitHub/Linear/Jira connectors | Hermes already has domain skills/plugins; integrate by capability, not by OpenCode wrapper. |
| OTEL export | Useful later as optional exporter; never emit message bodies by default. |
| MCP bridge | Hermes already supports MCP; import only specific servers after security review. |

## Guardrails

- Do not add a new core model tool for what a plugin/skill/CLI can do.
- Do not store raw secrets, message bodies, or full terminal output in telemetry.
- Default to `codex exec --cd <repo> --sandbox workspace-write`.
- Treat no-sandbox Codex modes as explicit operator exceptions, not defaults.
