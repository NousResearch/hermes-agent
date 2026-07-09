# Awesome OpenCode → Hermes-Agent War-room Mapping

Source catalog: `awesome-opencode/awesome-opencode`.

Principle: Hermes should not embed OpenCode as a hidden second runtime. Port the best patterns into Hermes-native plugins, skills, MCP entries, or CLI workflows.

## First wave implemented here

| OpenCode idea | Hermes-native implementation |
|---|---|
| `opencode-telemetry`, token/cost/session observability entries | `codex-ops` local SQLite run ledger for Codex terminal runs. |
| `opencode-snip`, context-pruning entries | `codex-ops` terminal output compaction for long Codex/dev/test logs. |
| log sanitizer entries | early high-confidence secret redaction before compaction. |
| `envsitter-guard`, `cc-safety-net` | Codex no-sandbox guardrails for `--danger-full-access`, `--dangerously-bypass-approvals-and-sandbox`, `--sandbox danger-full-access`, `-s danger-full-access`, and safe defaults in the skill. |
| mission-control / hcom / ensemble-style workflows | `codex-operations` skill protocol for lane planning, bounded agents, and verification. |

## Deferred tracks

| Track | Candidate sources | Hermes shape |
|---|---|---|
| Multi-agent worktree runner | mission-control, hcom, opencode-a2a, ensemble, flowdeck | CLI command or kanban/worktree orchestrator; not a core tool. |
| Project memory / RAG | harness-memory, OpenCodeRAG, SwarmVault, Simple Memory, Kibi, Lemma | project-local memory provider or MCP server with reviewable Markdown/SQLite state. |
| Code intelligence | tree-sitter-language-pack, semantic-anchors | service-gated code-intel plugin/MCP; no default schema cost. |
| Browser automation | Chromium browser plugin | optional browser backend/MCP bridge; avoid duplicating existing browser/computer-use tools. |
| Skill ecosystem | agent-skills-jdt, managed-skills, openskills | skill import/validation UX and curated optional-skill packs. |
| Server/process manager | server-manager/background entries | skill + CLI workflow over existing Hermes `terminal`/`process` tools. |
| OTEL export | opencode-plugin-otel | standalone plugin repo or optional exporter; never emit message bodies by default. |

## Council guardrails

- Keep prompt caching stable.
- Do not add a new core model tool for what a plugin/skill/CLI can do.
- Do not store raw secrets, message bodies, or full terminal output in telemetry.
- Use `config.yaml` for non-secret settings.
- Prefer local, inspectable state over opaque global memory.
- Verification must be real: test command, exit code, and diff inspection.
