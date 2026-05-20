# Agent Guide References

## Refactor analysis

`AGENTS.md` was reduced from a long-form manual into a short operational contract. The selection rule: keep only information that almost every coding assistant needs before its first edit; move conditional/subsystem detail into targeted references.

### Kept in root AGENTS.md
- **Mission + scope:** Agents need to know this repository is Hermes Agent and that root AGENTS.md is a fast-loading contract.
- **Environment + commands:** Needed before any edit: venv activation, test wrapper, config/log paths.
- **Load-bearing architecture map:** Small set of files that direct nearly every change; prevents blind edits.
- **Hard rules/policies:** Prompt caching, message alternation, profile-safe paths, secret handling, tests isolation.
- **Change-type routing table:** Fast pointer to the right detailed reference without loading 51k chars every session.
- **Verification + commit checklist:** Universal finish gate every coding assistant needs.

### Moved out of root AGENTS.md
- **Deep class/API explanations:** Useful when editing that subsystem, wasteful for unrelated tasks.
- **TUI, plugin, skin, skills, toolsets, delegation, cron, kanban deep dives:** Subsystem-specific; load only when relevant.
- **Long code samples and schemas:** Better kept in dedicated references to avoid bloating the root context.
- **Pitfall catalog:** Important but searchable by topic; root keeps only top invariants and links.
- **Testing rationale and anti-pattern examples:** Detailed guidance belongs in testing reference.

## Extracted reference map

- [overview-original-preamble](original-preamble.md) — 4 original lines
- [Development Environment](development-environment.md) — 11 original lines
- [Project Structure](project-structure.md) — 53 original lines
- [File Dependency Chain](file-dependency-chain.md) — 14 original lines
- [AIAgent Class (run_agent.py)](aiagent-class-run-agent-py.md) — 60 original lines
- [CLI Architecture (cli.py)](cli-architecture-cli-py.md) — 54 original lines
- [TUI Architecture (ui-tui + tui_gateway)](tui-architecture-ui-tui-tui-gateway.md) — 66 original lines
- [Adding New Tools](adding-new-tools.md) — 48 original lines
- [Dependency Pinning Policy](dependency-pinning-policy.md) — 23 original lines
- [Adding Configuration](adding-configuration.md) — 64 original lines
- [Skin/Theme System](skin-theme-system.md) — 89 original lines
- [Plugins](plugins.md) — 100 original lines
- [Skills](skills.md) — 111 original lines
- [Toolsets](toolsets.md) — 19 original lines
- [Delegation (`delegate_task`)](delegation-delegate-task.md) — 33 original lines
- [Curator (skill lifecycle)](curator-skill-lifecycle.md) — 34 original lines
- [Cron (scheduled jobs)](cron-scheduled-jobs.md) — 36 original lines
- [Kanban (multi-agent work queue)](kanban-multi-agent-work-queue.md) — 41 original lines
- [Important Policies](important-policies.md) — 30 original lines
- [Profiles: Multi-Instance Support](profiles-multi-instance-support.md) — 56 original lines
- [Known Pitfalls](known-pitfalls.md) — 65 original lines
- [Testing](testing.md) — 93 original lines

## Maintenance rule

- Keep repository-root `AGENTS.md` at about 150 lines or less.
- Add new subsystem detail to a file in this directory and link it from the routing table.
- If a rule applies to every agent on every task, summarize it in root `AGENTS.md`; otherwise keep it here.
