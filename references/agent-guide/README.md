# Agent Guide References

These files are normative companions to the root [`AGENTS.md`](../../AGENTS.md).
The root file keeps universal judgment rules concise; these references hold detailed,
area-specific guidance. Before editing, read the root file, then every reference whose
scope matches the touched files. Do not treat this directory as optional background.

| Reference | Scope |
| --- | --- |
| [Contribution and footprint](contribution-and-footprint.md) | Contribution intent, premise checks, rejected patterns, footprint ladder |
| [Development and project structure](development-and-project-structure.md) | Environment, repository map, TypeScript style, dependency chain |
| [Agent, CLI, TUI, and desktop architecture](agent-cli-and-tui-architecture.md) | Agent loop and user-facing runtime architecture |
| [Tools, configuration, dependencies, and themes](tools-config-and-themes.md) | Core tools, dependency pinning, config, cwd, skins |
| [Plugins and skills](plugins-and-skills.md) | Plugin surfaces, provider discovery, skill standards |
| [Toolsets and durable systems](toolsets-and-durable-systems.md) | Toolsets, delegation, curator, cron, kanban |
| [Policies, profiles, and pitfalls](policies-profiles-and-pitfalls.md) | Prompt caching, background notifications, profiles, known traps |
| [Testing](testing.md) | Test wrapper, isolation, behavior-focused test rules |

## Maintenance rule

Keep the root `AGENTS.md` and other core instruction documents at **250 lines or fewer**.
When detail would push one over the limit, move the detail into a focused document here
(or a nearer scoped `references/` directory), and leave an explicit routing link in the
source document. Keep each reference focused and at 250 lines or fewer where practical;
split it again instead of rebuilding another monolith.
