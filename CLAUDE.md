# Claude Project Memory - Hermes Agent

## Obsidian Context Bridge

Before substantial work in this repo, read:

- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/session-start-contract.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/global-context.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/context-packs/hermes-agent-dev.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/project-context.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/active-memory.md`
- `/Users/rattanasak/ObsidianVault/HermesAgent/projects/hermes-agent-dev/handoff.md`

Use repo-local `.hermes/context.md`, `.hermes/active.md`, and `.hermes/decisions.md` when they exist.

Do not load the whole vault by default. Load the listed context first, then search only when needed.

New memory or uncertain knowledge should go to `/Users/rattanasak/ObsidianVault/HermesAgent/review-queue/` before promotion.

## Prompt Shortcuts

When the user invokes `Use Act-As`, `Use Comply`, `Go to Sleep`, `Review Chat`, or an alias, read `/Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md` and then open the mapped prompt file. Do not guess or summarize the shortcut from memory.
