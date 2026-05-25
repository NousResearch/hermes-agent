# memd commands

- Root: `/home/aparcedodev/.hermes/hermes-agent/.memd`
- Commands: `27`

memd owns the native CLI surfaces. External bridge surfaces stay listed so they can be migrated, swapped, or reimplemented on other harnesses without pretending memd owns them.

## Native memd CLI

## memd commands

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd commands --output .memd`
- purpose: inspect the bundle command catalog itself

## memd status

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd status --output .memd`
- purpose: check bundle readiness and missing files

## memd state

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd state --output .memd`
- purpose: show canonical operator state across truth, claims, freshness, and divergence

## memd wake

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd wake --output .memd --intent current_task --write`
- purpose: refresh the wake surface before a turn

## memd resume

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd resume --output .memd --intent current_task`
- purpose: resume compact working memory for the current task

## memd lookup

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd lookup --output .memd --query "..."`
- purpose: run bundle-aware recall before answering

## memd checkpoint

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd checkpoint --output .memd --content "..." [--auto-commit] [--roadmap-set KEY=VALUE]`
- purpose: write short-term task state into the live backend (--auto-commit commits only small tracked dirty sets, --roadmap-set patches ROADMAP_STATE)

## memd claim create

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd claim create --output .memd --scope task:example`
- purpose: claim an active task scope with TTL-backed coordination state

## memd claim list

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd claim list --output .memd --summary`
- purpose: inspect active and expired task claims for the current bundle

## memd claim close

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd claim close --output .memd --scope task:example`
- purpose: release a claim when work is done or handed off

## memd remember

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd remember --output .memd --kind decision --content "..."`
- purpose: persist durable typed memory

## memd teach

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd teach --output .memd --content "fact the user just taught"`
- purpose: persist new user-taught facts with safe default tags/provenance

## memd handoff

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd handoff --output .memd --prompt`
- purpose: emit a compact takeover packet

## memd packs

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd packs --root .memd --summary`
- purpose: inspect visible harness packs in a bundle

## memd skills

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd skills --summary`
- purpose: inspect the discovered skill catalog

## memd hook capture

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd hook capture --output .memd --stdin --summary`
- purpose: record live turn changes and refresh bundle truth

## memd hook spill

- surface: `memd`
- kind: `CLI`
- ownership: `memd`
- role: `native-cli`
- compatibility: `memd-binary`, `bundle-root-present`
- command: `memd hook spill --output .memd --stdin --apply`
- purpose: spill compaction state into durable memory

## Bridge surfaces

## /memory

- surface: `Claude Code`
- kind: `slash`
- ownership: `external`
- role: `bridge-surface`
- compatibility: `bundle-root-present`, `claude-import-bridge`, `claude-project-bridge`
- command: `/memory`
- purpose: universal Claude bridge command surfaced during migration

## $gsd-autonomous

- surface: `Codex`
- kind: `external-skill`
- ownership: `external`
- role: `bridge-surface`
- compatibility: `codex-skill-installed`
- command: `$gsd-autonomous`
- purpose: universal Codex bridge command surfaced during migration

## $gsd-map-codebase

- surface: `Codex`
- kind: `external-skill`
- ownership: `external`
- role: `bridge-surface`
- compatibility: `codex-skill-installed`
- command: `$gsd-map-codebase`
- purpose: universal Codex bridge command surfaced during migration

## Bundle helpers

## .memd/agents/codex.sh

- surface: `Codex`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/codex.sh`
- purpose: launch the Codex harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/codex.sh`

## .memd/agents/teach.sh

- surface: `memd`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`, `user-taught-fact-capture`
- command: `.memd/agents/teach.sh --content "fact the user just taught"`
- purpose: capture new user-taught facts through the teach-safe helper
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/teach.sh`

## .memd/agents/claude-code.sh

- surface: `Claude Code`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/claude-code.sh`
- purpose: launch the Claude Code harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/claude-code.sh`

## .memd/agents/agent-zero.sh

- surface: `Agent Zero`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/agent-zero.sh`
- purpose: launch the Agent Zero harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/agent-zero.sh`

## .memd/agents/openclaw.sh

- surface: `OpenClaw`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/openclaw.sh`
- purpose: launch the OpenClaw harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/openclaw.sh`

## .memd/agents/opencode.sh

- surface: `OpenCode`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/opencode.sh`
- purpose: launch the OpenCode harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/opencode.sh`

## .memd/agents/hermes.sh

- surface: `Hermes`
- kind: `helper`
- ownership: `memd`
- role: `bundle-helper`
- compatibility: `bundle-root-present`, `launcher-script-present`, `launcher-script-executable`
- command: `.memd/agents/hermes.sh`
- purpose: launch the Hermes harness pack
- path: `/home/aparcedodev/.hermes/hermes-agent/.memd/agents/hermes.sh`

