# memd harness bridge matrix

Generated: 2026-05-25T04:05:22.729558164+00:00

Overall portability class: **adapter-required**

| Harness | Wired | Portability | Missing surfaces | Notes |
|---|---|---|---|---|
| codex | yes | harness-native | none | Codex is native when the config, hook, and skill surfaces are all present. |
| claude | no | adapter-required | hook | Claude is native when the settings and session hook surfaces exist. |
| claw | no | adapter-required | binary, config | Claw is memd-ready when the binary is installed, config exists, and memd skills are visible through shared skill roots. |
| openclaw | no | adapter-required | agents, bootstrap | OpenClaw is native when AGENTS.md and BOOTSTRAP.md bridge surfaces exist. |
| opencode | no | adapter-required | config, plugin | OpenCode is native when config, plugin, and command surfaces all route through memd. |

## Adapter Required Surface

If a harness is not wired, `memd` treats it as adapter-required and surfaces the missing bridge surfaces instead of pretending the skill is universally available.
