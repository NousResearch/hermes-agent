# Hermes Dev History Index

This file records the Codex-side development threads that explain the current
Hermes Desktop direction. It is intentionally stored in the repo because the
isolated Hermes Desktop state is separate from Codex thread history.

## Key Threads

| Thread ID | Topic | Why It Matters |
| --- | --- | --- |
| `019f268b-2cb0-7291-bca8-b1f02b394352` | AI employee center | Defines the desktop-first `AI 员工` direction: each employee maps to a Hermes profile, can open its own conversation, has editable `SOUL.md`, and uses a profile-specific embedded browser partition. |
| `019f2a4d-8032-7583-a582-94db7b813d4d` | Agent Homes roadmap | Defines the additive Agent Homes layer: local business database, asset folders, Obsidian export, right-side browser automation, and resource capture loops. |
| `019f2904-325c-7fe0-90d3-400bd1313dda` | Hermes architecture and open-source integration | Explains how Hermes should integrate external open-source projects through an adapter layer: tool, MCP server, plugin, or sidecar API, with optional desktop UI. |
| `019f30e8-52e7-7541-90ed-d2b1e4637f72` | MoneyPrinter Video Studio review | Reviews the first Video Studio / MoneyPrinterTurbo plan and narrows the MVP boundary. |

## Recovered Product Memory

This is the "Hermes uses Hermes to develop Hermes" version the user was
referring to. It is not just a normal Hermes chat-history database. The
important context lives across Codex/Chronicle history, repo docs, and the
development branch.

- Keep the base product as Hermes Desktop: left navigation, chat/main surface,
  right work panel/browser, settings, MCP, and skills.
- Extend through additive product layers, not by rewriting `run_agent.py`,
  the core tool schema, or the existing desktop chat surface.
- Treat `AI 员工` / Agent Homes as the business layer. Employees map to Hermes
  profiles, each profile can own conversation state, `SOUL.md`, tools, and a
  browser partition.
- Use the main-agent plus sub-agent pattern through an `Execution Brief`: the
  main agent extracts the goal, decisions, constraints, file boundaries, and
  acceptance criteria, dispatches independent workers, then integrates and
  verifies the result.
- For viral/video production, the worker split covers reference-video analysis,
  script/copy planning, storyboard/shot planning, asset binding, image
  generation, Seedance/Dreamina prompt generation, and final QA.
- For browser control, start from the existing right-rail/preview/webview
  infrastructure and connect it to Browser Automation / CDP / MCP-style
  actions. Do not start with a fresh browser shell unless the existing right
  rail cannot cover the milestone.
- For video generation, keep MoneyPrinterTurbo as a Desktop capability sidecar:
  `Video Studio` page -> Hermes adapter API -> `external/MoneyPrinterTurbo`
  FastAPI service -> output/artifact management.

## Current Local Code Anchors

| Area | Local Path |
| --- | --- |
| AI employee desktop page | `apps/desktop/src/app/ai-employees/index.tsx` |
| Desktop routes | `apps/desktop/src/app/routes.ts` |
| Desktop sidebar | `apps/desktop/src/app/chat/sidebar/index.tsx` |
| Video Studio page | `apps/desktop/src/app/video-studio/` |
| MoneyPrinterTurbo sidecar source | `external/MoneyPrinterTurbo/` |
| MoneyPrinter adapter | `capabilities/moneyprinter/` |
| Capability architecture doc | `docs/hermes-desktop-capability-architecture.md` |
| Video Studio design doc | `docs/capabilities/moneyprinter-video-studio.md` |
| Video Studio logic map | `docs/capabilities/video-studio-logic-map.md` |

## Chronicle Pointers

These Codex/Chronicle snapshots contain the context that looked like the
"missing chats" in the isolated desktop app:

| Snapshot | What It Restores |
| --- | --- |
| `2026-07-03T23-25-50-ebo1` | Agent Homes architecture roadmap, browser control design, storage split. |
| `2026-07-05T05-49-00-ento` | MoneyPrinterTurbo integration planning, Video Studio page boundary, browser capability context. |
| `2026-07-05T09-18-00-pvJg` | Video Studio UI, DeepSeek/MoneyPrinter config, main-agent/sub-agent Execution Brief workflow, viral ad remix skill. |
| `2026-07-05T09-54-00-Tfsb` | Armani 405 viral-ad production package, Video Studio UI, Hermes sessions such as `视频开发` and `F1看台热舞生成`. |

## Isolated Dev Startup

Use the fixed repo-local launcher:

```bash
hermes-dev-desktop start
```

This uses persistent, ignored local state:

```text
HERMES_HOME=/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home
HERMES_DESKTOP_USER_DATA_DIR=/Users/ruoyu/Library/Application Support/Hermes Dev
HERMES_DESKTOP_HERMES_ROOT=/Users/ruoyu/Documents/diannao/hermes-agent
```

The launcher refuses `/tmp`, `/private/tmp`, and `/var/tmp` by default because
those locations are not safe for long-lived Hermes Desktop development history.

If the shell command is missing, install or refresh it with:

```bash
/Users/ruoyu/Documents/diannao/hermes-agent/scripts/hermes-desktop-dev install-command
```

## Packaged Desktop Startup

For the unpacked package used during development:

```bash
hermes-dev-desktop pack
hermes-dev-desktop open-packaged
```

The launcher builds the development package as `Hermes Dev` with app id
`dev.ruoyu.hermes.desktop`, then `open-packaged` launches that packaged app with
the same fixed development state shown above.

For normal packaged macOS/Linux double-click startup, the Electron main process
now defaults Hermes state to that app's persistent user data directory:

```text
<Electron userData>/hermes-home
```

For `Hermes Dev`, that is `/Users/ruoyu/Library/Application Support/Hermes Dev`,
a separate Application Support directory from the normal `Hermes` desktop app.
This also keeps the packaged desktop app isolated from the user's local CLI
`~/.hermes`. Set `HERMES_DESKTOP_SHARE_CLI_HOME=1` only when intentionally
testing the packaged app against the CLI home.
