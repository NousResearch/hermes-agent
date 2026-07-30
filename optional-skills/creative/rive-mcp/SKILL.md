---
name: rive-mcp
description: "Create and edit Rive animations through MCP."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: creative
    tags: [Rive, Animation, MCP, Motion, State Machines, Design-to-Code]
    related_skills: [blender-mcp]
prerequisites:
  commands: [hermes]
---

# Rive Skill

Optional skill — **not active until installed**:

```bash
hermes skills install official/creative/rive-mcp
```

After install, scripts live under `~/.hermes/skills/creative/rive-mcp/`. In prose
below, `{SKILL_DIR}` means that directory (or this repo path before install).

Use Rive from Hermes through MCP. There are two real integration paths:

- **Official Rive desktop MCP** — local HTTP server in the Rive desktop editor
  at `http://127.0.0.1:9791/mcp`.
- **RiveMCP** — third-party standalone stdio MCP (`npx -y rivemcp`) that writes
  `.riv` / `.rev` files headlessly.

Do not invent a Rive CLI export path. Official Rive exports runtime `.riv`
files from the editor UI; official MCP requires the desktop editor to be open.

## When to Use

- The user wants to create or edit Rive artboards, shapes, animations, state
  machines, View Models, scripts, shaders, or runtime `.riv` files.
- The user already has the Rive desktop app open and wants AI to manipulate the
  current file.
- The user wants headless `.riv` / `.rev` generation and accepts RiveMCP's
  third-party licensing model.

## Prerequisites

Run the doctor first:

```bash
python {SKILL_DIR}/scripts/rive_doctor.py
```

Official path:

- Rive desktop editor / Early Access app on macOS or Windows.
- Rive app open, with a file and artboard created.
- MCP endpoint reachable at `http://127.0.0.1:9791/mcp`.

RiveMCP path:

- `node` / `npx`, or a downloaded RiveMCP binary.
- RiveMCP has 3 free exports per machine; after that it requires
  `RIVEMCP_LICENSE_KEY`.

## How to Run

### Path A — Official Rive desktop MCP

Use this when the user is working in the Rive editor and wants live changes.

```bash
hermes mcp add rive --url http://127.0.0.1:9791/mcp
```

When prompted whether the server requires auth, answer **no**. The Rive app
must be open before probing. Start a new Hermes session after adding the MCP so
the tools load.

Rive's official docs say to finish a prompt by typing `End Prompt` so the Rive
editor applies the AI changes. Treat this as part of the interaction contract.

### Path B — RiveMCP headless stdio

Use this when the user wants automation, CI, or file generation without the
Rive desktop app.

```bash
hermes mcp add rivemcp --command npx --args -y rivemcp
```

If the user has a paid key:

```bash
export RIVEMCP_LICENSE_KEY=...
hermes mcp add rivemcp --command npx --env RIVEMCP_LICENSE_KEY=$RIVEMCP_LICENSE_KEY --args -y rivemcp
```

Start a new Hermes session after adding the MCP.

## Quick Reference

| Goal | Path |
| --- | --- |
| Live-edit current Rive file | official `rive` HTTP MCP |
| Headless `.riv` / `.rev` creation | third-party `rivemcp` stdio MCP |
| Inspect official endpoint | `python {SKILL_DIR}/scripts/rive_doctor.py` |
| Configure official MCP | `hermes mcp add rive --url http://127.0.0.1:9791/mcp` |
| Configure RiveMCP | `hermes mcp add rivemcp --command npx --args -y rivemcp` |

## Procedure

1. Ask which path the user wants: **official desktop** or **headless RiveMCP**.
2. Run `rive_doctor.py`.
3. Configure the selected MCP using the commands above.
4. Start a new Hermes session so MCP tools load.
5. For official Rive, keep the Rive app open and use `End Prompt` when the
   editor asks for confirmation.
6. For RiveMCP, export `.riv` for runtime or `.rev` for editor handoff.

## Pitfalls

- **Official Rive MCP is desktop-bound.** It is not a headless server and is
  currently documented for macOS / Windows desktop editor only.
- **RiveMCP is third-party and licensed.** It offers free exports, then needs a
  license key. Make that explicit before choosing it.
- **Do not confuse `.riv` and `.rev`.** `.riv` is runtime output; `.rev` opens
  in the Rive editor for further manual editing.
- **Tool surfaces evolve.** After connecting, inspect the actual MCP tools with
  `hermes mcp configure rive` or `hermes mcp configure rivemcp`.

## Verification

- Official path: `rive_doctor.py` shows port `9791` reachable, and Hermes MCP
  probe lists Rive tools.
- RiveMCP path: Hermes MCP probe lists RiveMCP tools, then export a trivial
  `.riv` and open it in a Rive runtime/editor.

See `references/research.md` for source links and verified constraints.
