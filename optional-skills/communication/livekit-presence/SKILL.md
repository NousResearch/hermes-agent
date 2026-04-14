---
name: livekit-presence
description: Bootstrap a LiveKit voice presence for Hermes. Clone the official Python starter, export Hermes persona into it, and prepare local env/config for realtime rooms or telephony.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [LiveKit, Voice, Realtime, Telephony, Presence, Audio]
    related_skills: [telephony, fastmcp]
    category: communication
required_environment_variables:
  - name: LIVEKIT_URL
    prompt: LiveKit URL
    help: LiveKit Cloud or self-hosted server URL
    required_for: connecting the LiveKit voice agent
  - name: LIVEKIT_API_KEY
    prompt: LiveKit API key
    help: Create one in LiveKit Cloud or your self-hosted deployment
    required_for: authenticating the LiveKit voice agent
  - name: LIVEKIT_API_SECRET
    prompt: LiveKit API secret
    help: Pair this with LIVEKIT_API_KEY
    required_for: authenticating the LiveKit voice agent
prerequisites:
  commands: [python3, git]
---

# LiveKit Presence

Use this optional skill when the goal is to give Hermes a real-time voice presence instead of only a terminal or chat surface.

This skill does **not** mutate Hermes core. It bootstraps a companion LiveKit voice project from the official upstream starter, exports your Hermes persona from `SOUL.md`, and prepares the local environment so you can stand up a voice room or telephony-facing agent quickly.

## What It Covers

- clone or initialize the official `livekit-examples/agent-starter-python` project
- export Hermes persona from `~/.hermes/SOUL.md` into the LiveKit project
- write `.env.local` from the current shell or Hermes environment
- run a fast local doctor check before you waste time debugging LiveKit auth

## What It Does Not Cover

- it does not replace Hermes's own runtime loop
- it does not patch Hermes core tool loading
- it does not auto-deploy the LiveKit agent for you

Think of this as a companion presence layer, not a core rewrite.

## Install

```bash
hermes skills install official/communication/livekit-presence
```

Locate the helper script after install:

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/livekit-presence/scripts/livekit_presence.py' -print -quit)"
```

## Quick Start

Run a doctor pass first:

```bash
python3 "$SCRIPT" doctor
```

Bootstrap a local project:

```bash
python3 "$SCRIPT" bootstrap --target ~/workspace/hermes-livekit
```

If `lk` is installed, the helper prefers the official LiveKit CLI template flow. Otherwise it falls back to cloning the upstream starter repo directly.

Write `.env.local` from your current environment:

```bash
python3 "$SCRIPT" write-env --project ~/workspace/hermes-livekit
```

Refresh the exported Hermes persona snapshot later:

```bash
python3 "$SCRIPT" export-persona --project ~/workspace/hermes-livekit
```

## Recommended Workflow

1. Run `doctor` and confirm `git`, `uv`, and either `lk` or plain Git bootstrap are available.
2. Bootstrap into a canonical local workspace path.
3. Write `.env.local`.
4. Open `docs/hermes-persona.md` in the generated project and decide how much of the exported Hermes identity you want to merge into the LiveKit agent instructions.
5. Follow the upstream LiveKit starter workflow:

```bash
cd ~/workspace/hermes-livekit
uv sync
uv run python src/agent.py download-files
uv run python src/agent.py console
```

## Notes

- Upstream starter: `https://github.com/livekit-examples/agent-starter-python`
- Required env vars: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Optional extras depend on your model choice inside the LiveKit project

## Verification

Good output from `doctor` should show:

- a valid bootstrap method (`lk` or `git`)
- whether the three LiveKit env vars are present
- whether `~/.hermes/SOUL.md` exists

After bootstrap, verify these files exist in the generated project:

- `docs/hermes-persona.md`
- `docs/hermes-bootstrap.md`
- `.env.local` if you ran `write-env`
