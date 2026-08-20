---
name: secure-input
description: Secure masked credential entry for Hermes Agent.
version: 1.1.0
author: Peter (lesterppo)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, credentials, input, hermes]
    category: hermes-agent
    related_skills: [hermes-native-tool-wiring, hermes-tool-review]
---

# Secure Input Skill

Adds a `secure_input` tool that prompts the user with masked typing (****)
and writes the value directly to the Hermes `.env` file. The agent receives
only `{"stored": true, "key": "..."}` — the secret never enters the agent
context, session database, or logs.

## When to Use

- The agent needs to ask for an API key without exposing it in chat
- User wants a `/si KEY` slash command for quick credential entry
- A tool reports missing credentials and the agent should prompt securely

## Prerequisites

- Hermes Agent installed

## How to Run

Install by copying `secure_input.py` into `~/.hermes/hermes-agent/tools/`,
then restart Hermes:

```bash
cp secure_input.py ~/.hermes/hermes-agent/tools/
```

Enable the `secure_input` toolset via `hermes tools`, or use `/si KEY` directly.

## Quick Reference

| Method | Invocation |
|--------|-----------|
| Slash command | `/si OPENAI_API_KEY` or `/secure-input OPENAI_API_KEY` |
| Agent tool | Agent calls `secure_input(key="...")` when toolset enabled |
| CLI wrapper | `python3 secure-input.py --key OPENAI_API_KEY` |

## Procedure

1. Copy `secure_input.py` to `~/.hermes/hermes-agent/tools/`
2. Restart Hermes
3. Test: `/si TEST_KEY`
4. Enable the `secure_input` toolset via `hermes tools` for agent invocation

## Pitfalls

- **Not in `_HERMES_CORE_TOOLS`** — zero token cost by default. Must
  explicitly enable the `secure_input` toolset.
- **Python strings are immutable** — true memory scrubbing requires
  ctypes. The real defence is that the value never enters the agent
  context, session DB, or logs.
- **Temp file permissions**: uses `os.open(mode=0o600)` from creation,
  no umask race window. On Windows, POSIX file modes are not enforced
  by NTFS — the primary defence remains that the secret never enters
  agent context.
- **`export KEY=value` syntax**: handled via `removeprefix("export ")`.
- **Masked echo on Windows**: `termios` is unavailable, so input uses
  `getpass` (hidden, no per-character feedback) instead of `****`.

## Verification

Start Hermes and run: `/si TEST_KEY`
