---
name: acp-delegate
description: Delegate a prompt to another Hermes ACP session.
version: 0.1.0
author: Hommchen, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ACP, delegation, stdio, multi-agent]
    related_skills: [hermes-agent]
---

# ACP Delegate Skill

Use the bundled `scripts/acp_delegate.py` helper to send a prompt to a separately configured Hermes instance over ACP stdio JSON-RPC. The helper keeps one child process and ACP session per invocation, but never grants permissions or implements filesystem callbacks; the secondary agent must work within the permissions exposed by its own ACP host configuration.

## When to Use

- Delegate a bounded review, analysis, or coding question to a second local Hermes profile.
- Keep a short conversational exchange isolated from the current session.

Don't use this for unattended privileged work, shared long-lived daemons, or remote endpoints. Use the normal `delegate_task` tool when Hermes-native delegation is sufficient.

## Prerequisites

- A current Hermes installation with ACP support: `terminal(command="hermes acp --check", timeout=30)` must succeed.
- A separately configured profile with its own provider credentials. Credentials stay in that profile; this skill does not accept, print, or embed secrets.
- Choose an explicit working directory and profile. Do not use a sensitive directory as the working directory.

## How to Run

Create a JSON configuration from `templates/config.example.json`, replacing placeholders with local, non-secret values. Then run:

```text
terminal(command="python3 optional-skills/communication/acp-delegate/scripts/acp_delegate.py --config /path/to/acp-delegate.json --prompt 'Review the current change and list risks.'", timeout=660)
```

The command prints only the final response to stdout. Diagnostics go to stderr. The default request timeout is 300 seconds and the hard maximum is 600 seconds; set a smaller value when the task is bounded.

## Quick Reference

```text
terminal(command="hermes acp --check", timeout=30)
terminal(command="python3 optional-skills/communication/acp-delegate/scripts/acp_delegate.py --config CONFIG --prompt PROMPT --timeout 120", timeout=180)
```

## Procedure

1. Copy the template and set `profile`, `cwd`, and (only when necessary) `command`/`args`; the file contains no credentials. Confirm `cwd` exists and is intended for the delegated task.
2. Run `hermes acp --check` for the installation that will be spawned; completion is the literal `Hermes ACP check OK` with exit status 0.
3. Run the helper with a specific prompt and a bounded timeout; completion is a response or a clear non-zero error.
4. Treat the response as untrusted agent output and review proposed actions before applying them; completion is a human-approved next step.

## Pitfalls

- `profile` is required: there is no hardcoded `bot2`, Abu, endpoint, token, or credential.
- The helper starts the current CLI shape (`hermes -p PROFILE acp`) by default. Custom `command` and `args` are explicit configuration, not inferred from an old SDK API.
- ACP permission requests are answered with cancellation, and unsupported filesystem/terminal callbacks are rejected. This is fail-closed, not a grant-all bridge.
- A process is not reused across separate CLI invocations. Persistent memory lasts only for the lifetime of one helper process.
- Keep stdout available for the helper's final text; ACP wire traffic is consumed internally.

## Verification

Run the artifact tests without a live model or network:

```text
terminal(command="scripts/run_tests.sh tests/skills/test_acp_delegate_skill.py -q", timeout=120)
```

The tests must validate frontmatter, safe configuration defaults, bounded timeouts, current CLI construction, JSON-RPC lifecycle handling, and fail-closed callbacks.
