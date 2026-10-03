---
name: plugin-scanner
description: "Scan Agent Skills, plugins, MCP servers, and agent repositories before install or trust using the local HOL Guard plugin-scanner CLI."
version: 1.0.0
author: Hashgraph Online
license: Apache-2.0
platforms: [linux, macos, windows]
category: security
triggers:
  - "scan this skill before installing it"
  - "check this plugin for prompt injection"
  - "audit this MCP server"
  - "inspect this agent repository before trusting it"
  - "scan this SKILL.md"
  - "check this agent package before install"
toolsets:
  - terminal
  - file
metadata:
  hermes:
    tags: [Security, Agent-Skills, MCP, Supply-Chain, Prompt-Injection]
    related_skills: [oss-forensics]
---

# Plugin Scanner

Use HOL Guard's local `plugin-scanner` when a user wants to inspect an Agent Skill, plugin, MCP server, agent package, or repository before installing or trusting it.

`plugin-scanner` is provided by the open-source HOL Guard project. Scanning runs locally and does not require Guard Cloud.

## When to Use

Use this skill when the user asks to:

- scan or audit a `SKILL.md` before installation;
- inspect an MCP server or agent plugin for security risks;
- check a third-party agent repository before trusting it;
- look for prompt injection, credential exposure, unsafe commands, or suspicious package/install behavior;
- run a repeatable pre-install security check in CI.

For incident response or historical repository compromise investigation, prefer the related `oss-forensics` skill.

## Safety Rules

- Never execute code from the target repository just to scan it.
- Never run the target's install scripts, package lifecycle hooks, or arbitrary shell commands.
- Never read `.env` files, credential stores, private keys, or unrelated user secrets.
- Prefer scanning a local path or repository the user has already chosen to inspect.
- Treat scanner findings as security evidence, not a guarantee that a package is safe.
- Ask before installing `plugin-scanner` if the command is not already available.

## Procedure

### 1. Check whether the scanner is installed

Try the scanner directly so the check is portable across supported platforms:

```bash
plugin-scanner --help
```

If the command is unavailable, explain that `plugin-scanner` is a separate open-source CLI distribution from the HOL Guard repository and ask before changing the user's environment. A common isolated installation path is:

```bash
pipx install plugin-scanner
```

Do not assume an existing `hol-guard` installation also provides the `plugin-scanner` command. If `pipx` is unavailable, point the user to the plugin-scanner installation documentation instead of silently modifying the system Python environment.

### 2. Scan the selected target without executing it

For a repository or directory:

```bash
plugin-scanner scan PATH --format markdown
```

For machine-readable findings:

```bash
plugin-scanner scan PATH --format json
```

For Agent Skill or plugin structure validation:

```bash
plugin-scanner lint PATH
plugin-scanner verify PATH
```

Use the narrowest path that contains the material the user asked to inspect.

### 3. Interpret the result conservatively

Report:

1. the target that was scanned;
2. the highest severity finding;
3. concrete files or rules involved;
4. whether findings involve prompt injection, secret/exfiltration behavior, command execution, dependency/install behavior, or MCP-specific risk;
5. the recommended next action.

Do not say a package is "safe" solely because no finding was returned. Say that the current scan did not detect a covered issue.

## Verification

Before claiming the scan completed, confirm that:

- the scanner command returned normally;
- the requested target path was actually scanned;
- findings or the no-covered-issue result were captured;
- no target code or install hook was executed as part of the inspection.

## Sources

- HOL Guard: https://github.com/hashgraph-online/hol-guard
- Portable Agent Skill distribution: https://github.com/hashgraph-online/hol-guard-plugin/tree/main/skills/plugin-scanner
