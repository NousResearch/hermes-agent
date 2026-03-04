# Agent-Vault Skill — Placeholder-Based Secret Management

## Overview

Agent-Vault is a security layer that prevents AI agents from seeing or transmitting
sensitive information (API keys, tokens, passwords) to LLM provider servers. It
replaces real secrets with placeholders like `<agent-vault:key>` during file I/O,
so the agent can read and write config files containing credentials without ever
seeing their actual values.

**Use this skill whenever:**
- The user mentions API keys, tokens, passwords, or credentials
- You need to read or write `.env`, `.yaml`, `.json`, `.toml`, or config files
  that may contain secrets
- The user asks you to set up a service, configure an integration, or deploy anything
  requiring credentials
- You see `<agent-vault:...>` placeholders in file content

---

## Installation

### Check if agent-vault is installed
```bash
agent-vault --version
```

### Install via npm (one-time setup)
```bash
npm install -g @botiverse/agent-vault
```

### Or use npx (no install required, ~3s overhead per call)
```bash
npx @botiverse/agent-vault --version
```

### Initialise the vault (first time only)
```bash
agent-vault init
```

---

## How It Works
```
Agent sees:     api_key: <agent-vault:openai-key>
Disk contains:  api_key: sk-proj-abc123def456...
```

Agent-vault acts as a bidirectional redaction layer:
- **Read**: Returns file content with secrets replaced by `<agent-vault:key>` placeholders
- **Write**: Accepts content with placeholders and restores real values before writing to disk

The real secret **never enters the LLM context**.

---

## Commands Reference

### Safe Commands (agent can call these)
```bash
# Read a file — returns content with secrets replaced by placeholders
agent-vault read <file>

# Write a file — restores placeholders to real values before writing
agent-vault write <file> --content 'key: <agent-vault:openai-key>'

# Check if a secret exists in the vault
agent-vault has <key>

# List all stored key names (never shows values)
agent-vault list
```

### Human-Only Commands (TTY-guarded, never call these)
```bash
# Store a secret — requires interactive terminal, masked input
agent-vault set <key>

# View secret metadata (--reveal requires TTY, cannot be piped)
agent-vault get <key>

# Bulk import from .env file
agent-vault import .env

# Scan a file for unvaulted secrets
agent-vault scan <file>
```

> **Important:** Never attempt to call `agent-vault set`, `agent-vault get --reveal`,
> or `agent-vault import`. These are TTY-guarded and will fail from agent context.
> Always ask the user to run them in their own terminal.

---

## Core Workflows

### Workflow 1: Reading a config file safely

Instead of `read_file config.yaml`, use:
```bash
agent-vault read config.yaml
```

The output will show `<agent-vault:key-name>` wherever secrets exist. You can work
with the file structure safely — the real values never reach the LLM context.

### Workflow 2: Writing a config file with secrets
```bash
# First check what secrets are available
agent-vault list

# Write the file using placeholders for secret values
agent-vault write config.yaml --content 'openai_api_key: <agent-vault:openai-key>
database_url: <agent-vault:db-url>
debug: false'
```

Agent-vault restores the real values on disk automatically.

### Workflow 3: Setting up a new secret (instruct the user)

If a required secret is not in the vault, ask the user to add it:

> "Please run this in your terminal to store the secret:
> `agent-vault set openai-key`
> You'll be prompted to enter the value securely. Let me know when done."

Then verify:
```bash
agent-vault has openai-key
```

### Workflow 4: Vault not initialised

If you see an error like "vault not found" or "vault not initialised":

> "Please run `agent-vault init` in your terminal first, then `agent-vault set <key>`
> for each secret you want to use."

---

## UNVAULTED Secrets

If agent-vault detects a high-entropy string it doesn't recognise, it replaces it with:
```
<agent-vault:UNVAULTED:sha256:XXXXXXXX>
```

This means the file contains a secret that hasn't been registered in the vault yet.
Ask the user to run `agent-vault scan <file>` to identify and vault all unvaulted secrets.

---

## Decision Guide: agent-vault vs regular file tools

| Situation | Use |
|---|---|
| Reading `.env`, credential config files | `agent-vault read` |
| Writing any file that needs a secret | `agent-vault write` |
| Reading non-sensitive files (code, docs) | `read_file` (normal) |
| Writing non-sensitive files | `write_file` (normal) |
| Unsure if file has secrets | `agent-vault read` (safe default) |

---

## Pitfalls

- **Never ask the user to paste a secret into the chat.** Always use the vault workflow.
- **Never call `agent-vault get --reveal`.** It is TTY-guarded and will fail.
- **Node.js 18+ required.** If npm is not available, suggest the user install Node.js first.
- **Vault key is irreplaceable.** If `~/.agent-vault/vault.key` is lost, secrets cannot
  be recovered. Remind the user to back it up.
- **npx adds ~3s overhead.** For frequent use, recommend `npm install -g`.

---

## Verification

After writing a file with placeholders, verify the real values were restored:
```bash
# Check the file was written (do NOT read it with agent-vault — use a hash check)
sha256sum config.yaml

# Or ask the user to verify the file contents in their terminal
```

---

## References

- [agent-vault GitHub](https://github.com/botiverse/agent-vault) — Source (Apache-2.0)
- [Related: Security issue #363](https://github.com/NousResearch/hermes-agent/issues/363) — File tool output redaction gap
