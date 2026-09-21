---
name: credential-gating
description: "Block sensitive file access (.env, .key, secrets) unless user-approved via APPROVED: prefix. Adaptable hook pattern for credential-file protection."
version: 1.0.0
author: KENSEI (extracted from withkynam/vibecode-pro-max-kit)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, credentials, privacy, secrets, gating]
    related_skills: [hermes-agent, mcp-patterns]
---

# Credential Gating

## Purpose

Prevent accidental or unauthorized access to sensitive files containing credentials, tokens, or secrets. Agent-driven tool calls that touch `.env`, `.key`, `secrets.yaml`, or similar files should require explicit user approval.

## Threat Model

- Agent reads `.env` and leaks API keys into logs or output
- Agent writes to a secrets file and overwrites production credentials
- Agent includes credential contents in generated code or commit messages
- Agent scans credential files as part of overly-broad searches

## Gating Pattern

### Core Rule

Sensitive file access is **BLOCKED by default**. The agent must:

1. Detect the access attempt
2. Ask the user for explicit approval
3. If approved, retry with `APPROVED:` prefix on the file path
4. If denied, skip the file and continue

### Sensitive File Patterns

Block access to files matching these patterns unless explicitly approved:

| Pattern | Examples |
|---------|----------|
| `^\.env$` | `.env` |
| `^\.env\.` | `.env.local`, `.env.production`, `.env.staging` |
| `credentials` | `credentials.json`, `aws-credentials` |
| `secrets?\.ya?ml$` | `secrets.yaml`, `secret.yml` |
| `\.pem$` | TLS private keys |
| `\.key$` | SSH or API private keys |
| `id_rsa` | SSH RSA key |
| `id_ed25519` | SSH Ed25519 key |
| `*.tfstate` | Terraform state (may contain secrets) |
| `*.tfvars` | Terraform variables (often holds secrets) |
| `.dockerconfigjson` | Docker registry auth |
| `kubeconfig` | Kubernetes cluster credentials |

### Safe Exceptions

These patterns are **exempt** from blocking (examples/templates only):

- `.env.example`, `.env.sample`, `.env.template`
- `secrets.example.yaml`
- Any file explicitly prefixed with `example` or `sample`

### Approval Protocol

When the agent detects a blocked file access:

```
Agent: "I need to read [filename] which may contain sensitive data (API keys,
        passwords, tokens). Do you approve?"

User: "Yes" → Agent retries with APPROVED:path/to/file
User: "No"  → Agent skips the file and continues without it
```

### Post-Approval Checks

Even after approval:

- Strip `APPROVED:` prefix before actual file operation
- Flag suspicious paths (`..`, absolute paths outside project) with a warning
- Never include credential file contents in output sent to external APIs (LLM context, Telegram, etc.)
- Never commit credential files or their contents

## Hermes-Specific Adaptation

Hermes does not have native Claude/Codex hooks, but the pattern can be enforced via:

### Option A: Tool-Layer Guard (Recommended)

Wrap `read_file` / `write_file` / `search_files` / `terminal` calls in a pre-check:

```python
def guard_sensitive_access(tool_name, tool_input):
    paths = extract_paths(tool_input)  # file_path, path, pattern, command
    for p in paths:
        if is_sensitive(p) and not is_approved(p):
            return block_with_prompt(p)
    return allow()
```

### Option B: Skill-Level Reminder

Load this skill before any task that may touch project files. The skill injects the blocking rules into the agent's context, making it self-policing.

### Option C: Cron/Script Audit

For batch operations (cron jobs, automated scripts), pre-approve only a whitelist of safe file patterns. Any access outside the whitelist logs an alert.

## Implementation Checklist

- [ ] Define the sensitive-pattern list for your project
- [ ] Define the safe-exception list
- [ ] Choose enforcement layer (tool guard, skill reminder, or audit)
- [ ] Add `APPROVED:` prefix handling to any custom tool wrappers
- [ ] Add post-approval suspicious-path warning
- [ ] Ensure credential contents never leak to external outputs
- [ ] Document the pattern for all agent profiles

## Anti-Patterns

| Bad | Good |
|-----|------|
| Block everything in the project root indiscriminately | Block only credential-bearing files |
| Allow access because "the agent needs to know the API key" | APPROVE once, read what you need, then discard from context |
| Silently strip credentials from output without telling the user | Explicitly note when credential content is being redacted |
| Store approvals permanently | Treat each APPROVAL as single-session only |

## Related

- `systematic-debugging` — when investigating credential-related incidents
- `hermes-agent` — for configuring tool guards at the agent level
