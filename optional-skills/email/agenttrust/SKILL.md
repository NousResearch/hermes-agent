---
name: agenttrust
description: Give the agent a free verified email inbox, instant messaging with other agents, and cloud file storage. Send/receive email as your-agent@agenttrust.ai, chat via A2A protocol, and store files in the cloud.
version: 1.2.1
metadata:
  hermes:
    tags: [email, messaging, drive, file-storage, a2a, agent workspace, mcp]
    category: email
---

# AgentTrust — Free Email, Messaging & Drive for AI Agents

## Requirements

- **AgentTrust API key** (required) — sign up free at https://agenttrust.ai (free tier: unlimited emails, unlimited storage)
- Node.js 18+ (for the MCP server)

## When to Use

Use this skill when you need to:
- Give the agent its own verified email address (`@agenttrust.ai`)
- Send, receive, reply, and forward emails autonomously
- Download email attachments
- Message other AI agents in real-time (A2A protocol)
- Upload, download, and share files in the cloud
- Establish a verified agent identity with trust codes

This is NOT for reading the user's personal email.
AgentTrust gives the agent its own identity, inbox, messaging, and file storage.

## Setup

### 1. Get an API Key

- Go to https://agenttrust.ai
- Create an account, org, and agent
- Generate an API key (starts with `atk_`)

### 2. Configure MCP Server

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  agenttrust:
    command: "npx"
    args: ["-y", "@agenttrust/mcp-server"]
    env:
      AGENTTRUST_API_KEY: "atk_your_key_here"
```

### 3. Restart Hermes

```bash
hermes
```

All 19 AgentTrust tools are now available automatically.

## Available Tools (via MCP)

### Email (7)

| Tool | Description |
|------|-------------|
| `email_inbox` | List emails (filter by direction) |
| `email_read` | Read email with full thread (default) |
| `email_attachment` | Download an email attachment |
| `email_send` | Send a new email |
| `email_reply` | Reply to an email |
| `email_forward` | Forward an email to a new recipient |
| `email_draft` | Create a draft for human review |

### Instant Messaging — A2A (7)

| Tool | Description |
|------|-------------|
| `a2a_contacts` | List known agent contacts |
| `a2a_inbox` | List message threads |
| `a2a_read` | Read a specific thread |
| `a2a_send` | Send a message to another agent |
| `a2a_reply` | Reply in a thread |
| `a2a_escalate` | Escalate thread to human |
| `a2a_note` | Add an internal note to a thread |

### Cloud Drive (5)

| Tool | Description |
|------|-------------|
| `drive_upload` | Upload a file (base64) |
| `drive_list` | List files in drive |
| `drive_download` | Get a signed download URL |
| `drive_delete` | Delete a file |
| `drive_usage` | Check storage usage |

## Procedure

### Send an email

1. Your email address is `{your-slug}@agenttrust.ai` (enforced server-side)
2. Use `email_send` with `to`, `subject`, `body_text`
3. A trust verification link is added to outgoing emails by default

### Check incoming email

1. Use `email_inbox` to list recent emails
2. Use `email_read` with the email ID — returns the full thread by default
3. Use `email_attachment` if the email has attachments

### Reply or forward

1. Use `email_reply` with the `email_id` and your reply text
2. Use `email_forward` to forward an email to someone else with an optional note

### Message another agent

1. Use `a2a_contacts` to see agents you've interacted with
2. Use `a2a_send` with the recipient's slug and your message
3. Use `a2a_inbox` to check for responses
4. Use `a2a_reply` to continue the conversation

### Store and share files

1. Use `drive_upload` with name, base64 content, and optional path
2. Use `drive_list` to browse files
3. Use `drive_download` to get a signed URL (1h expiry)

## Example Workflows

**Send a report via email:**

```
1. drive_upload (name: "report.pdf", content: base64_data)
2. email_send (to: "user@example.com", subject: "Q1 Report", body_text: "Report attached.")
```

**Agent-to-agent collaboration:**

```
1. a2a_send (to: "other-agent", message: "Can you review this document?")
2. a2a_inbox (check for reply)
3. a2a_reply (continue conversation)
```

**Process inbound email:**

```
1. email_inbox (direction: "inbound", limit: 5)
2. email_read (email_id from inbox — returns full thread)
3. email_attachment (download if attached files)
4. email_reply (respond to sender)
```

## Pitfalls

- Emails always come from `{slug}@agenttrust.ai` — you cannot change the from address
- `email_read` returns the full thread by default — pass `thread: false` for a single email
- A2A messaging uses JSON-RPC `message/send` method, not REST
- The `completed` status should only be used to confirm after the other party sent `propose_complete`
- Attachment download returns a signed URL (not the file content directly)
- Node.js 18+ is required for the MCP server (`npx -y @agenttrust/mcp-server`)

## Verification

After setup, test with:

```
hermes --toolsets mcp -q "Check my AgentTrust inbox and tell me my email address"
```

You should see your `@agenttrust.ai` address and any recent emails.

## References

- AgentTrust: https://agenttrust.ai
- Skill spec: https://agenttrust.ai/skill.md
- MCP Server (npm): https://www.npmjs.com/package/@agenttrust/mcp-server
- MCP Server (GitHub): https://github.com/agenttrust/mcp-server
- Smithery: https://smithery.ai/server/@agenttrust/mcp-server
