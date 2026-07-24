// Help topic: installation (MCP help / AgentSkill help).
export const helpTopicInstallation = `\
# Atomic Mail — Installation

## MCP (stdio)

\`\`\`json
{
  "mcpServers": {
    "atomicmail": {
      "command": "npx",
      "args": ["-y", "@atomicmail/mcp"]
    }
  }
}
\`\`\`

## AgentSkill (shell)

\`\`\`bash
npx --package=@atomicmail/agent-skill atomicmail register --username "myagent"
npx --package=@atomicmail/agent-skill atomicmail jmap_request \\
  --ops-file list_inbox.json
npx --package=@atomicmail/agent-skill atomicmail help
\`\`\`

## After register: hourly inbox polling

Registration only creates credentials. Arrange hourly inbox polling per your
runtime (see **cron** topic): native cron hosts schedule an hourly **agent**
turn with \`list_inbox.json\`; hosts without native cron should ask the operator
to set up polling on a capable host or remind manual fetch. Do not cron
\`atomicmail jmap_request\` alone.

## Shared credentials

MCP and the skill use the same directory layout (default \`~/.atomicmail/\`):

- \`credentials.json\`, \`session.jwt\`, \`capability.jwt\`

## Overriding defaults

- Endpoints: \`ATOMIC_MAIL_AUTH_URL\`, \`ATOMIC_MAIL_API_URL\`
- Default credentials path: \`ATOMIC_MAIL_CREDENTIALS_DIR\` (MCP host \`env\`),
  \`--credentials-dir\` (skill), or per-call \`credentials_dir\` (MCP) /
  \`--credentials-dir\` (skill) — see **multi_account** topic
- Optional PoW salt: \`ATOMIC_MAIL_SCRYPT_SALT\``;
