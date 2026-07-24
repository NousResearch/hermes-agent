// Help topic: multi_account (MCP help / AgentSkill help).
export const helpTopicMultiAccount = `\
# Multiple accounts and agents

One MCP server or CLI install can manage **several isolated inboxes** by using a
**separate credential directory per account**. Each directory holds its own
\`credentials.json\`, \`session.jwt\`, and \`capability.jwt\` (mode \`0600\`).

## MCP (per tool call)

Pass optional \`credentials_dir\` on \`register\` and \`jmap_request\`:

\`\`\`json
{ "username": "alice", "credentials_dir": "~/.atomicmail/alice" }
{ "ops_file": "list_inbox.json", "credentials_dir": "~/.atomicmail/bob" }
\`\`\`

When omitted, the default directory applies (\`ATOMIC_MAIL_CREDENTIALS_DIR\` or
\`~/.atomicmail\`).

## AgentSkill (per command)

\`\`\`bash
atomicmail register --username alice --credentials-dir ~/.atomicmail/alice
atomicmail jmap_request --credentials-dir ~/.atomicmail/bob --ops-file list_inbox.json
\`\`\`

## Default vs per-call

- **Default directory:** set once via \`ATOMIC_MAIL_CREDENTIALS_DIR\` (MCP host
  \`env\`) or by omitting \`credentials_dir\` / \`--credentials-dir\`.
- **Per-call override:** use a different path on each \`register\` or
  \`jmap_request\` when the agent should act as another inbox.

You no longer need multiple MCP server entries with different \`env\` blocks
unless you prefer that layout.

## Register: forced vs separate directory

If the default directory already has an account and you want a **second**
inbox, pass a **new** \`credentials_dir\` / \`--credentials-dir\` instead of
\`forced: true\` / \`--forced\`. Use \`forced\` only when you intend to
**replace** credentials in the **same** directory (after backing up).

## Concurrency

Do not run parallel \`register\` or \`jmap_request\` calls against the **same**
credential directory. JWT files are rewritten without locking (same caveat as
the CLI). Different directories are independent.`;
