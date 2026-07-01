// Help topic: tools (MCP help / AgentSkill help).
export const helpTopicTools = `\
# Tool / CLI reference

## register

**MCP input:** \`{ "username": string, "credentials_dir"?: string, "forced"?: boolean }\`  
**Skill:** \`register --username NAME [--credentials-dir DIR] [--forced]\` (or \`--api-key KEY\`).

Usernames must be 5–21 characters (local-part of your \`@atomicmail.ai\`
address).

Creates an inbox or returns the same \`{ inbox, accountId }\` when the
username matches the stored inbox local-part. A **different** username
fails by default to protect existing credentials. To add another account without
replacing the current one, pass a **separate** \`credentials_dir\` (MCP) or
\`--credentials-dir\` (skill) — see **multi_account** topic. To replace
credentials in the **same** directory, pass **\`forced: true\`** (MCP) or
**\`--forced\`** (skill) explicitly after backing up.

**After a successful register,** arrange hourly inbox polling per your runtime
(see **cron** topic — native cron hosts schedule an agent turn with
\`list_inbox.json\`; no-native-cron hosts ask the operator or remind manual
fetch).

## jmap_request

**MCP input:** \`{ "credentials_dir"?: string, "using"?: string[], "ops"?: string, "ops_file"?: string,
"vars"?: Record<string, string>, "attachments"?: { path, filename?, content_type? }[] }\` —
keys in \`vars\` are names without \`$\` (e.g. \`TO\` for \`$TO\`). Exactly one of
\`ops\` or \`ops_file\`. When \`attachments\` is non-empty, each path is read on
the MCP host, \`POST\`ed to JMAP \`uploadUrl\` (RFC 8620), then
\`$ATTACHMENT_N_BLOB_ID\` / \`$ATTACHMENT_N_NAME\` / \`$ATTACHMENT_N_TYPE\` /
\`$ATTACHMENT_N_SIZE\` and \`$ATTACHMENT_COUNT\` are available in \`ops\` (same
semantics as if you had pasted those strings in \`vars\`).

**Skill:** \`jmap_request --ops '...'\` or \`--ops-file path\` plus
\`--credentials-dir\` (optional), plus optional \`--vars '<json>'\`,
\`--attachment PATH\` (repeatable), \`--attachment-path-base DIR\`, \`--using\`,
\`--dry-run\` (not with \`--attachment\`).

## help

**MCP:** \`{ "topic"?: string }\`  
**Skill:** \`help [--topic TOPIC]\`

Topics: overview, installation, auth, jmap_cheatsheet, tools, presets, cron,
multi_account, troubleshooting. Topic \`readme\` prints the published package \`README.md\`
(same layout as npm; requires install from npm).`;
