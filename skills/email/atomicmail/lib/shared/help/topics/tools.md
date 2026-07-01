# Tool / CLI reference

## register

- MCP: `{ "username": string, "credentials_dir"?: string, "forced"?: boolean }`
- Skill: `register --username NAME [--credentials-dir DIR] [--forced]`

## jmap_request

- MCP: one of `ops` or `ops_file`, optional `vars`, optional `attachments`.
- Skill: one of `--ops` or `--ops-file`, optional `--vars`, `--attachment`.

## help

- MCP: `{ "topic"?: string }`
- Skill: `help [--topic TOPIC]`
