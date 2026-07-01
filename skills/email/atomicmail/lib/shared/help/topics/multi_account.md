# Multiple accounts and agents

Use separate credential directories per account. Each directory stores:

- `credentials.json`
- `session.jwt`
- `capability.jwt`

Override per call with `credentials_dir` (MCP) or `--credentials-dir` (skill).
