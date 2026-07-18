---
name: a2a-peer
description: Contact an authenticated, configured Hermes A2A peer through the hermes a2a CLI.
---

# A2A peer

Use this skill when work should be sent to another Hermes instance registered as a named A2A peer.
The integration deliberately adds no permanent model tool. Invoke the CLI through the terminal.

## Safe workflow

1. Run `hermes a2a peer list` and choose an existing peer name. Never pass a URL to `card`, `ask`,
   `get`, `list`, or `cancel`; those commands accept configured names only.
2. Inspect capability metadata with `hermes a2a card PEER --json` when needed.
3. Prefer explicit stdin for prompts, especially multiline or shell-sensitive text:

   ```bash
   printf '%s\n' "$REQUEST" | hermes a2a ask PEER --stdin --json
   ```

4. The first request creates a context. Later `ask` calls continue the peer's saved context by
   default. Use `--new-context` for an unrelated conversation, or `--context-id ID` only when an
   exact context was returned by that peer.
5. Prefer `--json` for automation. Preserve the returned `taskId` and `contextId`; use
   `hermes a2a get PEER TASK_ID --json`, `list`, or `cancel` to inspect or control work.

Do not print, request, copy, or store A2A bearer tokens in prompts. Credentials are managed only by
`hermes a2a peer`, `principal`, and `credential` commands in the profile-local credential store.
Never reuse `API_SERVER_KEY` as an A2A credential.
