# Subdirectory AGENTS.md Progressive Discovery

> How Hermes discovers AGENTS.md in subdirectories during sessions.

## Official Mechanism

Per the [Prompt Assembly docs](https://hermes-agent.nousresearch.com/docs/developer-guide/prompt-assembly) (Layer 8: Context Files):

> AGENTS.md (CWD at startup; subdirectories discovered progressively during the session via agent/subdirectory_hints.py)

## Trigger Conditions

| Tool | Operation Type | Triggers Discovery? |
|------|---------------|---------------------|
| `terminal` (mkdir + write) | Create dir + write file | ✅ |
| `write_file` (new file) | Create dir + write file | ✅ |
| `terminal` (read-only) | Pure read | ❌ |
| `web_search` / `web_extract` | Network only | ❌ |
| `read_file` | Read existing | ❌ |

## Key Takeaways

- **Session startup**: Only CWD's AGENTS.md is loaded.
- **Mid-session**: Write operations trigger nearby subdirectory scanning.
- **Telegram gateway**: CWD is forced to `$HOME`; subdirectory AGENTS.md requires
  a write operation within the session to be discovered.
- **CLI mode**: CWD is user's current directory; subdirectory discovery works the same way.

## Verification Method

To confirm subdirectory AGENTS.md loading:
1. Create a subdirectory with an AGENTS.md containing a unique marker phrase.
2. Perform a write operation inside or near that subdirectory.
3. Ask the agent to recite the marker phrase — if loaded, the phrase appears in context.
