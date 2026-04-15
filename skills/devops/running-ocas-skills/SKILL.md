---
name: running-ocas-skills
description: >
  How to correctly execute OCAS skills — manually, via delegation, and via cron.
  Covers the MCP toolset trap, the "never rewrite as script" rule, and verification steps.
metadata:
  author: Indigo Karasu
  version: "1.0.0"
  hermes:
    tags: [ocas, mcp, delegation, cron]
---

# Running OCAS Skills

## Rules

1. **NEVER write a throwaway Python script to implement logic that a skill's SKILL.md already documents.** Run the skill as-is through the agent's orchestration layer. Skills are declarative specifications executed by the LLM — they are not Python packages with executable code. This applies even when you think "the skill has no binary" or "I need to do it myself." Skills are procedures for the agent to follow, not executables. If the skill describes how to do something, follow those instructions using your available tools.

2. **NEVER use `cronjob run` when the user says "run X now."** That triggers a cron schedule, not an immediate execution. Use `delegate_task` with the skill loaded (or run the logic directly if you have the tools).

3. **When delegating an OCAS skill that depends on MCP tools (mempalace, spotify, etc.), you MUST either:**
   - Omit the `toolsets` parameter entirely so the child inherits ALL parent tools including MCP, OR
   - Explicitly include the MCP toolset (e.g., `mcp-mempalace`) in the toolsets list.
   
   If you pass only toolsets like `["terminal", "file", "web"]`, MCP tools are excluded and the subagent will **simulate or hallucinate** writes instead of performing them. This is the #1 cause of "ran but nothing happened" failures.

4. **The cron scheduler creates AIAgent instances with `disabled_toolsets=["cronjob", "messaging", "clarify"]`** only. This means cron jobs inherit all other tools including MCP — they will work correctly IF the MCP server's command path is correct.

5. **When a sub-agent or cron job reports a "tool issue" or "permission error", 99% of the time it's because you gave it the wrong toolsets.** Do NOT assume the tool itself is broken. Check toolset inheritance first.

6. **Google OAuth scopes must include ALL APIs you need.** The token in `~/.hermes/google_token.json` must have scopes for Drive, Gmail, Calendar, Contacts, Sheets, and Docs. If any scope is missing (e.g., only `contacts` is present), API calls to that service will return 403. Always verify `token.scopes` before diagnosing API failures. Re-authorize with the full scope set if gaps are found.

7. **OCAS skills that access Google services (Bower, Vesper, Dispatch, Sands, Taste, Bower) need the full OAuth token.** If a skill reports 403 errors, check the token scopes first, not the skill logic.

## MCP Server Path Pitfall

Hermes runs in a venv whose `python3` is Python 3.11. MCP servers that install under the system Python 3.13 (like mempalace) MUST use an absolute path in `~/.hermes/config.yaml`:

```yaml
mcp:
  mempalace:
    command: /usr/bin/python3   # NOT "python3"
    args:
      - -m
      - mempalace.mcp_server
    enabled: true
```

If bare `python3` is used, the MCP server silently fails at import (`ModuleNotFoundError`), and any skill depending on it will appear to "work" but produce no persisted data.

## Verification After Skill Runs

After running a skill that writes to an external store (MemPalace, Weave, etc.):

1. Check the store directly — not just the skill's journal. Journal writes mean nothing if the external call failed silently.
2. For MemPalace: `mempalace status` (check drawer count), `mempalace search "<expected content>"`, and `~/.mempalace/wal/write_log.jsonl` (check recent timestamp).
3. For the skill's own files: check `ingestion_log.jsonl` and `decisions.jsonl` in the data directory to confirm the processing cursor advanced.

## Delegate Task Pattern

Use this pattern when running an OCAS skill manually:

```
delegate_task(
  goal="...",
  skills=["ocas-<skill-name>"],  # loads the SKILL.md
  # NO toolsets parameter — inherits all parent tools including MCP
)
```

If you must restrict toolsets, always include the MCP server the skill needs:
```
toolsets=["terminal", "file", "mcp-mempalace"]  # explicit MCP inclusion
```