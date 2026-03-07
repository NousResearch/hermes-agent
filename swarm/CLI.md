# Swarm CLI Integration

## Registering `/swarm` in cli.py

To wire up the `/swarm` command, two changes are needed in `/Users/andy/hermes-agent/cli.py`:

### 1. Add to COMMANDS dict (line ~726)

```python
COMMANDS = {
    ...
    "/copycode": "Import Claude Code commands as Hermes skills",
    "/swarm": "Multi-agent swarm orchestration (run, plan, status, cancel)",  # ADD THIS
    "/quit": "Exit the CLI (also: /exit, /q)",
}
```

### 2. Add dispatch in process_command (line ~1894, before the `else:` skill-command fallback)

```python
        elif cmd_lower == "/reload-mcp":
            self._reload_mcp()
        elif cmd_lower == "/copycode":
            self._copycode()
        elif cmd_lower.startswith("/swarm"):                          # ADD THIS BLOCK
            from swarm.cli import handle_swarm_command               #
            swarm_args = cmd_original.split(maxsplit=1)               #
            handle_swarm_command(                                     #
                swarm_args[1] if len(swarm_args) > 1 else "",         #
                self,                                                 #
            )                                                         #
        else:
            # Check for skill slash commands ...
```

The import is deferred (inside the branch) so the swarm module is only loaded when the user actually invokes `/swarm`, avoiding import-time overhead for users who never use it.

### Autocomplete

The `SlashCommandCompleter` (line ~739) already iterates over `COMMANDS`, so adding the entry to the dict is sufficient for tab-completion to work.
