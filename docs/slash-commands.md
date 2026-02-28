# Slash Commands Reference

Quick reference for all CLI slash commands in Hermes Agent.

## Navigation & Control

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit` | Exit the CLI (aliases: `/exit`, `/q`) |
| `/clear` | Clear screen and reset conversation |
| `/new` | Start a new conversation |
| `/reset` | Reset conversation (keep screen) |

## Tools & Configuration

| Command | Description |
|---------|-------------|
| `/tools` | List all available tools |
| `/toolsets` | List available toolsets |
| `/model` | Show or change the current model |
| `/model <name>` | Switch to a different model |
| `/config` | Show current configuration |
| `/prompt` | View/set custom system prompt |
| `/personality` | Set a predefined personality |

## Conversation

| Command | Description |
|---------|-------------|
| `/history` | Show conversation history |
| `/retry` | Retry the last message |
| `/undo` | Remove the last user/assistant exchange |
| `/save` | Save the current conversation |

## Advanced

| Command | Description |
|---------|-------------|
| `/cron` | Manage scheduled tasks |
| `/skills` | Search, install, or manage skills |
| `/platforms` | Show gateway/messaging platform status |

## Examples

### Changing Models

```
/model anthropic/claude-sonnet-4
```

### Setting a Custom Prompt

```
/prompt You are a helpful coding assistant specializing in Python.
```

### Managing Toolsets

Run with specific toolsets:
```bash
python cli.py --toolsets web,terminal
```

Then check enabled toolsets:
```
/toolsets
```

## Tips

- Commands are case-insensitive (`/HELP` = `/help`)
- Use Tab for autocomplete
- Most commands work mid-conversation
- `/clear` is useful for starting fresh without restarting
