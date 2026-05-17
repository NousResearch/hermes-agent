# CLI

The Hermes Agent CLI provides an interactive terminal interface for working with the agent.

## Running the CLI

```bash
# Basic usage
hermes

# With specific model
hermes --model "anthropic/-sonnet-4"

# With specific provider
hermes --provider nous        # Use Nous Portal (requires: hermes login)
hermes --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes --toolsets "web,terminal,skills"

# Resume previous sessions
hermes --continue             # Resume the most recent CLI session (-c)
hermes --resume <session_id>  # Resume a specific session by ID (-r)

# Verbose mode
hermes --verbose
```

## Architecture

The CLI is implemented in `cli.py` and uses:

- **Rich** - Welcome banner with ASCII art and styled panels
- **prompt_toolkit** - Fixed input area with command history
- **KawaiiSpinner** - Animated feedback during operations

```
┌─────────────────────────────────────────────────┐
│  HERMES-AGENT ASCII Logo                        │
│  ┌─────────────┐ ┌────────────────────────────┐ │
│  │  Caduceus   │ │ Model: -opus-4.5     │ │
│  │  ASCII Art  │ │ Terminal: local            │ │
│  │             │ │ Working Dir: /home/user    │ │
│  │             │ │ Available Tools: 19        │ │
│  │             │ │ Available Skills: 12       │ │
│  └─────────────┘ └────────────────────────────┘ │
└─────────────────────────────────────────────────┘
│ Conversation output scrolls here...             │
│                                                 │
│ User: Hello!                                    │
│ ────────────────────────────────────────────── │
│   (◕‿◕✿) 🧠 pondering... (2.3s)                │
│   ✧٩(ˊᗜˋ*)و✧ got it! (2.3s)                    │
│                                                 │
│ Assistant: Hello! How can I help you today?    │
├─────────────────────────────────────────────────┤
│ ❯ [Fixed input area at bottom]                  │
└─────────────────────────────────────────────────┘
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List available tools grouped by toolset |
| `/toolsets` | List available toolsets with descriptions |
| `/model [name]` | Show or change the current model |
| `/prompt [text]` | View/set/clear custom system prompt |
| `/personality [name]` | Set a predefined personality |
| `/clear` | Clear screen and reset conversation |
| `/reset` | Reset conversation only (keep screen) |
| `/history` | Show conversation history |
| `/save` | Save current conversation to file |
| `/config` | Show current configuration |
| `/quit` | Exit the CLI (also: `/exit`, `/q`) |

## Configuration

The CLI is configured via `cli-config.yaml`. Copy from `cli-config.yaml.example`:

```bash
cp cli-config.yaml.example cli-config.yaml
```

### Model & Provider Configuration

```yaml
model:
  default: "anthropic/-opus-4.6"
  base_url: "https://openrouter.ai/api/v1"
  provider: "auto"  # "auto" | "openrouter" | "nous"
```

**Provider selection** (`provider` field):
- `auto` (default): Uses Nous Portal if logged in (`hermes login`), otherwise falls back to OpenRouter/env vars.
- `openrouter`: Always uses `OPENROUTER_API_KEY` from `.env`.
- `nous`: Always uses Nous Portal OAuth credentials from `auth.json`.

Can also be overridden per-session with `--provider` or via `HERMES_INFERENCE_PROVIDER` env var.

### Terminal Configuration

The CLI supports multiple terminal backends:

```yaml
# Local execution (default)
terminal:
  env_type: "local"
  cwd: "."  # Current directory

# SSH remote execution (sandboxed - agent can't touch its own code)
terminal:
  env_type: "ssh"
  cwd: "/home/myuser/project"
  ssh_host: "my-server.example.com"
  ssh_user: "myuser"
  ssh_key: "~/.ssh/id_rsa"

# Docker container
terminal:
  env_type: "docker"
  docker_image: "python:3.11"

# Singularity/Apptainer (HPC)
terminal:
  env_type: "singularity"
  singularity_image: "docker://python:3.11"

# Modal cloud
terminal:
  env_type: "modal"
  modal_image: "python:3.11"
```

### Sudo Support

The CLI supports interactive sudo prompts:

```
┌──────────────────────────────────────────────────────────┐
│  🔐 SUDO PASSWORD REQUIRED                               │
├──────────────────────────────────────────────────────────┤
│  Enter password below (input is hidden), or:             │
│    • Press Enter to skip (command fails gracefully)      │
│    • Wait 45s to auto-skip                               │
└──────────────────────────────────────────────────────────┘

  Password (hidden): 
```

**Options:**
- **Interactive**: Leave `sudo_password` unset - you'll be prompted when needed
- **Configured**: Set `sudo_password` in `cli-config.yaml` to auto-fill
- **Environment**: Set `SUDO_PASSWORD` in `.env` for all runs

Password is cached for the session once entered.

### Toolsets

Control which tools are available:

```yaml
# Enable all tools
toolsets:
  - all

# Or enable specific toolsets
toolsets:
  - web
  - terminal
  - skills
```

Available toolsets: `web`, `search`, `terminal`, `browser`, `vision`, `image_gen`, `skills`, `moa`, `debugging`, `safe`

### Personalities

Predefined personalities for the `/personality` command:

```yaml
agent:
  personalities:
    helpful: "You are a helpful, friendly AI assistant."
    kawaii: "You are a kawaii assistant! Use cute expressions..."
    pirate: "Arrr! Ye be talkin' to Captain Hermes..."
    # Add your own!
```

Built-in personalities:
- `helpful`, `concise`, `technical`, `creative`, `teacher`
- `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`
- `noir`, `uwu`, `philosopher`, `hype`

## Animated Feedback

The CLI provides animated feedback during operations:

### Thinking Animation

During API calls, shows animated spinner with thinking verbs:
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

### Tool Execution Animation

Each tool type has unique animations:
```
  ⠋ (◕‿◕✿) 🔍 web_search... (0.8s)
  ▅ (≧◡≦) 💻 terminal... (1.2s)
  🌓 (★ω★) 🌐 browser_navigate... (2.1s)
  ✧ (✿◠‿◠) 🎨 image_generate... (4.5s)
```

## Multi-line Input

For multi-line input, end a line with `\` to continue:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

## Environment Variable Priority

For terminal settings, `cli-config.yaml` takes precedence over `.env`:

1. `cli-config.yaml` (highest priority in CLI)
2. `.env` file
3. System environment variables
4. Default values

This allows you to have different terminal configs for CLI vs batch processing.

## Session Management

- **History**: Command history is saved to `~/.hermes_history`
- **Conversations**: Use `/save` to export conversations
- **Reset**: Use `/clear` for full reset, `/reset` to just clear history
- **Session Logs**: Every session automatically logs to `logs/session_{session_id}.json`
- **Resume**: Pick up any previous session with `--resume` or `--continue`

### Resuming Sessions

When you exit a CLI session, a resume command is printed:

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

To resume:

```bash
hermes --continue                          # Resume the most recent CLI session
hermes -c                                  # Short form
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes -r 20260225_143052_a1b2c3           # Short form
hermes chat --resume 20260225_143052_a1b2c3  # Explicit subcommand form
```

Resuming restores the full conversation history from SQLite (`~/.hermes/state.db`). The agent sees all previous messages, tool calls, and responses — just as if you never left. New messages append to the same session in the database.

Use `hermes sessions list` to browse past sessions and find IDs.

### Session Logging

Sessions are automatically logged to the `logs/` directory:

```
logs/
├── session_20260201_143052_a1b2c3.json
├── session_20260201_150217_d4e5f6.json
└── ...
```

The session ID is displayed in the welcome banner and follows the format: `YYYYMMDD_HHMMSS_UUID`.

Log files contain:
- Full conversation history in trajectory format
- Timestamps for session start and last update
- Model and message count metadata

This is useful for:
- Debugging agent behavior
- Replaying conversations
- Training data inspection

### Context Compression

Long conversations can exceed model context limits. The CLI automatically compresses context when approaching the limit:

```yaml
# In cli-config.yaml
compression:
  enabled: true                    # Enable auto-compression
  threshold: 0.85                  # Compress at 85% of context limit  
  summary_model: "google/gemini-2.0-flash-001"
```

**How it works:**
1. Tracks actual token usage from each API response
2. When tokens reach threshold, middle turns are summarized
3. First 3 and last 4 turns are always protected
4. Conversation continues seamlessly after compression

**When compression triggers:**
```
📦 Context compression triggered (170,000 tokens ≥ 170,000 threshold)
   📊 Model context limit: 200,000 tokens (85% = 170,000)
   🗜️  Summarizing turns 4-15 (12 turns)
   ✅ Compressed: 20 → 9 messages (~45,000 tokens saved)
```

To disable compression:
```yaml
compression:
  enabled: false
```

## Quiet Mode

The CLI runs in "quiet mode" (`HERMES_QUIET=1`), which:
- Suppresses verbose logging from tools
- Enables kawaii-style animated feedback
- Hides terminal environment warnings
- Keeps output clean and user-friendly

For verbose output (debugging), use:
```bash
./hermes --verbose
```

## Skills Hub Commands

The Skills Hub provides search, install, and management of skills from online registries.

**Terminal commands:**
```bash
hermes skills search <query>                      # Search all registries
hermes skills search <query> --source github      # Search GitHub only
hermes skills install <identifier>                # Install with security scan
hermes skills install <id> --category devops      # Install into a category
hermes skills install <id> --force                # Override caution block
hermes skills inspect <identifier>                # Preview without installing
hermes skills list                                # List all installed skills
hermes skills list --source hub                   # Hub-installed only
hermes skills audit                               # Re-scan all hub skills
hermes skills audit <name>                        # Re-scan a specific skill
hermes skills uninstall <name>                    # Remove a hub skill
hermes skills publish <path> --to github --repo owner/repo
hermes skills snapshot export <file.json>         # Export skill config
hermes skills snapshot import <file.json>         # Re-install from snapshot
hermes skills tap list                            # List custom sources
hermes skills tap add owner/repo                  # Add a GitHub repo source
hermes skills tap remove owner/repo               # Remove a source
```

**Slash commands (inside chat):**

All the same commands work with `/skills` prefix:
```
/skills search kubernetes
/skills install openai/skills/skill-creator
/skills list
/skills tap add myorg/skills
```
