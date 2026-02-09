# NixOS Setup Guide for Hermes Agent

## Prerequisites

- NixOS with flakes enabled (`nix.settings.experimental-features = [ "nix-command" "flakes" ];`)
- Internet access (for initial Python/npm dependency installation)
- API keys for the services you want to use (at minimum: OpenRouter)

## Option 1: Development (nix develop)

For hacking on hermes-agent locally:

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix develop
# Shell automatically:
#   - Creates .venv with Python 3.11
#   - Installs all Python deps via uv
#   - Installs npm deps (agent-browser)
#   - Puts ripgrep, git, node on PATH

# Configure your API keys
hermes setup

# Start chatting
hermes
```

### Using direnv (recommended)

If you have [direnv](https://direnv.net/) installed, the included `.envrc` will
automatically activate the dev shell when you `cd` into the repo:

```bash
cd hermes-agent
direnv allow    # one-time approval

# From now on, entering the directory activates the environment automatically.
# On repeat entry, the stamp file check skips dependency installation (~instant).
```

## Option 2: Server Deployment (NixOS Module)

For running Hermes Agent as persistent services (Telegram/Discord bot, cron scheduler).

### Step 1: Add the flake input to your NixOS configuration

```nix
# flake.nix (your system flake)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    hermes-agent = {
      url = "github:NousResearch/hermes-agent";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, hermes-agent, ... }: {
    nixosConfigurations.my-server = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        hermes-agent.nixosModules.default
      ];
    };
  };
}
```

### Step 2: Create your secrets file

Store API keys in a file that will be loaded as environment variables. Use your preferred secrets manager (sops-nix, agenix, or a plain file with restricted permissions).

```bash
# /etc/hermes/secrets.env (chmod 600, owned by root)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx

# Telegram bot (get token from @BotFather, get user ID from @userinfobot)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789

# Discord bot (get from Discord Developer Portal)
# DISCORD_BOT_TOKEN=MTIz...
# DISCORD_ALLOWED_USERS=123456789012345678

# Optional tool API keys (tools auto-disable if key is missing)
# FIRECRAWL_API_KEY=fc-xxxxxxxxxxxxx
# BROWSERBASE_API_KEY=bb-xxxxxxxxxxxxx
# BROWSERBASE_PROJECT_ID=proj-xxxxxxxxxxxxx
# FAL_KEY=xxxxxxxxxxxxx
# NOUS_API_KEY=xxxxxxxxxxxxx
```

### Step 3: Configure the NixOS module

```nix
# configuration.nix
{ config, pkgs, ... }:
{
  services.hermes-agent = {
    enable = true;

    # Path to your secrets file
    environmentFile = "/etc/hermes/secrets.env";

    # Enable the messaging gateway (Telegram/Discord/WhatsApp)
    gateway.enable = true;

    # Enable the cron scheduler for recurring tasks
    cron.enable = true;

    # Optional overrides:
    # user = "hermes";               # default
    # group = "hermes";              # default
    # stateDir = "/var/lib/hermes";  # default
  };
}
```

### Step 4: Deploy

```bash
sudo nixos-rebuild switch
```

### Step 5: Verify

```bash
# Check setup completed (runs on first boot, installs Python/npm deps)
systemctl status hermes-agent-setup

# Check gateway is running
systemctl status hermes-agent-gateway
journalctl -u hermes-agent-gateway -f

# Check cron daemon
systemctl status hermes-agent-cron

# Test: send a message to your Telegram bot
```

## What happens under the hood

1. **hermes-agent-setup.service** (runs once, re-runs on package updates):
   - Copies the hermes-agent source to `/var/lib/hermes/app/`
   - Creates a Python 3.11 venv at `/var/lib/hermes/app/.venv`
   - Installs all Python dependencies via `uv pip install`
   - Installs npm dependencies (agent-browser for browser automation)
   - Creates the `~/.hermes/` config structure
   - Has `TimeoutStartSec = 300` to allow for slow network installs

2. **hermes-agent-gateway.service** (persistent):
   - Runs the messaging gateway (Telegram, Discord, WhatsApp)
   - Auto-restarts on failure
   - Reads API keys from your `environmentFile`
   - Sets `MESSAGING_CWD` to the state directory

3. **hermes-agent-cron.service** (persistent):
   - Checks for due scheduled tasks every 60 seconds
   - Manages cron jobs stored in `/var/lib/hermes/.hermes/cron/`

## Directory Layout on NixOS

```
/var/lib/hermes/                  # State directory (writable)
├── app/                          # Source tree copy
│   ├── .venv/                    # Python virtual environment
│   ├── node_modules/             # npm dependencies
│   ├── skills/                   # Knowledge documents
│   ├── mini-swe-agent/           # Terminal backend
│   ├── tinker-atropos/           # RL training
│   └── ...                       # Full source tree
└── .hermes/                      # User config & data
    ├── config.yaml               # Agent configuration
    ├── sessions/                 # Messaging sessions
    ├── cron/                     # Scheduled jobs
    │   ├── jobs.json
    │   └── output/
    └── logs/                     # Session logs
```

## Customizing the Agent

Edit the config file on the server:

```bash
sudo -u hermes nano /var/lib/hermes/.hermes/config.yaml
```

Key settings:
- `model.default` — Which LLM to use (default: `anthropic/claude-opus-4.6`)
- `terminal.env_type` — Terminal backend: `local`, `docker`, `ssh`
- `toolsets` — Which tools to enable (default: all)
- `compression` — Context window management settings
- `agent.max_turns` — Max tool-calling iterations per conversation

After editing, restart the services:

```bash
sudo systemctl restart hermes-agent-gateway
```

## Updating

When a new version of hermes-agent is released:

```bash
# Update the flake input
nix flake update hermes-agent

# Rebuild
sudo nixos-rebuild switch

# The setup service will automatically re-run and update dependencies
```

## Troubleshooting

```bash
# View gateway logs
journalctl -u hermes-agent-gateway -f

# View cron logs
journalctl -u hermes-agent-cron -f

# View setup logs (dep installation)
journalctl -u hermes-agent-setup

# Run hermes doctor interactively as the service user (with correct env vars)
sudo -u hermes bash -c 'source /var/lib/hermes/app/.venv/bin/activate && HERMES_HOME=/var/lib/hermes/.hermes hermes doctor'

# Check which tools are available
sudo -u hermes bash -c 'source /var/lib/hermes/app/.venv/bin/activate && HERMES_HOME=/var/lib/hermes/.hermes hermes status'
```

## Using with sops-nix (recommended for secrets)

```nix
{
  sops.secrets."hermes-agent-env" = {
    sopsFile = ./secrets/hermes.yaml;
    format = "yaml";
  };

  services.hermes-agent = {
    enable = true;
    environmentFile = config.sops.secrets."hermes-agent-env".path;
    gateway.enable = true;
  };
}
```

## Using with agenix

```nix
{
  age.secrets.hermes-env.file = ./secrets/hermes.env.age;

  services.hermes-agent = {
    enable = true;
    environmentFile = config.age.secrets.hermes-env.path;
    gateway.enable = true;
  };
}
```
