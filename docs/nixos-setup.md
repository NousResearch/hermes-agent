# Nix Setup Guide for Hermes Agent

## Prerequisites

- Nix with flakes enabled (we recommend [Determinate Nix](https://install.determinate.systems) which enables flakes by default)
- API keys for the services you want to use (at minimum: OpenRouter)

## Quick Start: `nix run`

```bash
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- chat
```

No clone needed. Nix fetches and builds everything.

## Install

All Python dependencies are pre-built as Nix derivations via uv2nix — no runtime pip install.

```bash
# From a clone
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix build
./result/bin/hermes setup
./result/bin/hermes chat

# Or install into your profile
nix profile install github:NousResearch/hermes-agent
hermes setup
```

## Development (`nix develop`)

For hacking on hermes-agent locally:

```bash
cd hermes-agent
nix develop
# Shell automatically:
#   - Creates .venv with Python 3.11
#   - Installs all Python deps via uv
#   - Installs npm deps (agent-browser)
#   - Puts ripgrep, git, node on PATH

hermes setup
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

## Persistent Messaging Gateway

To run hermes-agent as a **always-on service** for Telegram, Discord, or Slack (with built-in cron scheduler), use the [home-manager](https://github.com/nix-community/home-manager) module. This is how messaging platforms are meant to be run — the gateway stays up, receives messages, and responds. Works on any Linux distribution with Nix, not just NixOS.

> This assumes you already have home-manager set up. If you don't, see the [home-manager docs](https://nix-community.github.io/home-manager/) first.

### Step 1: Add the flake input

```nix
# ~/.config/home-manager/flake.nix (or wherever your HM flake lives)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hermes-agent.url = "github:NousResearch/hermes-agent";
  };

  outputs = { nixpkgs, home-manager, hermes-agent, ... }: {
    homeConfigurations."your-username" = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      modules = [
        hermes-agent.homeManagerModules.default
        ./home.nix
      ];
    };
  };
}
```

### Step 2: Enable in home.nix

```nix
# home.nix
{
  services.hermes-agent = {
    enable = true;
    gateway.enable = true;

    # All options with defaults:
    # hermesHome = "~/.hermes";           # config, sessions, memories
    # environmentFile = "~/.hermes/.env"; # API keys
    # messagingCwd = "~";                 # gateway working directory
    # addToPATH = true;                   # adds `hermes` CLI to PATH
  };
}
```

### Step 3: API keys

```bash
hermes setup
```

The interactive wizard walks you through configuring API keys (OpenRouter, Telegram, Discord, etc.) and writes them to `~/.hermes/.env`.

### Step 4: Enable linger + activate

```bash
# Lets user services survive logout
sudo loginctl enable-linger $USER

# Activate — all deps are pre-built, this is fast
home-manager switch
```

### Step 5: Verify

```bash
systemctl --user status hermes-agent-gateway
journalctl --user -u hermes-agent-gateway -f
hermes doctor
```

## Directory layout

```
~/.hermes/                           # Config & data
├── .env                             # API keys
├── config.yaml                      # Agent configuration
├── sessions/                        # Messaging sessions
├── memories/                        # Agent memories
├── skills/                          # Knowledge documents
├── cron/                            # Scheduled jobs
└── logs/                            # Session logs
```

## Customizing

```bash
$EDITOR ~/.hermes/config.yaml
```

Key settings:
- `model.default` — Which LLM to use (default: `anthropic/claude-opus-4.6`)
- `terminal.env_type` — Terminal backend: `local`, `docker`, `ssh`
- `toolsets` — Which tools to enable (default: all)

After editing, restart the service:

```bash
systemctl --user restart hermes-agent-gateway
```

## Updating

```bash
nix flake update hermes-agent --flake ~/.config/home-manager
home-manager switch
```

## Troubleshooting

```bash
# Gateway logs
journalctl --user -u hermes-agent-gateway -f

# Check CLI + deps
hermes doctor

# Restart gateway
systemctl --user restart hermes-agent-gateway

# Full rebuild (if something is really wrong)
nix build --rebuild
```
