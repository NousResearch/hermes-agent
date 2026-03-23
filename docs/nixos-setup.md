# Nix Setup Guide for Hermes Agent

## Prerequisites

- Nix with flakes enabled ([Determinate Nix](https://install.determinate.systems) recommended — enables flakes by default)
- API keys for the services you want to use (at minimum: an OpenRouter or Anthropic key)

## Quick Start: `nix run`

```bash
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- chat
```

No clone needed. Nix fetches and builds everything. All Python dependencies are
Nix derivations via uv2nix — no runtime pip.

## Install to Profile

```bash
# From remote
nix profile install github:NousResearch/hermes-agent
hermes setup

# From a clone
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix build
./result/bin/hermes setup
```

## Development Shell

```bash
cd hermes-agent
nix develop
# Shell provides:
#   - Python 3.11 venv with all deps (via uv)
#   - npm deps (agent-browser)
#   - ripgrep, git, node on PATH

hermes setup
hermes chat
```

### direnv (recommended)

The included `.envrc` activates the dev shell automatically:

```bash
cd hermes-agent
direnv allow    # one-time
# Subsequent entries are near-instant (stamp file skips dep install)
```

---

## NixOS Module

The flake exports `nixosModules.default` with two deployment modes:

| Mode | `container.enable` | How it runs | Use case |
|---|---|---|---|
| **Native** (default) | `false` | Hardened systemd service, runs directly on host | Standard deployment, maximum security |
| **Container** | `true` | Persistent Ubuntu container, hermes binary bind-mounted from `/nix/store` | Agent needs `apt`/`pip`/`npm` self-install capability |

Both modes share the same option surface. The module manages user creation,
directory setup, config generation, secrets, documents, and service lifecycle.

> **Note:** This module requires NixOS. For non-NixOS systems, use
> `nix profile install` + the CLI's built-in `hermes gateway install`.

### Add the Flake Input

```nix
# /etc/nixos/flake.nix (or your system flake)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    hermes-agent.url = "github:NousResearch/hermes-agent";
  };

  outputs = { nixpkgs, hermes-agent, ... }: {
    nixosConfigurations.your-host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        hermes-agent.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

### Minimal Configuration (Native Mode)

```nix
# configuration.nix
{
  services.hermes-agent = {
    enable = true;
    settings.model.default = "anthropic/claude-sonnet-4";
    environmentFiles = [ config.sops.secrets."hermes/env".path ];
    addToSystemPackages = true; # puts `hermes` CLI on PATH
  };
}
```

> **Important:** `addToSystemPackages = true` also sets `HERMES_HOME` system-wide
> so the interactive CLI shares state (sessions, skills, cron) with the gateway
> service. Without it, running `hermes` in your shell creates a separate
> `~/.hermes` directory.

### Minimal Configuration (Container Mode)

```nix
{
  services.hermes-agent = {
    enable = true;
    container.enable = true;
    settings.model.default = "anthropic/claude-sonnet-4";
    environmentFiles = [ config.sops.secrets."hermes/env".path ];
    addToSystemPackages = true;
  };
}
```

> Container mode auto-enables `virtualisation.docker.enable` via `mkDefault`.
> Override with `virtualisation.docker.enable = false;` if using podman
> (`container.backend = "podman"`).

### Full Example

```nix
{ config, ... }: {
  services.hermes-agent = {
    enable = true;
    container.enable = true;

    # ── Model ──────────────────────────────────────────────────────────
    settings = {
      model = {
        base_url = "https://openrouter.ai/api/v1";
        default = "anthropic/claude-opus-4.6";
      };
      toolsets = [ "all" ];
      max_turns = 100;
      terminal = { backend = "local"; cwd = "."; timeout = 180; };
      compression = {
        enabled = true;
        threshold = 0.85;
        summary_model = "google/gemini-3-flash-preview";
      };
      memory = { memory_enabled = true; user_profile_enabled = true; };
      display = { compact = false; personality = "kawaii"; };
      agent = { max_turns = 60; verbose = false; };
    };

    # ── Secrets ────────────────────────────────────────────────────────
    # See "Secrets Management" section below
    environmentFiles = [ config.sops.secrets."hermes/env".path ];

    # ── Documents ──────────────────────────────────────────────────────
    documents = {
      "SOUL.md" = builtins.readFile /home/user/.hermes/SOUL.md;
      "USER.md" = ./documents/USER.md;  # path reference
    };

    # ── MCP Servers ────────────────────────────────────────────────────
    mcpServers.filesystem = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-filesystem" "/data/workspace" ];
    };

    # ── Container options ──────────────────────────────────────────────
    container = {
      image = "ubuntu:24.04";           # default
      backend = "docker";               # or "podman"
      extraVolumes = [
        "/home/user/projects:/projects:rw"
      ];
      extraOptions = [
        "--gpus" "all"                   # GPU passthrough
      ];
    };

    # ── Service tuning ─────────────────────────────────────────────────
    addToSystemPackages = true;
    extraArgs = [ "--verbose" ];
    restart = "always";
    restartSec = 5;
  };
}
```

---

## Secrets Management

**Never put API keys in `environment`** — values end up in the Nix store
(world-readable). Use `environmentFiles` with a secrets manager.

Both `environment` and `environmentFiles` are merged into `$HERMES_HOME/.env`
at activation time (`nixos-rebuild switch`). Hermes reads this file on every
startup via `load_hermes_dotenv()`, so changes take effect on
`systemctl restart hermes-agent` — no container recreation needed.

### sops-nix

```nix
{
  sops = {
    defaultSopsFile = ./secrets/hermes.yaml;
    age.keyFile = "/home/user/.config/sops/age/keys.txt";
    secrets."hermes-env" = { format = "yaml"; };
  };

  services.hermes-agent.environmentFiles = [
    config.sops.secrets."hermes-env".path
  ];
}
```

The secrets file should contain key-value pairs:

```yaml
# secrets/hermes.yaml (encrypted with sops)
hermes-env: |
    OPENROUTER_API_KEY=sk-or-...
    TELEGRAM_BOT_TOKEN=123456:ABC...
    ANTHROPIC_API_KEY=sk-ant-...
```

### agenix

```nix
{
  age.secrets.hermes-env.file = ./secrets/hermes-env.age;

  services.hermes-agent.environmentFiles = [
    config.age.secrets.hermes-env.path
  ];
}
```

### OAuth / Auth Seeding

For platforms requiring OAuth (e.g., Discord), use `authFile` to seed
credentials on first deploy:

```nix
{
  services.hermes-agent = {
    authFile = config.sops.secrets."hermes/auth.json".path;
    # authFileForceOverwrite = true;  # overwrite on every activation
  };
}
```

The file is only copied if `auth.json` doesn't already exist (unless
`authFileForceOverwrite = true`). Runtime OAuth token refreshes are
written back to the state directory and preserved across rebuilds.

---

## Container Mode: Architecture

When `container.enable = true`, hermes runs inside a persistent Ubuntu
container with the Nix-built binary bind-mounted from the host:

```
Host                                    Container
────                                    ─────────
/nix/store/...-hermes-agent-0.1.0  ──►  /nix/store/... (ro)
/var/lib/hermes/                    ──►  /data/          (rw)
  ├── current-package -> /nix/store/...    (symlink, updated each rebuild)
  ├── .gc-root -> /nix/store/...           (prevents nix-collect-garbage)
  ├── .container-identity                  (sha256 hash, triggers recreation)
  ├── .hermes/                             (HERMES_HOME)
  │   ├── .env                             (merged from environment + environmentFiles)
  │   ├── config.yaml                      (Nix-generated, copied by activation)
  │   ├── .managed                         (marker file)
  │   ├── state.db
  │   ├── sessions/
  │   ├── memories/
  │   └── ...
  └── workspace/                           (MESSAGING_CWD)
      ├── SOUL.md                          (from documents option)
      └── (agent-created files)

Container writable layer (apt/pip/npm):   /usr, /home/hermes, /tmp
```

The container entrypoint is `/data/current-package/bin/hermes gateway run --replace`,
which resolves through the symlink to the current Nix store path.

### What Persists Across What

| Event | Container recreated? | `/data` (state) | Writable layer (`apt`/`pip`/`npm`) |
|---|---|---|---|
| `systemctl restart hermes-agent` | No | Persists | Persists |
| `nixos-rebuild switch` (code change) | No (symlink updated) | Persists | Persists |
| Host reboot | No | Persists | Persists |
| `nix-collect-garbage` | No (GC root) | Persists | Persists |
| Image change (`container.image`) | **Yes** | Persists | **Lost** |
| Volume/options change | **Yes** | Persists | **Lost** |
| `environment`/`environmentFiles` change | No | Persists | Persists |

The container is only recreated when its **identity hash** changes. The hash
covers: `image`, `extraVolumes`, `extraOptions`. Changes to `environment`,
`environmentFiles`, `settings`, `documents`, or the hermes package itself do
**not** trigger recreation — environment variables are written to
`$HERMES_HOME/.env` by the activation script and read by hermes at startup.
A `systemctl restart hermes-agent` is sufficient for env changes.

### When to Use Container Mode

Use container mode when:
- The agent needs to `apt install`, `pip install`, or `npm install` packages at runtime
- You want the agent to have a mutable Linux environment it can customize
- You're running untrusted or experimental tool configurations

Use native mode when:
- You want maximum security (systemd hardening: `NoNewPrivileges`, `ProtectSystem=strict`)
- The agent only needs tools already on the Nix-provided PATH
- You prefer a minimal, reproducible deployment

---

## Managed Mode

When hermes runs via the NixOS module, the following CLI commands are
**blocked** with a descriptive error:

| Blocked command | Reason |
|---|---|
| `hermes setup` | Config is declarative in `configuration.nix` |
| `hermes config edit` | Config is generated from `settings` |
| `hermes config set <key> <value>` | Config is generated from `settings` |
| `hermes gateway install` | Service is managed by NixOS |
| `hermes gateway uninstall` | Service is managed by NixOS |

Detection uses two signals:
1. `HERMES_MANAGED=true` environment variable (set by the systemd service)
2. `.managed` marker file in `HERMES_HOME` (set by the activation script,
   visible to interactive shells)

If you need to change configuration, edit your `configuration.nix` and run
`sudo nixos-rebuild switch`.

---

## Options Reference

### Core

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `false` | Enable the hermes-agent service |
| `package` | `package` | `hermes-agent` | The hermes-agent package |
| `user` | `str` | `"hermes"` | System user |
| `group` | `str` | `"hermes"` | System group |
| `createUser` | `bool` | `true` | Auto-create user/group |
| `stateDir` | `str` | `"/var/lib/hermes"` | State directory (`HERMES_HOME` parent) |
| `workingDirectory` | `str` | `"${stateDir}/workspace"` | Agent working directory (`MESSAGING_CWD`) |
| `addToSystemPackages` | `bool` | `false` | Add `hermes` CLI to system PATH and set `HERMES_HOME` system-wide so CLI and gateway share state |

### Configuration

| Option | Type | Default | Description |
|---|---|---|---|
| `settings` | `attrs` (deep-merged) | `{}` | Declarative config rendered as `config.yaml`. Supports arbitrary nesting; multiple definitions are merged via `lib.recursiveUpdate` |
| `configFile` | `null` or `path` | `null` | Path to an existing `config.yaml`. Overrides `settings` entirely if set |

### Secrets & Environment

| Option | Type | Default | Description |
|---|---|---|---|
| `environmentFiles` | `listOf str` | `[]` | Paths to env files with secrets (API keys). Passed as systemd `EnvironmentFile=` or docker `--env-file` |
| `environment` | `attrsOf str` | `{}` | Non-secret env vars. **Visible in Nix store** — do not put secrets here |
| `authFile` | `null` or `path` | `null` | OAuth credentials seed. Only copied on first deploy |
| `authFileForceOverwrite` | `bool` | `false` | Always overwrite `auth.json` from `authFile` |

### Documents

| Option | Type | Default | Description |
|---|---|---|---|
| `documents` | `attrsOf (either str path)` | `{}` | Workspace files. Keys are filenames, values are inline strings or paths. Installed into `workingDirectory` on activation |

### MCP Servers

| Option | Type | Default | Description |
|---|---|---|---|
| `mcpServers` | `attrsOf submodule` | `{}` | MCP server definitions, merged into `settings.mcp_servers` |
| `mcpServers.<name>.command` | `str` | — | Server command |
| `mcpServers.<name>.args` | `listOf str` | `[]` | Command arguments |
| `mcpServers.<name>.env` | `attrsOf str` | `{}` | Server environment |
| `mcpServers.<name>.timeout` | `null` or `int` | `null` | Timeout in seconds |

### Service Behavior

| Option | Type | Default | Description |
|---|---|---|---|
| `extraArgs` | `listOf str` | `[]` | Extra args for `hermes gateway` |
| `extraPackages` | `listOf package` | `[]` | Extra packages on service PATH (native mode only) |
| `restart` | `str` | `"always"` | systemd `Restart=` policy |
| `restartSec` | `int` | `5` | systemd `RestartSec=` |

### Container

| Option | Type | Default | Description |
|---|---|---|---|
| `container.enable` | `bool` | `false` | Enable OCI container mode |
| `container.backend` | `enum ["docker" "podman"]` | `"docker"` | Container runtime. Auto-enables `virtualisation.docker.enable` when `"docker"` |
| `container.image` | `str` | `"ubuntu:24.04"` | Base image. Pulled at runtime by Docker/Podman |
| `container.extraVolumes` | `listOf str` | `[]` | Extra volume mounts (`host:container:mode`) |
| `container.extraOptions` | `listOf str` | `[]` | Extra args passed to `docker create` |

---

## Directory Layout

### Native Mode

```
/var/lib/hermes/                     # stateDir (owned by hermes:hermes, 0750)
├── .hermes/                         # HERMES_HOME
│   ├── config.yaml                  # Nix-generated (overwritten each rebuild)
│   ├── .managed                     # Marker: CLI config mutation blocked
│   ├── .env                         # (not used — secrets via environmentFiles)
│   ├── auth.json                    # OAuth credentials (seeded, then self-managed)
│   ├── gateway.pid
│   ├── state.db
│   ├── sessions/
│   ├── memories/
│   ├── skills/
│   ├── cron/
│   └── logs/
└── workspace/                       # MESSAGING_CWD
    ├── SOUL.md                      # From documents option
    └── (agent-created files)
```

### Container Mode

Same layout, but mounted into the container as `/data`:

| Container path | Host path | Mode | Notes |
|---|---|---|---|
| `/nix/store` | `/nix/store` | `ro` | Hermes binary + all Nix deps |
| `/data` | `/var/lib/hermes` | `rw` | All state, config, workspace |
| `/home/hermes` | (container layer) | `rw` | Persists — agent home, `pip install --user` |
| `/usr`, `/usr/local` | (container layer) | `rw` | Persists — `apt`/`pip`/`npm` installs |
| `/tmp` | (container layer) | `rw` | Persists across restarts (lost on recreation) |

---

## Updating

```bash
# Update the flake input
nix flake update hermes-agent --flake /etc/nixos

# Rebuild — in container mode, the symlink updates without recreating the container
sudo nixos-rebuild switch
```

In container mode, the agent picks up the new binary immediately on restart.
No container recreation, no loss of `apt`/`pip`/`npm` installs.

## Flake Checks

The flake includes build-time verification:

```bash
# Run all checks
nix flake check

# Individual checks
nix build .#checks.x86_64-linux.package-contents   # binaries exist + version
nix build .#checks.x86_64-linux.cli-commands        # gateway/config subcommands
nix build .#checks.x86_64-linux.managed-guard       # HERMES_MANAGED blocks mutation
```

## Troubleshooting

### Service logs

```bash
# Native mode
journalctl -u hermes-agent -f

# Container mode — same unit name
journalctl -u hermes-agent -f

# Or directly from the container
docker logs -f hermes-agent
```

### Container inspection

```bash
# Service status
systemctl status hermes-agent

# Container state
docker ps -a --filter name=hermes-agent
docker inspect hermes-agent --format='{{.State.Status}}'

# Shell into the container
docker exec -it hermes-agent bash

# Check symlink
docker exec hermes-agent readlink /data/current-package

# Check identity hash
docker exec hermes-agent cat /data/.container-identity
```

### Force container recreation

If you need to reset the container writable layer (fresh Ubuntu):

```bash
sudo systemctl stop hermes-agent
docker rm -f hermes-agent
sudo rm /var/lib/hermes/.container-identity
sudo systemctl start hermes-agent
# Container will be recreated from scratch
```

### GC root verification

```bash
# Ensure the running package is protected
nix-store --query --roots $(docker exec hermes-agent readlink /data/current-package)
```

### Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot save configuration: managed by NixOS` | CLI guards active | Edit `configuration.nix` and `nixos-rebuild switch` |
| Container recreated unexpectedly | `extraVolumes`, `extraOptions`, or `image` changed | Expected behavior — writable layer is reset. Reinstall packages if needed |
| `hermes version` shows old version after rebuild | Container not restarted | `systemctl restart hermes-agent` |
| Permission denied on `/var/lib/hermes` | State dir is `0750 hermes:hermes` | Use `docker exec` or `sudo -u hermes` |
| `nix-collect-garbage` removed hermes | GC root missing or broken | Restart the service (`preStart` recreates the GC root) |
