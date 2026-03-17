# CLI Reference

## Installation

```bash
curl -fsSL https://cli.inference.sh | sh
```

## Global Commands

| Command | Description |
|---------|-------------|
| `infsh help` | Show help |
| `infsh version` | Show CLI version |
| `infsh update` | Update CLI to latest |
| `infsh login` | Authenticate |
| `infsh me` | Show current user |

## App Commands

### Discovery

| Command | Description |
|---------|-------------|
| `infsh app list` | List available apps |
| `infsh app list --category <cat>` | Filter by category (image, video, audio, text, other) |
| `infsh app search <query>` | Search apps |
| `infsh app list --search <query>` | Search apps (flag form) |
| `infsh app list --featured` | Show featured apps |
| `infsh app list --new` | Sort by newest |
| `infsh app list --page <n>` | Pagination |
| `infsh app list -l` | Detailed table view |
| `infsh app list --save <file>` | Save to JSON file |
| `infsh app my` | List your deployed apps |
| `infsh app get <app>` | Get app details |
| `infsh app get <app> --json` | Get app details as JSON |

### Execution

| Command | Description |
|---------|-------------|
| `infsh app run <app> --input <file>` | Run app with input file |
| `infsh app run <app> --input '<json>'` | Run with inline JSON |
| `infsh app run <app> --input <file> --no-wait` | Run without waiting for completion |
| `infsh app sample <app>` | Show sample input |
| `infsh app sample <app> --save <file>` | Save sample to file |

## Task Commands

| Command | Description |
|---------|-------------|
| `infsh task get <task-id>` | Get task status and result |
| `infsh task get <task-id> --json` | Get task as JSON |
| `infsh task get <task-id> --save <file>` | Save task result to file |

### Development

| Command | Description |
|---------|-------------|
| `infsh app init` | Create new app (interactive) |
| `infsh app init <name>` | Create new app with name |
| `infsh app test --input <file>` | Test app locally |
| `infsh app deploy` | Deploy app |
| `infsh app deploy --dry-run` | Validate without deploying |
| `infsh app pull <id>` | Pull app source |
| `infsh app pull --all` | Pull all your apps |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `INFSH_API_KEY` | API key (overrides config) |

## Shell Completions

```bash
# Bash
infsh completion bash > /etc/bash_completion.d/infsh

# Zsh
infsh completion zsh > "${fpath[1]}/_infsh"

# Fish
infsh completion fish > ~/.config/fish/completions/infsh.fish
```

## App Name Format

Apps use the format `namespace/app-name`:

- `falai/flux-dev-lora` - fal.ai's FLUX 2 Dev
- `google/veo-3` - Google's Veo 3
- `infsh/sdxl` - inference.sh's SDXL
- `bytedance/seedance-1-5-pro` - ByteDance's Seedance
- `xai/grok-imagine-image` - xAI's Grok

Version pinning: `namespace/app-name@version`

## Documentation

- [CLI Setup](https://inference.sh/docs/extend/cli-setup) - Complete CLI installation guide
- [Running Apps](https://inference.sh/docs/apps/running) - How to run apps via CLI
- [Creating an App](https://inference.sh/docs/extend/creating-app) - Build your own apps
- [Deploying](https://inference.sh/docs/extend/deploying) - Deploy apps to the cloud
