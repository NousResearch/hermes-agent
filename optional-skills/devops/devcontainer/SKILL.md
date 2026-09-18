---
name: devcontainer
description: Scaffold and validate devcontainer.json configurations.
version: 1.0.0
author: Thomas Bale (TumCucTom)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [devcontainer, docker, vscode, codespaces, reproducible-dev, dev-environment]
    related_skills: [docker-management]
    category: devops
---

# Devcontainer

Scaffolds `.devcontainer/` directories for VS Code Dev Containers, GitHub Codespaces, and other devcontainer-aware editors. Emits a `devcontainer.json` (the open-standard config file described at [containers.dev](https://containers.dev/)) and an optional matching `Dockerfile`. Validates existing configurations against the schema for the most common foot-guns.

Use this when the user wants a reproducible development environment that ships with the repo — so contributors and CI both build the same way, and onboarding is a single click.

## When to Use

- User asks for a devcontainer, a `.devcontainer/` folder, or "VS Code dev container"
- User mentions GitHub Codespaces, Gitpod, or any cloud dev environment
- User wants to standardize the team's dev environment in-repo
- User wants to add a feature like Docker-in-Docker, AWS CLI, or Node to an existing dev container
- User asks "make this repo easy to onboard onto"

Don't use for:
- Production Dockerfiles / Docker images — see `docker-management` (devcontainers are for dev environments, not deploy artifacts)
- One-off disposable containers — `docker run` is enough
- CI containers — those are usually pinned in `.github/workflows/*.yml`, not devcontainer.json

## Prerequisites

A working Docker (or compatible) runtime is required to actually *use* the dev container, but the skill itself only needs Python — it writes config files and never invokes Docker. Editors that consume the config (VS Code with the Dev Containers extension, Codespaces, etc.) handle the rest.

```bash
# Verify Python
python --version
```

If the user also wants to run the dev container, they need:

```bash
docker --version
```

and one of:
- VS Code + Dev Containers extension
- JetBrains Gateway (for IDE backends)
- `devcontainer` CLI: `npm install -g @devcontainers/cli`

## How to Run

The skill writes files to disk. All work goes through the `terminal` tool to run the helper script, or `read_file` / `write_file` for direct edits.

- **Scaffold a fresh `.devcontainer/`**: `python scripts/init.py <target-dir> --python 3.11 --features docker-in-docker`
- **Add devcontainer.json to an existing repo**: `python scripts/init.py . --node 20`
- **Validate an existing config**: `python scripts/init.py . --validate`
- **Show what a generated config would look like (no writes)**: `python scripts/init.py . --python 3.11 --dry-run`

For direct edits to `devcontainer.json`, the schema lives at `https://containers.dev/schema/dev-container.json` — the comments in generated files point at the right field for each block.

## Quick Reference

| Flag | What it does |
|---|---|
| `--python <version>` | Pin Python (e.g. `3.11`, `3.12`). Adds a `python` feature and sets the image to `mcr.microsoft.com/devcontainers/python:<version>-bookworm`. |
| `--node <version>` | Pin Node (e.g. `20`, `22`). Adds a `node` feature. |
| `--features <list>` | Comma-separated list of devcontainer features (e.g. `docker-in-docker,git,aws-cli`). |
| `--vscode-extensions <list>` | Comma-separated extension IDs to install in the container. |
| `--dockerfile` | Generate a matching `Dockerfile` (default: use the official base image, no custom Dockerfile). |
| `--image <name>` | Override the base image. Useful for language-specific base images other than Python/Node. |
| `--port <list>` | Comma-separated ports to forward (e.g. `3000,8000`). |
| `--post-create <cmd>` | Single command to run after container creation. Use for `pip install -r requirements.txt` etc. |
| `--validate` | Validate the existing `.devcontainer/devcontainer.json` and exit. |
| `--dry-run` | Print what would be written; don't touch the filesystem. |

## Procedure

### 1. Decide the surface

Ask (or infer) what the user needs:
- A pre-baked image? Pass `--python` or `--node` to use the official `mcr.microsoft.com/devcontainers/*` base.
- A custom image built locally? Pass `--dockerfile` to emit a `Dockerfile` and reference it.
- Just want to add a feature to an existing config? Don't scaffold — read the existing file and add to the `features` object directly via the `patch` tool.

### 2. Scaffold

```bash
cd /path/to/repo
python scripts/init.py . --python 3.11 --features docker-in-docker --vscode-extensions ms-python.python
```

This creates `.devcontainer/devcontainer.json` (and `.devcontainer/Dockerfile` if `--dockerfile` was passed). Refuses to overwrite an existing config unless `--force` is passed.

### 3. Customize

Open the generated `devcontainer.json` and tweak:
- `postCreateCommand` — install Python deps, run `npm ci`, etc.
- `customizations.vscode.extensions` — add/remove editor extensions
- `forwardPorts` — ports the dev server uses
- `mounts` — bind-mounts for caches (e.g. `~/.cache/pip`)

### 4. Validate

```bash
python scripts/init.py . --validate
```

Catches: missing `image` or `dockerFile`, malformed JSON, unknown top-level fields, `features` not being a map.

### 5. Hand off

Tell the user to open the repo in VS Code and run "Dev Containers: Reopen in Container" (or push to a branch and open in Codespaces).

## Pitfalls

- **Devcontainers require Docker.** The skill writes config; the user's runtime runs it. If the user is on macOS and doesn't have Docker Desktop / OrbStack / Colima, the container will fail to start — the config is correct, the host is missing.
- **The base image's user is `vscode` (UID 1000), not `root`.** Generated `Dockerfile`s run `apt-get` as `root` in the build stage, then `USER vscode` at runtime. If the user adds a `RUN` step that needs root, put it in the Dockerfile (build stage) — not in `postCreateCommand` (which runs as `vscode`).
- **Features are pinned by version.** Generated configs use `docker-in-docker` (no version pin) for readability, but the user should pin to a specific feature version (e.g. `docker-in-docker:2`) for reproducibility once they've settled on the setup. The skill warns about unpinned features on `--validate`.
- **The `customizations.vscode.extensions` field is for VS Code only.** JetBrains Gateway ignores it. If the user is on JetBrains, they need `customizations.jetbrains` instead.
- **Forwarded ports aren't published to the host network by default.** Ports listed in `forwardPorts` are forwarded to the local machine, not the LAN. Use `otherPortsAttributes` to control bind addresses.
- **Devcontainer.json is JSON, not JSONC.** Generated files are pure JSON (no comments) so they parse cleanly in every consumer. The README in the generated directory is plain markdown, not embedded in the JSON.
- **The schema is strict about field order.** VS Code tolerates any order, but Codespaces and the devcontainer CLI are pickier about a few fields. Generated files use the order recommended by the schema.
- **`--validate` doesn't catch all schema violations.** It catches the common foot-guns (missing image, malformed JSON, wrong feature shape) but does not do a full JSON-schema validation. For that, point `ajv` at the official schema.

## Verification

A single command proves the skill is wired up and the scaffolder works:

```bash
python scripts/init.py /tmp/devcontainer-smoketest --python 3.11 --features docker-in-docker --dry-run
```

Expected: prints a JSON object to stdout that includes `"image": "mcr.microsoft.com/devcontainers/python:3.11-bookworm"` and a `features` block referencing `ghcr.io/devcontainers/features/docker-in-docker:2`. Exit 0. If anything else prints (traceback, "command not found"), the install is broken — check that the script is on `PATH` or run it with the explicit `python` prefix.
