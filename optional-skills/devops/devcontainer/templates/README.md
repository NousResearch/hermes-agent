# Dev Container

This directory defines a reproducible development environment for the
project. Open the repo in VS Code (with the Dev Containers extension),
or push to a branch and open in GitHub Codespaces.

## What's inside

- `devcontainer.json` — the open-standard config (https://containers.dev/).
  Tells the host editor what image to build, which features to add,
  which VS Code extensions to install, and which ports to forward.
- `Dockerfile` — present if the project needs a custom image; absent
  when the official Microsoft dev-container base image is enough.

## How to use

### VS Code

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open this repo.
3. Run the command: **Dev Containers: Reopen in Container**.
4. The first build takes a few minutes. Subsequent opens are fast.

### Codespaces

1. Push the repo to GitHub.
2. Click "Code" → "Codespaces" → "Create codespace on \<branch\>".

### devcontainer CLI (any editor)

```bash
npm install -g @devcontainers/cli
devcontainer up --workspace-folder .
```

## Re-running after edits

If you change `devcontainer.json` or `Dockerfile`, the running container
won't pick up the change automatically. Use **Dev Containers: Rebuild
Container** (VS Code) or `devcontainer up --build-no-cache --workspace-folder .`
(CLI).
