---
name: pnpm-global-virtual-store
description: Global virtual store for shared node_modules across checkouts, git-worktree multi-agent setups, and isolated global packages
---

# Global Virtual Store, Git Worktrees & Global Packages

## Global virtual store

By default each project has its own `node_modules/.pnpm` virtual store containing hard links to the content-addressable store. With the **global virtual store** enabled, pnpm keeps one shared virtual store at `<store-path>/links/` (find it via `pnpm store path`), and each project's `node_modules` contains only **symlinks** into it.

```yaml title="pnpm-workspace.yaml"
enableGlobalVirtualStore: true
```

```
# Default (per-project .pnpm with hard links)
project-a/node_modules/lodash -> .pnpm/lodash@4.17.21/node_modules/lodash

# Global virtual store (symlink to shared location)
project-a/node_modules/lodash -> <store>/links/@/lodash/4.17.21/<hash>/node_modules/lodash
project-b/node_modules/lodash -> <store>/links/@/lodash/4.17.21/<hash>/node_modules/lodash  # same target
```

- **Package identity = hash of the dependency graph.** Two projects with the same `lodash@4.17.21` and the same transitive tree point at the exact same directory (NixOS-style). Different peers ⇒ separate entries.
- **Near-zero per-project cost** and **instant installs** once a version is in the store.
- In **pnpm v11** it is the default for `pnpm dlx`/`pnx` and global installs; for **project** installs it is still **opt-in/experimental**.

### Limitations

- **CI:** auto-disabled (no warm cache to benefit from).
- **Trust:** the store is shared writable state — only for mutually trusting projects/users/jobs; protect the path with filesystem permissions.
- **ESM hoisting:** relies on `NODE_PATH`, which Node ignores for ESM imports. If ESM deps import undeclared packages, resolution fails. Fix with `packageExtensions` or the `@pnpm/plugin-esm-node-path` config dependency.

## Git worktrees for multi-agent development

Git worktrees let you check out many branches simultaneously, each in its own directory, sharing one `.git` object store. Combined with the global virtual store, every worktree gets a fully functional `node_modules` that is almost free on disk — ideal for running multiple AI agents in parallel.

```sh
# Bare repo as the hub, one worktree per branch/agent
git clone --bare https://github.com/your-org/your-monorepo.git your-monorepo
cd your-monorepo
git worktree add ./main main
git worktree add ./feature-auth feat/auth
git worktree add ./fix-api fix/api-error
```

```yaml title="pnpm-workspace.yaml"
packages:
  - 'packages/*'
enableGlobalVirtualStore: true
```

```sh
cd main && pnpm install            # first install fills the global store
cd ../feature-auth && pnpm install # subsequent worktrees: nearly instant, just symlinks
```

Each worktree has its own `node_modules` tree (so agents can install different versions on different branches without conflict), but all package contents come from the one shared store. Remove a worktree with `git worktree remove ./feature-auth`.

> The pnpm repo itself uses this setup and ships helper scripts (`pnpm worktree:new <branch|pr>`). Assumes all worktrees/agents share the same trust boundary.

## Global packages (v11 isolated installs)

`pnpm add -g` was redesigned in v11 for isolation. Each globally installed package (or group) gets its own install directory with its own `package.json`, `node_modules/`, and lockfile, so global tools can't break each other via peer/hoisting conflicts. Installs are stored at `{pnpmHomeDir}/global/v11/{hash}/` and share the global virtual store.

```sh
pnpm add -g typescript prettier      # space-separated = separate isolated installs each
pnpm add -g eslint,prettier          # comma-separated = ONE shared install group
pnpm remove -g eslint                # removes only eslint's group
pnpm add -g --allow-build=esbuild esbuild   # pre-approve build scripts
pnpm list -g                         # always works at depth 0
pnpm bin -g                          # global bin dir = $PNPM_HOME/bin
```

- `pnpm install -g` (no args) is **not** supported — use `pnpm add -g <pkg>`.
- Binaries live in `$PNPM_HOME/bin` (not `$PNPM_HOME` directly). Run `pnpm setup` after upgrading to put it on PATH.
- Register a local package's bins globally with `pnpm add -g .` (replaces `pnpm link --global`).
- `pnpm list -g --depth=<n>` (n>0) only works for a single install group.

## Key Points

- `enableGlobalVirtualStore: true` ⇒ `node_modules` is symlinks into one shared, hash-addressed store.
- Best for many checkouts of the same repo (git worktrees, parallel agents); auto-disabled in CI.
- Watch out for ESM packages importing undeclared deps (NODE_PATH limitation).
- v11 global installs are isolated per package; comma-list to share a group; bins live in `$PNPM_HOME/bin`.

<!--
Source references:
- https://pnpm.io/global-virtual-store
- https://pnpm.io/git-worktrees
- https://pnpm.io/global-packages
- https://pnpm.io/settings#enableglobalvirtualstore
-->
