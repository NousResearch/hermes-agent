---
name: pnpm-workspaces
description: Monorepo support with workspaces for managing multiple packages
---

# pnpm Workspaces

pnpm has built-in support for monorepos (multi-package repositories) through workspaces.

## Setting Up Workspaces

Create `pnpm-workspace.yaml` at the repository root:

```yaml
packages:
  # Include all packages in packages/ directory
  - 'packages/*'
  # Include all apps
  - 'apps/*'
  # Include nested packages
  - 'tools/*/packages/*'
  # Exclude test directories
  - '!**/test/**'
```

## Workspace Protocol

Use `workspace:` protocol to reference local packages:

```json
{
  "dependencies": {
    "@myorg/utils": "workspace:*",
    "@myorg/core": "workspace:^",
    "@myorg/types": "workspace:~"
  }
}
```

### Protocol Variants

| Protocol | Behavior | Published As |
|----------|----------|--------------|
| `workspace:*` | Any version | Actual version (e.g., `1.2.3`) |
| `workspace:^` | Compatible version | `^1.2.3` |
| `workspace:~` | Patch version | `~1.2.3` |
| `workspace:^1.0.0` | Semver range | `^1.0.0` |

## Filtering Packages

Run commands on specific packages using `--filter`:

```bash
# By package name
pnpm --filter @myorg/app build
pnpm -F @myorg/app build

# By directory path
pnpm --filter "./packages/core" test

# Glob patterns
pnpm --filter "@myorg/*" lint
pnpm --filter "!@myorg/internal-*" publish

# All packages
pnpm -r build
pnpm --recursive build
```

### Dependency-based Filtering

```bash
# Package and all its dependencies
pnpm --filter "...@myorg/app" build

# Package and all its dependents
pnpm --filter "@myorg/core..." test

# Both directions
pnpm --filter "...@myorg/shared..." build

# Changed since git ref
pnpm --filter "...[origin/main]" test
pnpm --filter "[HEAD~5]" lint
```

## Workspace Commands

### Install dependencies
```bash
# Install all workspace packages
pnpm install

# Add dependency to specific package
pnpm --filter @myorg/app add lodash

# Add workspace dependency
pnpm --filter @myorg/app add @myorg/utils
```

### Run scripts
```bash
# Run in all packages with that script
pnpm -r run build

# Run in topological order (dependencies first)
pnpm -r --workspace-concurrency=1 run build

# Run in parallel
pnpm -r --parallel run test

# Stream output
pnpm -r --stream run dev
```

### Execute commands
```bash
# Run command in all packages
pnpm -r exec pwd

# Run in specific packages
pnpm --filter "./packages/**" exec rm -rf dist
```

## Workspace Settings

Configure in `pnpm-workspace.yaml` using **camelCase** keys (these settings no longer belong in `.npmrc`):

```yaml title="pnpm-workspace.yaml"
packages:
  - 'packages/*'

# Link workspace packages automatically
linkWorkspacePackages: true
# Prefer workspace packages over registry
preferWorkspacePackages: true
# Single lockfile for the whole workspace (recommended)
sharedWorkspaceLockfile: true
# Workspace protocol handling on publish
saveWorkspaceProtocol: rolling
# Concurrent workspace scripts
workspaceConcurrency: 4
# Use root deps to resolve peers of all projects
resolvePeersFromWorkspaceRoot: true
# Scripts required in every project (else `pnpm -r run <name>` fails)
requiredScripts:
  - build
```

### Per-package configuration (packageConfigs)

There are no per-subproject `.npmrc` files. Set package-specific settings from the root file:

```yaml title="pnpm-workspace.yaml"
packageConfigs:
  project-1:
    saveExact: true
  project-2:
    savePrefix: '~'
```

## Publishing Workspaces

When publishing, `workspace:` protocols are converted:

```json
// Before publish
{
  "dependencies": {
    "@myorg/utils": "workspace:^"
  }
}

// After publish
{
  "dependencies": {
    "@myorg/utils": "^1.2.3"
  }
}
```

Use `--no-git-checks` for publishing from CI:
```bash
pnpm publish -r --no-git-checks
```

## Best Practices

1. **Use workspace protocol** for internal dependencies
2. **Enable `linkWorkspacePackages`** for automatic linking
3. **Use shared lockfile** for consistency
4. **Filter by dependencies** when building to ensure correct order
5. **Use catalogs** for shared external dependency versions (defined in this same file)
6. **Keep all pnpm settings in `pnpm-workspace.yaml`** (camelCase), not `.npmrc`

## Example Project Structure

```
my-monorepo/
├── pnpm-workspace.yaml
├── package.json
├── pnpm-lock.yaml
├── packages/
│   ├── core/
│   │   └── package.json
│   ├── utils/
│   │   └── package.json
│   └── types/
│       └── package.json
└── apps/
    ├── web/
    │   └── package.json
    └── api/
        └── package.json
```

<!-- 
Source references:
- https://pnpm.io/workspaces
- https://pnpm.io/filtering
- https://pnpm.io/npmrc#workspace-settings
-->
