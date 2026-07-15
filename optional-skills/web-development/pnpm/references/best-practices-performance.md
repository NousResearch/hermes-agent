---
name: pnpm-performance-optimization
description: Tips and tricks for faster installs and better performance
---

# pnpm Performance Optimization

pnpm is fast by default, but these optimizations can make it even faster.

## Install Optimizations

### Use Frozen Lockfile

Skip resolution when lockfile exists:

```bash
pnpm install --frozen-lockfile
```

This is faster because pnpm skips the resolution phase entirely.

### Prefer Offline Mode

Use cached packages when available:

```bash
pnpm install --prefer-offline
```

### Skip Optional Dependencies

If you don't need optional deps:

```bash
pnpm install --no-optional
```

### Skip Scripts

For CI or when scripts aren't needed:

```bash
pnpm install --ignore-scripts
```

**Caution:** Some packages require postinstall scripts to work correctly.

### Only Build Specific Dependencies

Build-script approval is a single `allowBuilds` map (replaces `onlyBuiltDependencies`/`neverBuiltDependencies`). Only allowed packages run install scripts:

```yaml title="pnpm-workspace.yaml"
allowBuilds:
  esbuild: true
  '@swc/core': true
  core-js: false      # explicitly skip
```

Packages not listed are treated as unreviewed (blocked by default). See `features-supply-chain-security` for the full build-approval workflow.

## Store Optimizations

### Side Effects Cache

Cache native module build results (enabled by default):

```yaml title="pnpm-workspace.yaml"
sideEffectsCache: true
```

This caches the results of postinstall scripts, speeding up subsequent installs.

### Global Virtual Store

For many checkouts of the same repo (e.g. git worktrees / multiple agents), enable the global virtual store so each project's `node_modules` is just symlinks into one shared store — near-zero per-checkout cost. Auto-disabled in CI.

```yaml title="pnpm-workspace.yaml"
enableGlobalVirtualStore: true
```

### Shared Store

A single content-addressable store is used for all projects by default:

```yaml title="pnpm-workspace.yaml"
storeDir: ~/.local/share/pnpm/store
```

Benefits: packages downloaded once, hard links save disk space, faster cached installs.

### Store Maintenance

Periodically clean unused packages:

```bash
# Remove unreferenced packages
pnpm store prune

# Check store integrity
pnpm store status
```

## Workspace Optimizations

### Parallel Execution

Run workspace scripts in parallel:

```bash
pnpm -r --parallel run build
```

Control concurrency:
```yaml title="pnpm-workspace.yaml"
workspaceConcurrency: 8
```

### Stream Output

See output in real-time:

```bash
pnpm -r --stream run build
```

### Filter to Changed Packages

Only build what changed:

```bash
# Build packages changed since main branch
pnpm --filter "...[origin/main]" run build
```

### Topological Order

Build dependencies before dependents:

```bash
pnpm -r run build
# Automatically runs in topological order
```

For explicit sequential builds:
```bash
pnpm -r --workspace-concurrency=1 run build
```

## Network Optimizations

Network/registry settings are camelCase in `pnpm-workspace.yaml` (registry URLs may also go in `registries`):

```yaml title="pnpm-workspace.yaml"
registries:
  default: https://registry.npmmirror.com/
fetchRetries: 3
fetchRetryMintimeout: 10000
fetchRetryMaxtimeout: 60000
networkConcurrency: 16          # auto: clamp(workers x 3, 16, 64)
httpProxy: http://proxy.company.com:8080
httpsProxy: http://proxy.company.com:8080
```

## Lockfile Optimization

### Single Lockfile (Monorepos)

Use shared lockfile for all packages (default):

```yaml title="pnpm-workspace.yaml"
sharedWorkspaceLockfile: true
```

Benefits:
- Single source of truth
- Faster resolution
- Consistent versions across workspace

### Lockfile-only Mode

Only update lockfile without installing:

```bash
pnpm install --lockfile-only
```

## Benchmarking

### Compare Install Times

```bash
# Clean install
rm -rf node_modules pnpm-lock.yaml
time pnpm install

# Cached install (with lockfile)
rm -rf node_modules
time pnpm install --frozen-lockfile

# With store cache
time pnpm install --frozen-lockfile --prefer-offline
```

### Profile Resolution

Debug slow installs:

```bash
# Verbose logging
pnpm install --reporter=append-only

# Debug mode
DEBUG=pnpm:* pnpm install
```

## Configuration Summary

Optimized `pnpm-workspace.yaml` for performance:

```yaml title="pnpm-workspace.yaml"
# Install behavior
autoInstallPeers: true
sideEffectsCache: true
optimisticRepeatInstall: true

# Build approval (only what's necessary)
allowBuilds:
  esbuild: true
  '@swc/core': true

# Network
fetchRetries: 3
networkConcurrency: 16

# Workspace
workspaceConcurrency: 4

# Many checkouts of the same repo
enableGlobalVirtualStore: true
```

## Quick Reference

| Scenario | Command/Setting |
|----------|-----------------|
| CI installs | `pnpm ci` / `pnpm install --frozen-lockfile` |
| Offline development | `--prefer-offline` |
| Control native builds | `allowBuilds` map |
| Parallel workspace | `pnpm -r --parallel run build` |
| Build changed only | `pnpm --filter "...[origin/main]" build` |
| Clean store | `pnpm store prune` |
| Many worktrees/agents | `enableGlobalVirtualStore: true` |

<!--
Source references:
- https://pnpm.io/settings
- https://pnpm.io/cli/install
- https://pnpm.io/filtering
- https://pnpm.io/global-virtual-store
-->

