---
name: pnpm-store
description: Content-addressable storage system that makes pnpm fast and disk-efficient
---

# pnpm Store

pnpm uses a content-addressable store to save disk space and speed up installations. All packages are stored once globally and hard-linked to project `node_modules`.

## How It Works

1. **Global Store**: Packages are downloaded once to a central store
2. **Hard Links**: Projects link to store instead of copying files
3. **Content-Addressable**: Files are stored by content hash, deduplicating identical files

### Storage Layout

```
<store-dir>/                # Global content-addressable store (pnpm store path)
└── files/
    └── <hash>/             # Files stored by content hash

project/
└── node_modules/
    ├── .pnpm/              # Virtual store (hard links to global store)
    │   ├── lodash@4.17.21/
    │   │   └── node_modules/
    │   │       └── lodash/
    │   └── express@4.18.2/
    │       └── node_modules/
    │           ├── express/
    │           └── <deps>/  # Flat structure for dependencies
    ├── lodash -> .pnpm/lodash@4.17.21/node_modules/lodash
    └── express -> .pnpm/express@4.18.2/node_modules/express
```

## Store Commands

```bash
# Show store location
pnpm store path

# Remove unreferenced packages
pnpm store prune

# Check store integrity
pnpm store status

# Add package to store without installing
pnpm store add <pkg>
```

## Configuration

Store/linker settings live in `pnpm-workspace.yaml` (camelCase), not `.npmrc`.

### Store Location

```yaml title="pnpm-workspace.yaml"
storeDir: ~/.local/share/pnpm/store
```

The default store path is OS-specific (e.g. `~/.local/share/pnpm/store` on Linux, `~/Library/pnpm/store` on macOS). Find it with `pnpm store path`.

### Virtual Store

The virtual store (`.pnpm` in `node_modules`) contains hard links to the global store:

```yaml title="pnpm-workspace.yaml"
virtualStoreDir: node_modules/.pnpm
virtualStoreDirMaxLength: 60   # lower this for long-path issues on Windows
nodeLinker: hoisted            # alternative flat layout
```

## Disk Space Benefits

pnpm saves significant disk space:

- **Deduplication**: Same package version stored once across all projects
- **Content deduplication**: Identical files across different packages stored once
- **Hard links**: No copying, just linking

### Check disk usage

```bash
# Compare actual vs apparent size
du -sh node_modules        # Apparent size
du -sh --apparent-size node_modules  # With hard links counted
```

## Global Virtual Store

With `enableGlobalVirtualStore: true`, projects skip the per-project `node_modules/.pnpm` directory entirely; their `node_modules` contains only symlinks into one shared virtual store at `<store-path>/links/`, keyed by dependency-graph hash. In pnpm v11 it is the default for `pnpm dlx`/`pnx` and global installs; for project installs it is still opt-in. See `features-global-virtual-store` for details and the git-worktrees multi-agent workflow.

```yaml title="pnpm-workspace.yaml"
enableGlobalVirtualStore: true
```

## Node Linker Modes

Configure how `node_modules` is structured (`nodeLinker` in `pnpm-workspace.yaml`):

```yaml title="pnpm-workspace.yaml"
nodeLinker: isolated   # default: symlinked virtual store (strict, no phantom deps)
# nodeLinker: hoisted  # flat node_modules (npm-like) for tools that dislike symlinks
# nodeLinker: pnp      # Plug'n'Play, no node_modules (set `symlink: false` too)
```

### Isolated Mode (Default)

- Strict dependency resolution
- No phantom dependencies
- Packages can only access declared dependencies

### Hoisted Mode

- Flat `node_modules` like npm
- For compatibility with tools that don't support symlinks
- Loses strictness benefits

## Side Effects Cache

Cache build outputs for native modules (enabled by default):

```yaml title="pnpm-workspace.yaml"
sideEffectsCache: true
sideEffectsCacheReadonly: false   # only read the cache, don't create it
```

## Read-only / Frozen Store

`frozenStore: true` (v11.7+) lets `pnpm install` run against a read-only store (Nix store, read-only bind mount, OCI layer). Pair with `--offline --frozen-lockfile`; the store must already contain everything, including approved build outputs.

```bash
pnpm install --frozen-store --offline --frozen-lockfile
```

## Shared Store Across Machines

For CI/CD, you can share the store:

```yaml
# GitHub Actions example
- uses: pnpm/action-setup@v4
  with:
    run_install: false

- name: Get pnpm store directory
  shell: bash
  run: echo "STORE_PATH=$(pnpm store path --silent)" >> $GITHUB_ENV

- uses: actions/cache@v4
  with:
    path: ${{ env.STORE_PATH }}
    key: ${{ runner.os }}-pnpm-store-${{ hashFiles('**/pnpm-lock.yaml') }}
```

## Troubleshooting

### Store corruption
```bash
# Verify and fix store
pnpm store status
pnpm store prune
```

### Hard link issues (network drives, Docker)
```yaml title="pnpm-workspace.yaml"
# auto (default) tries clone -> hardlink -> copy
packageImportMethod: copy
```

### Permission issues
```bash
# Fix store permissions (find the path with `pnpm store path`)
chmod -R u+w "$(pnpm store path)"
```

<!--
Source references:
- https://pnpm.io/symlinked-node-modules-structure
- https://pnpm.io/cli/store
- https://pnpm.io/settings#storedir
- https://pnpm.io/global-virtual-store
-->
