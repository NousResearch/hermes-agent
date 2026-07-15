---
name: pnpm-catalogs
description: Centralized dependency version management for workspaces
---

# pnpm Catalogs

Catalogs provide a centralized way to manage dependency versions across a workspace. Define versions once, use everywhere.

## Basic Usage

Define a catalog in `pnpm-workspace.yaml`:

```yaml
packages:
  - 'packages/*'

catalog:
  react: ^18.2.0
  react-dom: ^18.2.0
  typescript: ~5.3.0
  vite: ^5.0.0
```

Reference in `package.json` with `catalog:`:

```json
{
  "dependencies": {
    "react": "catalog:",
    "react-dom": "catalog:"
  },
  "devDependencies": {
    "typescript": "catalog:",
    "vite": "catalog:"
  }
}
```

`catalog:` is shorthand for `catalog:default`. The `catalog:` protocol is valid in `package.json` `dependencies`, `devDependencies`, `peerDependencies`, and `optionalDependencies`, plus in `overrides` inside `pnpm-workspace.yaml`. It also works on the CLI: `pnpm add react@catalog:` and `pnx shx@catalog:`.

## Named Catalogs

Create multiple catalogs for different scenarios:

```yaml
packages:
  - 'packages/*'

# Default catalog
catalog:
  lodash: ^4.17.21

# Named catalogs
catalogs:
  react17:
    react: ^17.0.2
    react-dom: ^17.0.2
  
  react18:
    react: ^18.2.0
    react-dom: ^18.2.0
  
  testing:
    vitest: ^1.0.0
    "@testing-library/react": ^14.0.0
```

Reference named catalogs:

```json
{
  "dependencies": {
    "react": "catalog:react18",
    "react-dom": "catalog:react18"
  },
  "devDependencies": {
    "vitest": "catalog:testing"
  }
}
```

## Keeping overrides in sync with a catalog

Reference a catalog from `overrides` so the version lives in exactly one place:

```yaml title="pnpm-workspace.yaml"
catalog:
  foo: ^1.0.0

overrides:
  foo: 'catalog:'          # or catalog:<name>
```

## Settings

```yaml title="pnpm-workspace.yaml"
# How `pnpm add` interacts with the default catalog (v10.12+)
catalogMode: manual        # manual (default) | prefer | strict
# strict: only catalog versions allowed; prefer: fall back if no match
cleanupUnusedCatalogs: true  # remove unused catalog entries on install (v10.15+)
```

## Benefits

1. **Single source of truth**: Update version in one place
2. **Consistency**: All packages use the same version
3. **Easy upgrades**: Change version once, affects entire workspace
4. **Fewer merge conflicts**: package.json files stay untouched on upgrades

## Catalog vs Overrides

| Feature | Catalogs | Overrides |
|---------|----------|-----------|
| Purpose | Define versions for direct dependencies | Force versions for any dependency |
| Scope | Direct dependencies only | All dependencies (including transitive) |
| Usage | `"pkg": "catalog:"` | Applied automatically |
| Opt-in | Explicit per package.json | Global to workspace |

## Publishing with Catalogs

When publishing, `catalog:` references are replaced with actual versions:

```json
// Before publish (source)
{
  "dependencies": {
    "react": "catalog:"
  }
}

// After publish (published package)
{
  "dependencies": {
    "react": "^18.2.0"
  }
}
```

## Migration from Overrides

If you're using overrides for version consistency:

```yaml
# Before (using overrides)
overrides:
  react: ^18.2.0
  react-dom: ^18.2.0
```

Migrate to catalogs for cleaner dependency management:

```yaml
# After (using catalogs)
catalog:
  react: ^18.2.0
  react-dom: ^18.2.0
```

Then update package.json files to use `catalog:`. To migrate an existing workspace automatically:

```bash
pnpx codemod pnpm/catalog
```

## Best Practices

1. **Use default catalog** for commonly shared dependencies
2. **Use named catalogs** for version variants (e.g., different React versions)
3. **Keep catalog minimal** - only include shared dependencies
4. **Combine with workspace protocol** for internal packages

```yaml
catalog:
  # External shared dependencies
  lodash: ^4.17.21
  zod: ^3.22.0

# Internal packages use workspace: protocol instead
# "dependencies": { "@myorg/utils": "workspace:^" }
```

<!-- 
Source references:
- https://pnpm.io/catalogs
-->
