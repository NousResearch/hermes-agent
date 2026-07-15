---
name: pnpm-config-dependencies
description: Share and centralize pnpm hooks, settings, patches, catalogs, and overrides across repos via config dependencies
---

# pnpm Config Dependencies

Config dependencies are npm packages that pnpm installs **before** all regular dependencies, so they can supply hooks, settings, patches, catalogs, and overrides that are reused across many repositories. They let you keep one shared "pnpm config" package and consume it everywhere.

## Declaring config dependencies

They live in `pnpm-workspace.yaml`; their integrity is recorded in a dedicated env-lockfile document inside `pnpm-lock.yaml`.

```yaml title="pnpm-workspace.yaml"
configDependencies:
  my-configs: "1.0.0"
```

Add one with the `--config` flag:

```bash
pnpm add --config my-configs
pnpm add --config @myorg/pnpm-plugin-my-catalogs
```

## Constraints

- **No regular `dependencies`.** They may declare `optionalDependencies`, but only one level deep.
- **No lifecycle scripts** (`preinstall`, `postinstall`, …).
- `optionalDependencies` (used for platform-specific binaries, esbuild-style) must use **exact** versions — ranges/tags are rejected, keeping installs reproducible.

## Auto-loaded plugins

A config dependency named `pnpm-plugin-*`, `@*/pnpm-plugin-*`, or `@pnpm/plugin-*` has its `pnpmfile.mjs` (or `.cjs`) loaded automatically from the package root.

## Use cases

### Import hook logic from a shared package

Because config deps install before the pnpmfile loads, you can import from them:

```js title=".pnpmfile.mjs"
import { readPackage } from '.pnpm-config/my-hooks'

export const hooks = { readPackage }
```

### Share settings & catalogs via updateConfig

A plugin can inject settings/catalog entries through the `updateConfig` hook:

```js title="@myorg/pnpm-plugin-my-catalogs/pnpmfile.mjs"
export const hooks = {
  updateConfig(config) {
    config.catalogs.default ??= {}
    config.catalogs.default['is-odd'] = '1.0.0'
    return config
  }
}
```

After installing it as a config dependency, consumers can use the catalog:

```bash
pnpm add is-odd@catalog:   # installs is-odd@1.0.0, writes "is-odd": "catalog:"
```

### Share patch files

Reference patches stored inside a config dependency:

```yaml title="pnpm-workspace.yaml"
configDependencies:
  my-patches: "1.0.0"
patchedDependencies:
  react: "node_modules/.pnpm-config/my-patches/react.patch"
```

## Key Points

- Centralize hooks, settings, catalogs, overrides, and patches in one package, consumed across repos.
- Declared via `configDependencies` in `pnpm-workspace.yaml`; installed before regular deps.
- No regular dependencies and no lifecycle scripts; `optionalDependencies` need exact versions.
- `pnpm-plugin-*` / `@pnpm/plugin-*` packages auto-load their pnpmfile.
- Pair with the `updateConfig` hook to push settings/catalogs into consuming projects.

<!--
Source references:
- https://pnpm.io/config-dependencies
- https://pnpm.io/pnpmfile#hooksupdateconfigconfig-config--promiseconfig
-->
