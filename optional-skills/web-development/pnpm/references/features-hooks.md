---
name: pnpm-hooks
description: Customize resolution, config, packing, and fetching with .pnpmfile.mjs hooks, finders, and custom resolvers/fetchers
---

# pnpm Hooks (.pnpmfile.mjs)

pnpm hooks customize installation. Declare them in `.pnpmfile.mjs` (ESM, preferred) or `.pnpmfile.cjs` (CommonJS), located next to the lockfile (workspace root for a monorepo).

> The modern format uses ESM `export const hooks = { ... }`. The old CommonJS `module.exports = { hooks }` still works in `.pnpmfile.cjs`.

## Setup

```js title=".pnpmfile.mjs"
export const hooks = {
  readPackage,
  afterAllResolved,
  updateConfig,
  beforePacking,
}
```

## Hook reference

| Hook | When | Use |
|------|------|-----|
| `readPackage(pkg, ctx)` | after a dependency manifest is parsed | mutate a dependency's `package.json` (affects resolution) |
| `afterAllResolved(lockfile, ctx)` | after resolution | mutate the lockfile before it's written |
| `updateConfig(config)` | before install | mutate pnpm's settings (great with config dependencies) |
| `beforePacking(pkg)` | before `pnpm pack`/`publish` tarball | customize the **published** manifest only |
| `preResolution(opts)` | after reading lockfiles, before resolution | inspect/modify lockfile objects |
| `importPackage(dir, opts)` | when writing to node_modules | change how packages are linked |

## readPackage

Called for every package before resolution. Common uses:

```js title=".pnpmfile.mjs"
function readPackage(pkg, context) {
  // Add a missing peer dependency
  if (pkg.name === 'some-broken-package') {
    pkg.peerDependencies = { ...pkg.peerDependencies, react: '*' }
  }
  // Pin a transitive version
  if (pkg.dependencies?.lodash) pkg.dependencies.lodash = '^4.17.21'
  // Drop a problematic optional dep
  delete pkg.optionalDependencies?.fsevents
  // Replace a deprecated dep
  if (pkg.dependencies?.['old-pkg']) {
    pkg.dependencies['new-pkg'] = pkg.dependencies['old-pkg']
    delete pkg.dependencies['old-pkg']
  }
  return pkg
}

export const hooks = { readPackage }
```

> Mutations are not written to disk; they only affect resolution. Delete `pnpm-lock.yaml` to re-resolve an already-locked dependency. Removing `scripts` here does **not** stop a build — use the `allowBuilds` setting instead. To persist a change to a dependency's files, use `pnpm patch`.

## updateConfig

Modify pnpm's own settings programmatically — most powerful when shipped in a config dependency so settings are shared across repos.

```js title=".pnpmfile.mjs"
export const hooks = {
  updateConfig(config) {
    return Object.assign(config, {
      enablePrePostScripts: false,
      optimisticRepeatInstall: true,
      resolutionMode: 'lowest-direct',
      verifyDepsBeforeRun: 'install',
    })
  }
}
```

```js
// Add a catalog entry from a plugin
export const hooks = {
  updateConfig(config) {
    config.catalogs.default ??= {}
    config.catalogs.default['is-odd'] = '1.0.0'
    return config
  }
}
```

## beforePacking

Customize the manifest that ends up in the published tarball without touching your local `package.json`.

```js title=".pnpmfile.mjs"
export const hooks = {
  beforePacking(pkg) {
    delete pkg.devDependencies
    pkg.main = './dist/index.js'
    return pkg
  }
}
```

## afterAllResolved

```js title=".pnpmfile.mjs"
export const hooks = {
  afterAllResolved(lockfile, context) {
    context.log(`Resolved ${Object.keys(lockfile.packages || {}).length} packages`)
    return lockfile
  }
}
```

## Finders (pnpm list / why)

Custom predicates used via `--find-by`:

```js title=".pnpmfile.mjs"
export const finders = {
  react17: (ctx) => ctx.readManifest().peerDependencies?.react === '^17.0.0'
}
```

```bash
pnpm why --find-by=react17
```

## Custom resolvers & fetchers (advanced)

Register top-level `resolvers`/`fetchers` to support new package schemes (e.g. `my-protocol:pkg`). Each is an object with cheap `canResolve`/`canFetch` guards plus `resolve`/`fetch`. Custom resolvers run before built-ins; custom resolution `type` fields must use the `custom:` prefix.

```js title=".pnpmfile.cjs"
const resolver = {
  canResolve: (dep) => dep.alias.startsWith('@company/'),
  resolve: async (dep) => ({
    id: `${dep.alias}@${dep.bareSpecifier}`,
    resolution: { type: 'custom:cdn', cdnUrl: '...' },
  }),
}
const fetcher = {
  canFetch: (id, res) => res.type === 'custom:cdn',
  fetch: (cafs, res, opts, fetchers) =>
    fetchers.remoteTarball(cafs, { tarball: res.cdnUrl, integrity: res.integrity }, opts),
}
module.exports = { resolvers: [resolver], fetchers: [fetcher] }
```

> `hooks.fetchers` was removed in v11 — use the top-level `fetchers` export instead.

## Related settings

```yaml title="pnpm-workspace.yaml"
ignorePnpmfile: false                  # ignore the pnpmfile entirely
pnpmfile: ['.pnpmfile.mjs']            # local pnpmfile location(s)
globalPnpmfile: ~/.pnpm/global_pnpmfile.mjs
```

## Hooks vs Overrides

| | Hooks (.pnpmfile) | Overrides (pnpm-workspace.yaml) |
|--|-------------------|---------------------------------|
| Logic | JavaScript | declarative |
| Scope | any manifest field, config, lockfile, packing | versions |
| Use when | conditional/complex fixes | simple version pins |

Prefer `overrides`/`packageExtensions` for simple cases; use hooks for conditional logic, config sharing, or packing tweaks.

## Key Points

- Prefer `.pnpmfile.mjs` with `export const hooks`/`finders`/`resolvers`/`fetchers`.
- New hooks: `updateConfig` (mutate settings), `beforePacking` (published manifest), `preResolution`, `importPackage`.
- Pair `updateConfig` with config dependencies to share settings/catalogs across repos.
- `--ignore-scripts` does **not** disable the pnpmfile; use `ignorePnpmfile`.

<!--
Source references:
- https://pnpm.io/pnpmfile
- https://pnpm.io/finders
- https://pnpm.io/config-dependencies
-->
