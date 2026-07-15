---
name: pnpm-configuration
description: Configuring pnpm via pnpm-workspace.yaml (settings), the global config.yaml, and .npmrc (auth only)
---

# pnpm Configuration

pnpm settings are split into **two** categories. Knowing where each goes is the single most important config concept in current pnpm:

| Category | Stored in | Format |
|----------|-----------|--------|
| **All pnpm/install settings** (`nodeLinker`, `hoistPattern`, `autoInstallPeers`, `overrides`, `catalog`, …) | `pnpm-workspace.yaml` (project) and `config.yaml` (global) | YAML, **camelCase** keys |
| **Auth & registry credentials** (`_authToken`, `cert`, `key`, …) | `.npmrc` (project, gitignored) and global `rc` | INI |

> **Important changes:** pnpm no longer reads settings from the `pnpm` field of `package.json`, and `.npmrc` is now used **only** for authentication/registry credentials. Everything else belongs in `pnpm-workspace.yaml`. Keys in YAML are **camelCase** (e.g. `nodeLinker`), not the kebab-case used by old `.npmrc` files.

## pnpm-workspace.yaml (primary config)

Place at the workspace/project root. Even a single-package project uses this file for pnpm settings.

```yaml title="pnpm-workspace.yaml"
# Workspace packages (omit for a single-package repo)
packages:
  - 'packages/*'
  - 'apps/*'
  - '!**/test/**'

# Common install settings (camelCase)
nodeLinker: isolated          # isolated (default) | hoisted | pnp
autoInstallPeers: true
strictPeerDependencies: false
savePrefix: '^'
saveExact: false
hoistPattern:
  - '*eslint*'
  - '*babel*'
publicHoistPattern: []
shamefullyHoist: false
dedupeDirectDeps: false
resolutionMode: highest       # highest | time-based | lowest-direct

# Centralized version management
catalog:
  react: ^18.2.0

# Force dependency versions (root only)
overrides:
  lodash: ^4.17.21
  'foo@^1.0.0>bar': ^2.0.0

# Extend/patch broken package manifests
packageExtensions:
  react-redux:
    peerDependencies:
      react-dom: '*'

# Peer dependency rules
peerDependencyRules:
  ignoreMissing:
    - '@babel/*'
  allowedVersions:
    react: '17 || 18'
```

## Global configuration (config.yaml)

User-level non-auth settings live in a global YAML `config.yaml`:

- `$XDG_CONFIG_HOME/pnpm/config.yaml` (if set)
- Linux: `~/.config/pnpm/config.yaml`
- macOS: `~/Library/Preferences/pnpm/config.yaml`
- Windows: `~/AppData/Local/pnpm/config/config.yaml`

The companion global `rc` file (same directory, named `rc`) holds only registry/auth settings.

## Per-project settings in a workspace (packageConfigs)

There are no per-subproject `.npmrc` files anymore. Set per-package config via `packageConfigs` in the root `pnpm-workspace.yaml`:

```yaml title="pnpm-workspace.yaml"
packageConfigs:
  # Map form: keyed by package name
  project-1:
    saveExact: true
  project-2:
    savePrefix: '~'
  # Array form: pattern-matched rules
  # - match: ['project-1', 'project-2']
  #   modulesDir: node_modules
  #   saveExact: true
```

## .npmrc — authentication only

Keep auth tokens out of the repo (gitignore the project `.npmrc`). Auth files, highest priority first:

1. `<workspace root>/.npmrc` (project, gitignored)
2. `<pnpm config>/auth.ini` (written by `pnpm login`)
3. `~/.npmrc` (fallback for npm compatibility)

```ini title=".npmrc"
//registry.npmjs.org/:_authToken=${NPM_TOKEN}
@myorg:registry=https://npm.myorg.com/
//npm.myorg.com/:_authToken=${MYORG_TOKEN}
```

Configure registries themselves (non-secret) in `pnpm-workspace.yaml`:

```yaml title="pnpm-workspace.yaml"
registries:
  default: https://registry.npmjs.org/
  '@my-org': https://private.example.com/
# Named registry aliases usable as a prefix, e.g. `pnpm add work:@corp/lib`
namedRegistries:
  work: https://npm.work.example.com/
```

> Security: since v11, env-variable expansion is disabled for registry/proxy URLs and credential keys in the **project** `.npmrc` (to stop a malicious repo from leaking secrets). Put dynamic-token lines in the user-level auth file instead.

## The `pnpm config` command

```bash
# Writes to global config.yaml / rc by default
pnpm config set nodeVersion 22.0.0
pnpm config set --location=project nodeVersion 22.0.0   # writes pnpm-workspace.yaml

# JSON values create arrays/objects
pnpm config set --location=project --json allowBuilds '{"react": true}'

# get/list print JSON (no longer INI) since v11
pnpm config get nodeLinker
pnpm config get 'allowBuilds.react'
pnpm config list
```

## Environment variables

Use `pnpm_config_*` (or `PNPM_CONFIG_*`). pnpm **no longer reads `npm_config_*`**.

```bash
pnpm_config_save_exact=true pnpm add foo
```

## Notable settings that changed names

| Old (removed) | Replacement | Notes |
|---------------|-------------|-------|
| `onlyBuiltDependencies`, `neverBuiltDependencies`, `ignoredBuiltDependencies`, `onlyBuiltDependenciesFile` | `allowBuilds: { name: true\|false }` | Single map controlling build-script approval. See supply-chain-security. |
| `managePackageManagerVersions`, `packageManagerStrict`, `packageManagerStrictVersion`, `COREPACK_ENABLE_STRICT` | `pmOnFail: download\|ignore\|warn\|error` | Behavior when running pnpm version ≠ declared one. |
| `useNodeVersion` | `devEngines.runtime` (in `package.json`) | Runtime pinning. |
| `auditConfig.ignoreCves` | `auditConfig.ignoreGhsas` | Use GHSA IDs. |
| `allowNonAppliedPatches` | `allowUnusedPatches` | `ignorePatchFailures` removed (patches now always throw). |
| `package.json#pnpm` field | `pnpm-workspace.yaml` | No longer read at all. |

## Package Manager / Runtime pinning (package.json)

```json
{
  "packageManager": "pnpm@10.0.0",
  "devEngines": {
    "packageManager": { "name": "pnpm", "version": ">=11.0.0 <12.0.0", "onFail": "download" },
    "runtime": { "name": "node", "version": "22.x", "onFail": "download" }
  }
}
```

`devEngines.packageManager` supports ranges (resolved version stored in lockfile); `packageManager` requires an exact version. Override `onFail` without editing the manifest via `pmOnFail` / `runtimeOnFail` settings.

## Key Points

- All pnpm settings go in `pnpm-workspace.yaml` (camelCase) or global `config.yaml`; `.npmrc` is auth/registry only.
- `package.json#pnpm` and `npm_config_*` env vars are no longer read.
- Use `packageConfigs` for per-package settings inside a workspace.
- Build-script approval is now one `allowBuilds` map; package-manager strictness is one `pmOnFail` setting.
- `pnpm config get`/`list` output JSON, and `--location=project` writes to `pnpm-workspace.yaml`.

<!--
Source references:
- https://pnpm.io/settings
- https://pnpm.io/configuring
- https://pnpm.io/npmrc
- https://pnpm.io/pnpm-workspace_yaml
- https://pnpm.io/package_json
- https://pnpm.io/cli/config
-->
