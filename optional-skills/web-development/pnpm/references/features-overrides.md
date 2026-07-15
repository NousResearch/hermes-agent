---
name: pnpm-overrides
description: Force specific versions of dependencies including transitive dependencies
---

# pnpm Overrides

Overrides let you force specific versions of packages, including transitive dependencies. Useful for fixing security vulnerabilities or compatibility issues.

## Basic Syntax

Define overrides in `pnpm-workspace.yaml`. They can only be set at the **root** of the project.

> The `pnpm.overrides` field in `package.json` is **no longer read** (pnpm no longer reads any settings from `package.json#pnpm`). Move overrides to `pnpm-workspace.yaml`.

```yaml title="pnpm-workspace.yaml"
packages:
  - 'packages/*'

overrides:
  # Override all versions of a package
  lodash: ^4.17.21

  # Override specific version range
  "foo@^1.0.0": ^1.2.3

  # Override nested dependency (only zoo inside qar@1)
  "qar@1>zoo": "2"

  # Override to different package
  "underscore": "npm:lodash@^4.17.21"

  # Reference a catalog so the version stays in sync
  "react": "catalog:"
```

## Override Patterns

### Override all instances
```yaml
overrides:
  lodash: ^4.17.21
```
Forces all lodash installations to use ^4.17.21.

### Override specific parent version
```yaml
overrides:
  "foo@^1.0.0": ^1.2.3
```
Only override foo when the requested version matches ^1.0.0.

### Override nested dependency
```yaml
overrides:
  "express>cookie": ^0.6.0
  "foo@1.x>bar@^2.0.0>qux": ^1.0.0
```
Override cookie only when it's a dependency of express.

### Replace with different package
```yaml
overrides:
  # Replace underscore with lodash
  "underscore": "npm:lodash@^4.17.21"
  
  # Use local file
  "some-pkg": "file:./local-pkg"
  
  # Use git
  "some-pkg": "github:user/repo#commit"
```

### Remove a dependency
```yaml
overrides:
  "unwanted-pkg": "-"
  "foo@1.0.0>bar": "-"   # great for skipping unused optionalDependencies
```
The `-` removes the package entirely.

### Override peer dependencies

Overrides also apply to `peerDependencies`:

```yaml title="pnpm-workspace.yaml"
overrides:
  "react-dom>react": "18.1.0"
```

- Semver ranges, `workspace:`, and `catalog:` keep the entry as a peer dependency.
- Non-range specifiers (`link:`, `file:`) move it into `dependencies`.
- `-` removes the peer dependency entirely.

## Common Use Cases

### Security Fix

Force patched version of vulnerable package:

```yaml
overrides:
  # Fix CVE in transitive dependency
  "minimist": "^1.2.6"
  "json5": "^2.2.3"
```

### Deduplicate Dependencies

Force single version when multiple are installed:

```yaml
overrides:
  "react": "^18.2.0"
  "react-dom": "^18.2.0"
```

### Fix Peer Dependency Issues

```yaml
overrides:
  "@types/react": "^18.2.0"
```

### Replace Deprecated Package

```yaml
overrides:
  "request": "npm:@cypress/request@^3.0.0"
```

## Hooks Alternative

For more complex scenarios, use `.pnpmfile.mjs`:

```js title=".pnpmfile.mjs"
function readPackage(pkg, context) {
  // Override dependency version
  if (pkg.dependencies?.lodash) {
    pkg.dependencies.lodash = '^4.17.21'
  }

  // Add missing peer dependency
  if (pkg.name === 'some-package') {
    pkg.peerDependencies = {
      ...pkg.peerDependencies,
      react: '*'
    }
  }

  return pkg
}

export const hooks = {
  readPackage
}
```

Or extend a manifest declaratively with `packageExtensions` (no JS needed):

```yaml title="pnpm-workspace.yaml"
packageExtensions:
  react-redux:
    peerDependencies:
      react-dom: '*'
```

## Overrides vs Catalogs

| Feature | Overrides | Catalogs |
|---------|-----------|----------|
| Affects | All dependencies (including transitive) | Direct dependencies only |
| Usage | Automatic | Explicit `catalog:` reference |
| Purpose | Force versions, fix issues | Version management |
| Granularity | Can target specific parents | Package-wide only |

## Debugging

Check which version is resolved:

```bash
# See resolved versions
pnpm why lodash

# List all versions
pnpm list lodash --depth=Infinity
```

<!--
Source references:
- https://pnpm.io/settings#overrides
- https://pnpm.io/settings#packageextensions
- https://pnpm.io/pnpmfile
-->
