---
name: pnpm-aliases
description: Install packages under custom names for versioning, forks, or alternatives
---

# pnpm Aliases

pnpm supports package aliases using the `npm:` protocol. This lets you install packages under different names, use multiple versions of the same package, or substitute packages.

## Basic Syntax

```bash
pnpm add <alias>@npm:<package>@<version>
```

In `package.json`:
```json
{
  "dependencies": {
    "<alias>": "npm:<package>@<version>"
  }
}
```

## Use Cases

### Multiple Versions of Same Package

Install different versions side by side:

```json
{
  "dependencies": {
    "lodash3": "npm:lodash@3",
    "lodash4": "npm:lodash@4"
  }
}
```

Usage:
```js
import lodash3 from 'lodash3'
import lodash4 from 'lodash4'
```

### Replace Package with Fork

Substitute a package with a fork or alternative:

```json
{
  "dependencies": {
    "original-pkg": "npm:my-fork@^1.0.0"
  }
}
```

All imports of `original-pkg` will resolve to `my-fork`.

### Replace Deprecated Package

```json
{
  "dependencies": {
    "request": "npm:@cypress/request@^3.0.0"
  }
}
```

### Scoped to Unscoped (or vice versa)

```json
{
  "dependencies": {
    "vue": "npm:@anthropic/vue@^3.0.0",
    "@myorg/utils": "npm:lodash@^4.17.21"
  }
}
```

## CLI Usage

### Add with alias

```bash
# Add lodash under alias
pnpm add lodash4@npm:lodash@4

# Add fork as original name
pnpm add request@npm:@cypress/request
```

### Add multiple versions

```bash
pnpm add react17@npm:react@17 react18@npm:react@18
```

## With TypeScript

For type resolution with aliases, you may need to configure TypeScript:

```json
// tsconfig.json
{
  "compilerOptions": {
    "paths": {
      "lodash3": ["node_modules/lodash3"],
      "lodash4": ["node_modules/lodash4"]
    }
  }
}
```

Or use `@types` packages with aliases:

```json
{
  "devDependencies": {
    "@types/lodash3": "npm:@types/lodash@3",
    "@types/lodash4": "npm:@types/lodash@4"
  }
}
```

## Combined with Overrides

Force all transitive dependencies to use an alias:

```yaml
# pnpm-workspace.yaml
overrides:
  "underscore": "npm:lodash@^4.17.21"
```

This replaces all `underscore` imports (including in dependencies) with lodash.

## Git and Local Aliases

Aliases work with any valid pnpm specifier:

```json
{
  "dependencies": {
    "my-fork": "npm:user/repo#commit",
    "local-pkg": "file:../local-package"
  }
}
```

## Registry Aliases (namedRegistries)

Distinct from package aliases: a `namedRegistries` prefix selects *which registry* a package is fetched from.

```yaml title="pnpm-workspace.yaml"
namedRegistries:
  work: https://npm.work.example.com/
```

```bash
pnpm add work:@corp/lib@^2.0.0   # resolves @corp/lib against the work registry
```

The built-in `gh:` alias points at GitHub Packages. Auth is reused from per-URL `.npmrc` entries.

## Best Practices

1. **Clear naming**: Use descriptive alias names that indicate purpose
   ```json
   "lodash-legacy": "npm:lodash@3"
   "lodash-modern": "npm:lodash@4"
   ```

2. **Document aliases**: explain why aliases exist

3. **Prefer overrides for global replacement**: to replace a package everywhere, use `overrides` (in `pnpm-workspace.yaml`) instead of aliases

4. **Test thoroughly**: Aliased packages may have subtle differences in behavior

<!--
Source references:
- https://pnpm.io/aliases
- https://pnpm.io/settings#namedregistries
-->

