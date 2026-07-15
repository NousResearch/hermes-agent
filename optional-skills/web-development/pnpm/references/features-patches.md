---
name: pnpm-patches
description: Patch third-party packages directly with customized fixes
---

# pnpm Patches

pnpm's patching feature lets you modify third-party packages directly. Useful for applying fixes before upstream releases or customizing package behavior.

## Creating a Patch

### Step 1: Initialize Patch

```bash
pnpm patch <pkg>@<version>

# Example
pnpm patch express@4.18.2
```

This creates a temporary directory with the package source and outputs the path:

```
You can now edit the following folder: /tmp/abc123...
```

### Step 2: Edit Files

Navigate to the temporary directory and make your changes:

```bash
cd /tmp/abc123...
# Edit files as needed
```

### Step 3: Commit Patch

```bash
pnpm patch-commit <path-from-step-1>

# Example
pnpm patch-commit /tmp/abc123...
```

This creates a `.patch` file in `patches/` and records it in `pnpm-workspace.yaml`:

```
patches/
└── express@4.18.2.patch
```

```yaml title="pnpm-workspace.yaml"
patchedDependencies:
  express@4.18.2: patches/express@4.18.2.patch
```

> `patchedDependencies` (like all pnpm settings) now lives in `pnpm-workspace.yaml`, not the `package.json#pnpm` field.

## Patch File Format

Patches use standard unified diff format:

```diff
diff --git a/lib/router/index.js b/lib/router/index.js
index abc123..def456 100644
--- a/lib/router/index.js
+++ b/lib/router/index.js
@@ -100,6 +100,7 @@ function createRouter() {
   // Original code
-  const timeout = 30000;
+  const timeout = 60000; // Extended timeout
   return router;
 }
```

## Managing Patches

### List Patched Packages

```bash
pnpm list --depth=0
# Shows (patched) marker for patched packages
```

### Update a Patch

```bash
# Edit existing patch
pnpm patch express@4.18.2

# After editing
pnpm patch-commit <path>
```

### Remove a Patch

```bash
pnpm patch-remove <pkg>@<version>

# Example  
pnpm patch-remove express@4.18.2
```

Or manually:
1. Delete the patch file from `patches/`
2. Remove the entry from `patchedDependencies` in `pnpm-workspace.yaml`
3. Run `pnpm install`

## Patch Configuration

### Multiple Packages / Workspaces

Patches are shared across the whole workspace from the root `pnpm-workspace.yaml`:

```yaml title="pnpm-workspace.yaml"
patchedDependencies:
  express@4.18.2: patches/express@4.18.2.patch
  lodash@4.17.21: patches/lodash@4.17.21.patch
  '@types/node@20.10.0': patches/@types__node@20.10.0.patch
```

A version-less key (`express:`) patches every installed version. All workspace packages using a matching version get the patch.

### Patches from a config dependency

Patch files can live inside a shared config dependency and be referenced by path:

```yaml title="pnpm-workspace.yaml"
configDependencies:
  my-patches: '1.0.0'
patchedDependencies:
  react: node_modules/.pnpm-config/my-patches/react.patch
```

### allowUnusedPatches

```yaml title="pnpm-workspace.yaml"
allowUnusedPatches: true   # don't fail when a listed patch wasn't applied
```

> `ignorePatchFailures` was **removed** in v11. A patch that fails to apply now always throws. When several patches are grouped, all errors are reported together at the end.

## Best Practices

1. **Version specificity**: Patches are tied to exact versions. Update patches when upgrading dependencies.

2. **Document patches**: Add comments explaining why the patch exists:
   ```bash
   # In patches/README.md
   ## express@4.18.2.patch
   Fixes timeout issue. PR pending: https://github.com/expressjs/express/pull/1234
   ```

3. **Minimize patches**: Keep patches small and focused. Large patches are hard to maintain.

4. **Track upstream**: Note upstream issues/PRs so you can remove patches when fixed.

5. **Test patches**: Ensure patched code works correctly in your use case.

## Troubleshooting

### Patch fails to apply

```
ERR_PNPM_PATCH_FAILED  Cannot apply patch
```

The package version changed. Recreate the patch:
```bash
pnpm patch-remove express@4.18.2
pnpm patch express@4.18.2
# Reapply changes
pnpm patch-commit <path>
```

### Patch not applied

Ensure:
1. Version in `patchedDependencies` matches installed version exactly
2. Run `pnpm install` after adding patch configuration

<!--
Source references:
- https://pnpm.io/cli/patch
- https://pnpm.io/cli/patch-commit
- https://pnpm.io/config-dependencies
-->
