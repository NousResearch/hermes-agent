---
name: pnpm-supply-chain-security
description: Build-script approval (allowBuilds), minimum release age, trust policy, and exotic-subdep blocking for safer installs
---

# pnpm Supply-Chain Security

pnpm blocks several attack vectors by default. Agents installing dependencies must understand these, since installs can fail or prompt on them.

## Build-script approval (allowBuilds)

By default pnpm does **not** run dependency lifecycle scripts (`preinstall`/`install`/`postinstall`). Packages must be explicitly approved. Approval lives in one `allowBuilds` map in `pnpm-workspace.yaml`.

```yaml title="pnpm-workspace.yaml"
allowBuilds:
  esbuild: true
  core-js: false
  # version selectors are supported
  nx@21.6.4 || 21.6.5: true
```

- Packages **not listed** are unreviewed and blocked by default.
- `strictDepBuilds: true` (default) ⇒ unreviewed builds make install exit non-zero (`ERR_PNPM_IGNORED_BUILDS`). Set `false` to warn instead.
- During install, unreviewed packages with build scripts are auto-added to `pnpm-workspace.yaml` with a placeholder so you can set `true`/`false`.

> `allowBuilds` replaces the removed `onlyBuiltDependencies`, `neverBuiltDependencies`, `ignoredBuiltDependencies`, `onlyBuiltDependenciesFile`, and `ignoreDepScripts`.

### Approving builds

```bash
pnpm approve-builds            # interactive prompt
pnpm approve-builds --all      # approve all pending
pnpm approve-builds esbuild fsevents !core-js   # ! = deny
pnpm add --allow-build=esbuild my-bundler       # approve while adding
pnpm add -g --allow-build=esbuild esbuild       # global (replaces approve-builds -g)
```

### Escape hatch (dangerous)

```yaml title="pnpm-workspace.yaml"
dangerouslyAllowAllBuilds: true   # runs ALL build scripts now and in the future — avoid
```

## Minimum release age

Delay installing freshly published versions so malicious releases (usually pulled within an hour) are avoided. Applies to **all** deps, including transitive.

```yaml title="pnpm-workspace.yaml"
minimumReleaseAge: 1440          # minutes; default 1440 (1 day) since v11
minimumReleaseAgeExclude:        # always install newest of these immediately
  - webpack
  - '@myorg/*'
  - nx@21.6.5                    # exempt a specific version
```

- `minimumReleaseAgeStrict` — when no in-range version satisfies the age, fail (default when you set `minimumReleaseAge` yourself) vs. fall back.
- `minimumReleaseAgeIgnoreMissingTime` — skip the check for registries that omit the `time` field (default `true`).

## Trust policy

Fail if a package's trust level **decreased** vs earlier releases (e.g. was published by a trusted publisher, now only has provenance or nothing).

```yaml title="pnpm-workspace.yaml"
trustPolicy: no-downgrade        # off (default) | no-downgrade
trustPolicyExclude:
  - 'chokidar@4.0.3'
trustPolicyIgnoreAfter: 525600   # ignore the check for pkgs published > N minutes ago
```

## Block exotic transitive sources

```yaml title="pnpm-workspace.yaml"
blockExoticSubdeps: true   # default
```

When `true`, only **direct** dependencies may use exotic sources (git repos, direct tarball URLs); all transitive deps must come from a trusted source (registry, local path, workspace link, or trusted GitHub repos).

## Lockfile integrity

Since v11, a downloaded tarball whose hash doesn't match `pnpm-lock.yaml` is a hard error (`ERR_PNPM_TARBALL_INTEGRITY`) — protecting committed lockfiles from a compromised registry/proxy. `--force` and `pnpm update` do **not** bypass it.

```bash
pnpm install --update-checksums   # narrow opt-in after verifying the new bytes
```

## Trusted store/cache

The content-addressable store, global virtual store, and metadata cache are part of pnpm's trust domain. Share them only between mutually trusting users/jobs and protect with filesystem permissions. `verifyStoreIntegrity` (default `true`) detects accidental corruption but does not make a writable-by-untrusted store safe.

## Key Points

- Dependency build scripts are blocked until approved via `allowBuilds` / `pnpm approve-builds`; unreviewed builds fail by default (`strictDepBuilds`).
- `minimumReleaseAge` (default 1 day in v11) delays new releases; `trustPolicy: no-downgrade` blocks trust regressions; `blockExoticSubdeps` limits transitive git/tarball sources.
- Tarball integrity mismatches are fatal; use `--update-checksums` only after verification.
- Treat the store/cache as trusted shared state.

<!--
Source references:
- https://pnpm.io/settings#allowbuilds
- https://pnpm.io/cli/approve-builds
- https://pnpm.io/settings#minimumreleaseage
- https://pnpm.io/settings#trustpolicy
- https://pnpm.io/settings#blockexoticsubdeps
- https://pnpm.io/supply-chain-security
-->
