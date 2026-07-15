---
name: pnpm-cli-commands
description: Essential pnpm commands for package management, running scripts, workspaces, publishing, and runtimes
---

# pnpm CLI Commands

pnpm provides a comprehensive CLI. Commands resemble npm/yarn but with unique features.

## Installation Commands

```bash
pnpm install            # install all deps (alias: pnpm i)
pnpm add <pkg>          # production dependency
pnpm add -D <pkg>       # devDependency       (also -d)
pnpm add -O <pkg>       # optionalDependency  (also -o)
pnpm add -E <pkg>       # exact version       (also -e)
pnpm add <pkg>@<version>
pnpm remove <pkg>       # aliases: rm, uninstall, un
pnpm update             # alias: up
pnpm update --latest    # ignore semver ranges (-L)
pnpm update -i          # interactive
```

### Clean / reproducible installs

```bash
pnpm install --frozen-lockfile   # fail if lockfile would change (auto in CI)
pnpm ci                          # clean install = pnpm clean + install --frozen-lockfile
pnpm clean                       # remove node_modules in all workspace projects (alias: purge)
pnpm clean --lockfile            # also delete pnpm-lock.yaml
```

> Since v11, an integrity mismatch against the lockfile is a hard error (`ERR_PNPM_TARBALL_INTEGRITY`). Use `pnpm install --update-checksums` only after verifying the new bytes. In CI, pnpm also fails on lockfiles written by a newer pnpm major.

## Script Commands

```bash
pnpm run <script>        # or just: pnpm <script>
pnpm run build -- --watch
pnpm run --if-present build
pnpm set-script test "vitest run"   # add/update a scripts entry (alias: ss)
pnpm exec <cmd>          # run a local binary, e.g. pnpm exec eslint .
```

- **Hidden scripts:** names starting with `.` (e.g. `.helper`) can't be run directly, only called from other scripts.
- **Built-in vs script conflict:** `clean`, `setup`, `deploy`, `rebuild` prefer a same-named `package.json` script. Force the built-in with `pnpm pm <name>` (e.g. `pnpm pm clean`).

### dlx / pnx — run without installing

```bash
pnx create-vite my-app          # pnx == pnpm dlx == pnpx
pnpm dlx degit user/repo dest
pnx shx@catalog:                # catalog: protocol supported
pnx --package=@scope/tool tool --help
```

> `dlx`/`pnx` honor supply-chain settings (`minimumReleaseAge`, `trustPolicy`) and use the global virtual store by default in v11.

## Workspace Commands

```bash
pnpm -r run <script>              # run in all packages (alias: --recursive)
pnpm --filter <pattern> run <script>
pnpm --filter "./packages/**" run build
pnpm --filter "@myorg/*" run lint
pnpm -r --parallel run dev
```

### Filter patterns

```bash
pnpm --filter <pkg-name> <cmd>      # by name (-F shorthand)
pnpm --filter "./packages/core" test
pnpm --filter "...@scope/app" build   # package + its dependencies
pnpm --filter "@scope/core..." test   # package + its dependents
pnpm --filter "...[origin/main]" build  # changed since git ref
```

## Patches

```bash
pnpm patch <pkg>@<version>     # opens an editable copy, prints a path
pnpm patch-commit <path>       # writes patches/*.patch and records it
pnpm patch-remove <pkg>@<version>
```

## Linking local packages

```bash
pnpm link <dir>          # link a path into this project's node_modules (path only!)
pnpm add -g .            # register the current package's bins globally
```

> Breaking in v11: `pnpm link` accepts **only relative/absolute paths** (no global store resolution, no `--global`, no bare `pnpm link`). Use `pnpm add -g .` to expose bins system-wide.

## Global packages (v11 isolated installs)

```bash
pnpm add -g typescript prettier   # each gets its own isolated install dir
pnpm add -g eslint,prettier       # comma = ONE shared install group
pnpm add -g --allow-build=esbuild esbuild
pnpm remove -g <pkg>
pnpm list -g
pnpm bin -g                       # show global bin dir ($PNPM_HOME/bin)
```

> `pnpm install -g` (no args) is not supported. After upgrading to v11 run `pnpm setup` so `$PNPM_HOME/bin` is on PATH.

## Runtimes (Node/Deno/Bun)

```bash
pnpm runtime set node 22 -g       # install & expose node (alias: rt)
pnpm runtime set node lts -g
pnpm runtime set deno 2 -g
pnpm install --no-runtime         # skip installing devEngines.runtime entries
```

## Store management

```bash
pnpm store path        # store location (prints removed size after prune)
pnpm store prune       # GC unreferenced packages (+ global virtual store links)
pnpm store status
```

## Inspection / registry

```bash
pnpm list                 # alias: ls
pnpm why <pkg>            # reverse-dependency tree (dedupes subtrees)
pnpm why --find-by=<finder>   # custom finder from .pnpmfile.mjs
pnpm outdated
pnpm audit
pnpm peers check          # report unmet/missing peers from the lockfile
pnpm view <pkg> [field]   # registry metadata (aliases: info, show)
pnpm whoami
pnpm rebuild
pnpm import               # create pnpm-lock.yaml from npm/yarn lockfile
pnpm dedupe
```

## Publishing

```bash
pnpm pack
pnpm publish -r --no-git-checks
pnpm version patch|minor|major|2.0.0    # bump version, commit + tag (v11)
pnpm version prerelease --preid beta
pnpm deprecate <pkg>@<range> "message"
pnpm dist-tag add <pkg>@<version> <tag>
pnpm unpublish <pkg>@<version>          # discouraged; prefer deprecate
pnpm sbom --sbom-format cyclonedx       # SBOM: cyclonedx (1.7) | spdx (2.3)
pnpm stage publish ...                  # staged publishing (defer 2FA)
```

## Maintenance & version management

```bash
pnpm self-update [<version>]   # updates the packageManager pin, or installs globally
pnpm with current install      # run a specific pnpm version for one command
pnpm with 11.0.0 install
pnpm approve-builds [--all]    # review dependency build scripts (writes allowBuilds)
```

## Useful Flags

```bash
pnpm install --ignore-scripts
pnpm install --prefer-offline
pnpm install --prod            # -P, omit devDependencies
pnpm install --no-optional
pnpm install --strict-peer-dependencies
```

## Key Points

- `pnpm ci` = clean + frozen install; CI auto-enables frozen-lockfile.
- `dlx`/`pnpx` are aliases of `pnx`; global installs are now isolated per package (comma-list to share).
- `pnpm link` only takes paths; use `pnpm add -g .` for global bins.
- Manage Node/Deno/Bun with `pnpm runtime set`; skip them at install with `--no-runtime`.
- New publishing/registry commands: `version`, `view`, `whoami`, `deprecate`, `dist-tag`, `unpublish`, `sbom`, `stage`.

<!--
Source references:
- https://pnpm.io/cli/install
- https://pnpm.io/cli/add
- https://pnpm.io/cli/run
- https://pnpm.io/filtering
- https://pnpm.io/cli/link
- https://pnpm.io/global-packages
- https://pnpm.io/cli/runtime
- https://pnpm.io/cli/version
- https://pnpm.io/cli/with
- https://pnpm.io/cli/sbom
-->
