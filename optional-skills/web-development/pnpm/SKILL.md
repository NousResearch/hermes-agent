---
name: pnpm
description: Node.js package manager with strict dependency resolution. Use when running pnpm specific commands, configuring workspaces via pnpm-workspace.yaml, or managing dependencies with catalogs, patches, overrides, config dependencies, or the global virtual store.
metadata:
  author: Anthony Fu
  version: "2026.6.22"
  source: Generated from https://github.com/pnpm/pnpm, scripts located at https://github.com/antfu/skills
---

pnpm is a fast, disk space efficient package manager. It uses a content-addressable store to deduplicate packages across all projects on a machine, and enforces strict dependency resolution by default, preventing phantom dependencies.

**Configuration model (important):** pnpm settings now live in `pnpm-workspace.yaml` (and the global `config.yaml`) using **camelCase** keys. `.npmrc` is used **only** for authentication/registry credentials, and the `pnpm` field of `package.json` is no longer read. When working in a pnpm project, check `pnpm-workspace.yaml` for settings/workspace structure and `.npmrc` only for auth. Always use `--frozen-lockfile` (or `pnpm ci`) in CI.

> The skill is based on pnpm 10.x, generated at 2026-06-22. It also covers v11 behavior changes (config split, isolated global packages, `allowBuilds`, `pmOnFail`, global virtual store) where current docs describe them.

## Core

| Topic | Description | Reference |
|-------|-------------|-----------|
| CLI Commands | install/add/remove/update, run, dlx/pnx, workspace, runtime, publishing (version, view, sbom, stage) | [core-cli](references/core-cli.md) |
| Configuration | pnpm-workspace.yaml settings (camelCase), global config.yaml, packageConfigs, .npmrc auth | [core-config](references/core-config.md) |
| Workspaces | Monorepo support: filtering, workspace protocol, shared lockfile, packageConfigs | [core-workspaces](references/core-workspaces.md) |
| Store | Content-addressable store, virtual store, node linker modes, frozen/read-only store | [core-store](references/core-store.md) |

## Features

| Topic | Description | Reference |
|-------|-------------|-----------|
| Catalogs | Centralized dependency versions; catalogMode, catalog: in overrides | [features-catalogs](references/features-catalogs.md) |
| Overrides | Force versions (incl. transitive & peer deps); packageExtensions | [features-overrides](references/features-overrides.md) |
| Patches | Modify third-party packages; patchedDependencies in pnpm-workspace.yaml | [features-patches](references/features-patches.md) |
| Aliases | Install under custom names (npm:) and registry aliases (namedRegistries) | [features-aliases](references/features-aliases.md) |
| Hooks | .pnpmfile.mjs hooks (readPackage, updateConfig, beforePacking), finders, resolvers/fetchers | [features-hooks](references/features-hooks.md) |
| Peer Dependencies | Auto-install, strict mode, rules, dedupePeers, peers check | [features-peer-deps](references/features-peer-deps.md) |
| Config Dependencies | Share hooks/settings/catalogs/patches across repos via configDependencies | [features-config-dependencies](references/features-config-dependencies.md) |
| Global Virtual Store | Shared node_modules, git-worktree multi-agent setups, isolated global packages | [features-global-virtual-store](references/features-global-virtual-store.md) |
| Supply-Chain Security | Build approval (allowBuilds), minimumReleaseAge, trustPolicy, lockfile integrity | [features-supply-chain-security](references/features-supply-chain-security.md) |

## Best Practices

| Topic | Description | Reference |
|-------|-------------|-----------|
| CI/CD Setup | GitHub Actions, GitLab, Docker, pnpm ci, store caching, frozen lockfiles | [best-practices-ci](references/best-practices-ci.md) |
| Migration | npm/Yarn → pnpm, phantom deps, and pnpm v10 → v11 config migration | [best-practices-migration](references/best-practices-migration.md) |
| Performance | Install optimizations, allowBuilds, global virtual store, workspace parallelization | [best-practices-performance](references/best-practices-performance.md) |
