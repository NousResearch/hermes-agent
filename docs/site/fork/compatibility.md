---
title: Compatibility
description: How the ForgeGuard fork release, upstream base, and Hermes product version map to image tags and desktop artifacts.
order: 61
status: stable
---

# Compatibility

This page maps the ForgeGuard fork to the upstream product. For platform,
engine, and architecture support, see
[Platform compatibility](../reference/compatibility.md).

## Version mapping

| Field | Value | Source |
|---|---|---|
| Upstream base (`FORK_UPSTREAM_BASE`) | `v2026.8.31` | `FORK_UPSTREAM_BASE` marker at the repository root (surfaced as the "Upstream release" line in each release's notes). |
| Fork release line | `v0.21.1` | Latest [fork release](https://github.com/forgeguard-ai/hermes-agent/releases). |
| Hermes product version | `0.21.1` (fork line; upstream `v2026.8.31` is product `0.21.0`) | `pyproject.toml` (names the release tag and title). |
| Runtime image | `runtime-<version>` (+ `-<sha>`, `-latest`; `cli-*` retired in fork v0.20.7) | `ghcr.io/forgeguard-ai/hermes-agent`. |
| Desktop artifacts | `.AppImage`/`.deb`/`.rpm`, `.dmg`/`.zip`, `-setup.exe`/`-portable.exe` | Attached to the fork release (versioned by the Release tag). |

> **Version-sensitive.** These values describe the current fork state. Always
> confirm against the live `FORK_UPSTREAM_BASE` marker and the newest
> [release](https://github.com/forgeguard-ai/hermes-agent/releases) — this fork
> advances quickly. Since `v0.20.4` the fork tag is the fork's **own** patch
> line: it started from the Hermes product version (`v0.19.0` … `v0.20.3`) and
> now bumps by one on every fork release, sync or not — so it can run ahead of
> upstream's product number (`v0.20.4` ships upstream `v2026.8.16.2`, which is
> product `0.20.3`). The upstream base a release tracks is always the
> "Upstream release" line in its notes / `FORK_UPSTREAM_BASE`. Releases up to
> `v2026.7.1-forgeguard.6` carry older date-shaped tags named after the
> upstream base; `-forgeguard.<n>` re-cut suffixes appear only if a version is
> ever re-released.

## What the fork tracks

ForgeGuard syncs to upstream **tagged releases**, not the moving `upstream/main`
tip, so each fork sync point is stable and reproducible. When upstream cuts a new
tagged release that ForgeGuard adopts, `FORK_UPSTREAM_BASE` advances and
subsequent releases record the new base in their notes; the release tag itself
follows the Hermes product version, so a new fork release line starts when the
product version changes.

## Upstream features not independently supported by ForgeGuard

ForgeGuard packages and releases the upstream product; it does not independently
support, extend, or guarantee:

- the provider and model catalog;
- the messaging platform adapters;
- the security/isolation model;
- the native installer and non-container installation paths;
- product features and configuration.

For all of those, the upstream product documentation is authoritative:
[Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/). Report bugs in
those areas upstream — see
[How ForgeGuard relates to upstream](./upstream.md#where-to-file-issues).

## Related

- [ForgeGuard changes](./forgeguard-changes.md)
- [How ForgeGuard relates to upstream](./upstream.md)
- [Platform compatibility](../reference/compatibility.md)
- [Releases and upgrades](../operations/releases-and-upgrades.md)
