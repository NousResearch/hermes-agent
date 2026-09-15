# Hermes Plugin Catalog

Curated, Nous-approved Hermes plugins. Each YAML file in this directory
(except `removed.yaml`) is one catalog entry, discoverable via
`hermes plugins catalog` / `hermes plugins search` and installable with
`hermes plugins install <name>`.

## Admission policy

Presence in this directory **is** the trust signal. The rules that keep it
meaningful:

1. **Human-merged gate.** Entries are added *only* via a PR to the
   `hermes-agent` repository, reviewed and merged by a maintainer. There is
   no self-serve registry, no automated ingestion.
2. **Exact SHA pins are mandatory.** Every entry pins a full 40-character
   commit SHA. Branches, tags, and short SHAs are rejected by the loader.
   Installs clone the repository and check out exactly that commit.
3. **No self-updating code.** A listed plugin must not fetch and replace
   its own files (in-app "check for updates", signed release downloaders,
   remote `plugin.js` loaders). The exact SHA pin *is* the trust model; a
   self-updater lets an installed copy move to a commit nobody reviewed.
   Updates reach users only through a SHA-bump PR here plus
   `hermes plugins update <name>`. Keep the updater in the standalone
   distribution if you want one; strip it from the catalog build.
4. **SHA bumps are new PRs.** Updating an entry's pin is a new PR whose diff
   (old SHA → new SHA) is re-reviewed like any other change — reviewers are
   expected to look at the upstream commit range being adopted.
5. **Owner-or-major-contributor submissions only.** An entry may only be
   submitted by the plugin repository's owner or a major contributor to it.
   Drive-by submissions of third-party repos are declined.
6. **Declared capabilities must match reality.** The `capabilities:` block
   (tools, hooks, middleware, env vars) must match what the plugin actually
   registers at the pinned commit. Validation fails the entry otherwise —
   undeclared capability creep is treated as a security issue.

## Entry schema

```yaml
name: example-plugin        # [a-z0-9_-]{1,64}, the catalog key
repo: https://github.com/owner/repo   # https:// only
sha: <40-hex commit sha>    # mandatory exact pin
subdir: ""                  # optional path within the repo
description: One-line description.
maintainer: OwnerName
tier: official              # official | community (default community)
category: memory            # desktop | memory | platform | web | tools | voice | automation | models | general
                            # (default desktop) — the shelf the entry sits on at /docs/plugins
requires_hermes: ">=0.19"   # optional
docs_url: ""                # optional
platforms: []               # optional, e.g. [linux, macos]; empty = all
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env: []
```

## removed.yaml — the blocklist

When an entry is pulled from the catalog for security or policy reasons, it
is recorded in `removed.yaml` with a reason and date. The installer refuses
to install anything matching a removed entry's name or repo URL, so a
malicious plugin cannot be re-installed from a stale identifier after
removal. Removals, like additions, land via reviewed PRs.
