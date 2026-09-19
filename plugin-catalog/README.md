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
5. **Owner-or-major-contributor submissions, or a maintainer-curated sweep.**
   An entry may be submitted by the plugin repository's owner or a major
   contributor to it; drive-by submissions of third-party repos are declined.
   Hermes maintainers may also add entries in batches from a reviewed sweep
   of community plugins (every pin validated and scanned at the pinned
   commit, self-updater and credential-store checks run, English-first UI).
   Authors of swept-in entries keep control: a PR from the owner adjusting
   or removing their entry is accepted on request, and SHA bumps stay
   owner-or-maintainer PRs under rule 4.
6. **Declared capabilities must match reality.** The `capabilities:` block
   (tools, hooks, middleware, env vars) must match what the plugin actually
   registers at the pinned commit. Validation fails the entry otherwise —
   undeclared capability creep is treated as a security issue.
7. **The install scanner runs at admission.** `hermes plugins validate` includes
   the `security scan` check: `dangerous` fails the entry; `caution` findings
   appear as warnings in the CI log and the reviewer reads them before merging.
   In exchange, installs at the pinned SHA accept `caution` without a prompt
   (`dangerous` still blocks). Review the warnings; do not merge past them.

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
version: "1.4.0"            # optional human label for the sha (quote it); shown as "1.4.0 @ abcd1234"
image: ""                   # optional https image on a GitHub host, 2:1 (e.g. 1200x600), e.g.
                            # https://raw.githubusercontent.com/owner/repo/<sha>/docs/banner.png
platforms: []               # optional, e.g. [linux, macos]; empty = all
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env: []
```

`version` and `image` are cosmetic: neither is parsed or used to pick what
installs. The sha stays the release; bump `version` in the same PR that bumps
`sha` so the label on the card matches the code. Images must live on
`raw.githubusercontent.com`, `github.com` or `*.githubusercontent.com` so
the Desktop catalog never fetches from third-party hosts; pin the raw URL to
the entry's commit and the picture is as immutable as the code.

## removed.yaml — the blocklist

When an entry is pulled from the catalog for security or policy reasons, it
is recorded in `removed.yaml` with a reason and date. The installer refuses
to install anything matching a removed entry's name or repo URL, so a
malicious plugin cannot be re-installed from a stale identifier after
removal. Removals, like additions, land via reviewed PRs.

Delisting is different from removal: an entry that is merely unmaintained, superseded, or
squatting a name it is not affiliated with is deleted from the catalog (plain file removal,
users who already installed it are unaffected) and is welcome back under a distinct name.

## Names

The catalog key and the manifest `name:` are what users search, install, and — for memory
providers — put in `memory.provider`. A `memory` / `exclusive` entry must not reuse the name of
another provider or of a well-known upstream project it is not affiliated with: two providers
registering the same `register_memory_provider` name make `memory.provider` ambiguous
(whichever loads last wins). Reviewers check the registered provider name, not just the file
name, and the affiliated project gets the bare key.
