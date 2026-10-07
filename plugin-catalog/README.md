# Hermes Plugin Catalog

Curated, Nous-approved Hermes plugins. Each YAML file in this directory
(except `removed.yaml`) is one catalog entry, discoverable via
`hermes plugins catalog` / `hermes plugins search` and installable with
`hermes plugins install <name>`.

## Admission policy

Presence in this directory **is** the trust signal. The rules that keep it
meaningful:

<!-- admission-rules:start (mirrored in website/docs/developer-guide/plugins/catalog-submission.md; tests/website/test_catalog_rules_mirror.py keeps them identical) -->
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
8. **Desktop plugins stay inside the SDK surface.** A `desktop/plugin.js` runs
   in the Desktop renderer with the app's full authority (the loader isolates
   errors, not capabilities), so a listed one may only use the plugin SDK:
   no prototype patching (`X.prototype.y =`, `Object.defineProperty(...prototype`),
   no `eval`/`new Function`, no `import()` of anything but `@hermes/plugin-sdk`
   / `react` (app bundle chunks, blob or http URLs included), no script-tag
   injection, no reaching into the app's internal stores or its own markup
   (querying `data-slot` / `data-tour` / `data-sidebar` / `data-testid`
   elements from `document`, or a `document.body` MutationObserver, to restyle,
   hide, click or rewrite core UI). `hermes plugins validate` refuses these at
   admission (`desktop surface` check); a plugin that needs a capability the
   SDK lacks asks for an SDK hook instead of patching around it.
9. **No runtime overrides of Hermes core.** A listed plugin extends Hermes only
   through public surfaces: hooks, middleware, provider profiles and the
   other `register_*` APIs, and Desktop SDK slots and routes. It must not
   replace, wrap or rebind core functions, methods, module attributes or
   private dicts in place (`AIAgent.<method> = ...`, `setattr(server, ...)`,
   `sys.modules[...]`, writes into a core module's tables). Two plugins
   patching the same seam silently break each other, and every core release
   can break both. `hermes plugins validate` refuses these at admission (`no
   core override` check). If the hook you need does not exist, open an issue
   describing it: we would rather add the seam than list a patch.
10. **Dependency security policy is the plugin's.** Hermes's 14-day
   `exclude-newer` quarantine covers Hermes's own dependencies only; a plugin's
   `python_dependencies` / `pyproject.toml` install under the plugin's policy
   (no quarantine, still inside Hermes's core constraints). Reviewers read the
   dependency list at the pinned SHA: bare floors (`>=X` with no upper bound)
   and floors on the newest release get a request for the oldest
   API-compatible floor plus an upper bound, and authors are strongly
   recommended to run their own release quarantine (`uv --exclude-newer` in
   their CI) — see the developer guide's *Dependency security policy*. A
   recent floor alone is not grounds to hold an entry.

11. **Credentials stay with their owner.** A plugin reads the credentials it is
   configured with: the env vars in `requires_env` and its own `config_schema`
   secrets. Reading another tool's login (a vendor CLI's token file, a browser
   profile) must be disclosed in the PR and is a trust-tier call for a
   maintainer. Refreshing, rotating or writing another client's OAuth tokens, or
   presenting itself as another vendor's client, is not admitted without an
   explicit maintainer ruling; a read-only build is the usual way through.
12. **Approvals and unattended runs are respected.** A plugin never routes around
   Hermes's approval system: no auto-approving, no disabling guards, and no
   spawning Hermes or shell children that inherit YOLO or non-interactive mode
   to run commands nobody approved. Anything that waits for a person (a prompt,
   an OAuth browser flow) fails cleanly or times out under cron, the messaging
   gateway and other unattended runs instead of hanging the agent.
13. **Risky behaviour is disclosed.** What a user would want to know before
   installing goes in the PR description and the plugin's README: network calls
   to third-party services, reads outside the plugin's own data, shell commands,
   long-running background processes, stored credentials. Telemetry and usage
   reporting are opt-in. Reviewers summarise these as disclosure lines on the
   entry PR; undisclosed behaviour found in review is a request for changes.
14. **Compatibility metadata is truthful.** `requires_hermes` is a SemVer floor
   (`">=0.21.5"`), never a CalVer date, and never newer than the current release
   (the loader skips the plugin otherwise). `version` matches the pinned code,
   and Python dependencies resolve under Hermes's core constraints
   (`hermes plugins validate --install-deps` is what CI runs).
15. **No skins or forks of bundled plugins.** A change to a bundled plugin is a
   PR against `hermes-agent`, not a competing listing, and vendor-lookalike skins
   are not listed under Nous branding.
16. **One listing per plugin lineage.** A fork of a listed community plugin is
   listed only when it is materially different from the original: a different
   transport or architecture, or capability the original lacks and its author
   declined or has not answered a PR for 30 days. Improvements to a listed plugin
   go upstream as a PR to its author. A fork that renames, rebrands or adds small
   changes is declined in favour of the original. A listed fork names its origin
   in its disclosure line (`Derived from <entry>`).
<!-- admission-rules:end -->

The step-by-step submission guide, with the same rules and what reviewers check,
lives at
[Submitting to the plugin catalog](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission).

## Entry schema

```yaml
name: example-plugin        # [a-z0-9_-]{1,64}, the catalog key
repo: https://github.com/owner/repo   # https:// only
sha: <40-hex commit sha>    # mandatory exact pin
subdir: ""                  # optional; plain relative path inside the repo
                            # ([A-Za-z0-9._/-]+ only: no '..', '.', empty
                            # segments, absolute, or backslash forms)
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
screenshots: []             # optional, up to 6 https images on a GitHub host; gallery on /docs/plugins/<name>
readme: true                # optional, default true; the README at the PINNED SHA renders on /docs/plugins/<name>
platforms: []               # optional, e.g. [linux, macos]; empty = all
title: ""                   # optional human name on cards ("NVIDIA App"); default = name
onboarding: false           # optional; offered on the Desktop onboarding card (official entries only)
known_issues: []            # optional strings shown at the install prompt; informational, never blocks
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env: []
```

`version`, `image`, `screenshots` and `readme` are cosmetic: none is parsed or
used to pick what installs. The sha stays the release; bump `version` in the
same PR that bumps `sha` so the label on the card matches the code. Images must
live on `raw.githubusercontent.com`, `github.com` or `*.githubusercontent.com`
so the Desktop catalog and the docs site never fetch from third-party hosts;
pin the raw URL to the entry's commit and the picture is as immutable as the
code.

Every entry gets a page at `https://hermes-agent.nousresearch.com/docs/plugins/<name>`
and every maintainer a page at `/docs/plugins/by/<maintainer>`, both generated
from these files at docs build time. `screenshots:` fills the page's gallery;
the build fetches the README (from `subdir` if set, else the repo root) **at the
pinned sha** — GitHub and GitLab repos, on by default, `readme: false` opts out — and renders it
through an allowlist (raw HTML dropped, links and images resolved against the
pinned tree). The page therefore shows the README the reviewer read, and it
changes only when a reviewed re-pin lands.

## removed.yaml — the blocklist

When an entry is pulled from the catalog for security or policy reasons, it
is recorded in `removed.yaml` with a reason and date. The installer refuses
to install anything matching a removed entry's name or repo URL, so a
malicious plugin cannot be re-installed from a stale identifier after
removal. A plugin that lands on the list after a user installed it stops
updating, cannot be enabled and is refused at load time. Removals, like
additions, land via reviewed PRs.

Delisting is different from removal: an entry that is unmaintained, superseded, squatting a
name it is not affiliated with, or no longer meets a rule is deleted from the catalog (plain
file removal, users who already installed it are unaffected). It is welcome back once the
problem is fixed; a name squatter comes back under a distinct name. When a new rule affects
entries that are already listed, their authors get an issue on their repository explaining
the change and time to update before anything is delisted.

## Names

The catalog key and the manifest `name:` are what users search, install, and — for memory
providers — put in `memory.provider`. A `memory` / `exclusive` entry must not reuse the name of
another provider or of a well-known upstream project it is not affiliated with: two providers
registering the same `register_memory_provider` name make `memory.provider` ambiguous
(whichever loads last wins). Reviewers check the registered provider name, not just the file
name, and the affiliated project gets the bare key.
