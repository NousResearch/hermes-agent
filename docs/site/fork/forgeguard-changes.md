---
title: ForgeGuard changes
description: What the ForgeGuard fork adds on top of upstream Hermes Agent — runtime and CLI images, desktop installers, the release scheme, and CI guards.
order: 60
status: stable
---

# ForgeGuard changes

ForgeGuard maintains a distribution of upstream
[`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent). It
tracks upstream **tagged releases** and adds a packaging and release overlay. It
does not change how the Hermes agent behaves. This page summarises what the fork
adds.

## Runtime images

One image variant is published to `ghcr.io/forgeguard-ai/hermes-agent` from the
multi-target `Dockerfile`:

- **`runtime-*`** — a full supervised server image. s6-overlay supervises the web
  dashboard and per-profile gateways; a boot reconciler restores gateways after a
  restart. Browser tools and messaging + Matrix adapters are baked in. Layout
  matches upstream's Docker image (`/opt/hermes` install, `/opt/data` state
  volume).
A second **`cli-*`** variant — a lean interactive image for distrobox / one-off
CLI use — was published until **v0.20.7 retired it** along with its Dockerfile
stage. Agent Command's distrobox deployment kind is gone (`remote-docker` is the
only kind and it deploys `runtime-*`), so it cost a build slot on every release
for no consumer, and because the release job needs the image build, a failing
cli build blocked the whole release. Tags published before v0.20.7 still pull
and are frozen at v0.20.6.

The published image carries the OCI labels `com.forgeguard.hermes.prebaked=1`
and `com.forgeguard.hermes.variant=runtime`. See
[Runtime images](../deployment/runtime-images.md) and
[Distrobox / CLI image](../deployment/distrobox-cli.md).

## Desktop installers

Prebuilt Hermes Desktop installers are attached to each fork release:

- **Linux:** `.AppImage`, `.deb`, `.rpm` (unsigned).
- **macOS:** `.dmg`, `.zip` (ad-hoc signed, **not notarized** — no Apple
  Developer credentials on this fork).
- **Windows:** `-setup.exe` (NSIS, per-user) and `-portable.exe` (unsigned, x64;
  SmartScreen "More info → Run anyway" on first launch). From `v0.20.4`.

See [Desktop artifacts](../deployment/desktop-artifacts.md).

## Release and version scheme

Releases are tagged with the Hermes Agent product version they ship (from
`pyproject.toml`), e.g. `v0.19.0`. A re-cut of an already-released product
version — such as a fork-only fix before the next upstream sync — adds a
`-forgeguard.<n>` suffix (`v0.19.0-forgeguard.2`, counting the plain tag as
cut 1). The upstream release the fork's `main` is synced to (recorded in the
`FORK_UPSTREAM_BASE` marker) appears in each release's notes for traceability.
Releases up to `v2026.7.1-forgeguard.6` used older date-shaped
`<upstream-base>-forgeguard.<n>` tags. Image `-<version>` tags are immutable;
`-latest` tags roll. See
[Releases and upgrades](../operations/releases-and-upgrades.md) and
[Image tag families](../reference/image-tags.md).

## Fork CI and workflow guards

The fork adds release automation (a release-on-merge orchestrator that calls
reusable image and desktop build workflows) and guards that prevent
upstream-only publishing, deploy, and scheduled workflows from running on the
fork. These are maintainer-facing and are documented under
[`docs/maintainers/`](https://github.com/forgeguard-ai/hermes-agent/tree/main/docs/maintainers);
they are not needed to consume the artifacts.

## ForgeGuard-only container behaviour

Beyond upstream's Docker image, the ForgeGuard images add: the multi-target
`Dockerfile` split (runtime vs CLI), the `com.forgeguard.hermes.*` labels,
pre-baked distrobox host-integration in the CLI image, and the
`HERMES_UID`/`HERMES_GID` (with `PUID`/`PGID` alias) volume-ownership remapping.
Both images also set `HERMES_DISABLE_UPDATE_CHECKS=1`: a release-pinned image
never asks upstream whether it is behind (the banner check, `hermes update`,
`GET /api/hermes/update/check` all short-circuit) — a standalone switch, so
offline mode proper stays the profile's decision. Dashboard authentication,
persistence, ports, and health semantics otherwise follow upstream.

## Desktop app behaviour

The fork's desktop builds default to **Client Mode**: the first-run chooser
preselects connecting to an already-running Hermes backend (self-hosted
container, VPS, home server), with a local backend as the secondary choice.
The connection dialog supports an opt-in TLS bypass for self-signed
certificates and remembers recent endpoints. See the upstream
[desktop guide](https://hermes-agent.nousresearch.com/docs/user-guide/desktop)
for the shared product behaviour.

Several remote gateways can be registered side by side in **Settings →
Connections**, and each one keeps its own sign-in. Upstream gives every
non-primary password- or cookie-authenticated gateway its own cookie jar, but
decides which jar from the gateway's *current* status — so a sign-in made
before pressing Save, or a **Make primary**, left the session in a jar nothing
read any more, and switching to that gateway reported "not signed in". The
fork carries the session to the right jar whenever a save, a primary change or
an apply moves it. The self-signed opt-in is also available per gateway in the
same editor (upstream's registry dropped it), and a server that refuses the
hostname outright (TLS "unrecognized name" — a reverse proxy with no site for
it) is now reported as that rather than as a certificate error.

Three further fork defaults, all in service of a static self-hosted client:

- **No update checks.** Upstream's desktop polls its own repository (`git
  ls-remote` / a GitHub compare every 30 minutes), decorates the version pills
  with "N commits behind", and raises an "Update ready" toast; there is no
  config flag for any of it. The fork turns the whole mechanism off at build
  time (`apps/desktop/src/lib/fork-config.ts`): the poller never starts, the
  toast never shows, and the client/backend version pills are plain, hideable
  readouts.
- **The context meter starts visible.** Upstream `v2026.8.16` hid it by
  default; on a self-hosted gateway with a large window it is the readout the
  operator watches most, so the fork shows it (the status bar's context menu
  still hides it).
- **Cmd+Q quits in one press.** Upstream's quit handler deferred for its
  async teardown without returning, tore the overlays/PTYs down on the
  cancelled pass and left the process in the Dock until a second Cmd+Q; the
  fork runs the teardown once and lets the last window closing end the process
  during a quit (`v0.20.5`).
- **Re-authentication after the backend is recreated never costs settings.**
  A recreated agent container answers the client's WebSocket-ticket refresh
  with 401; the fork carries the HTTP status on that error, drops only the
  stale native token, and offers **Sign in** on the boot-failure dialog — the
  connection, first-run choice and appearance preferences stay put. (Before
  this the dialog offered everything but sign-in, and the only way out was
  deleting the app's Application Support data.)

Some earlier fork-only desktop features have since been absorbed by upstream
and are no longer fork deltas: the Settings → Appearance **Text Size** control
shipped in fork releases on the `v2026.7.1` base was superseded in upstream
`v2026.7.20` by the equivalent **UI Scale** setting (same persisted zoom, plus
Ctrl/Cmd+wheel zoom and half-step keyboard shortcuts), which the fork now uses
unchanged. Likewise, the fork's voice-conversation **mic re-arm fix** (fork
releases `v0.19.3` and earlier) was superseded in upstream `v2026.8.16` by the
live-speech rewrite of the voice loop, which fixes the same dead-mic class
structurally; the fork keeps only its regression test. The fork's other
runtime fixes carried onto the `v2026.8.16` base remain fork deltas: the
`"no-key"` deploy-sentinel placeholder fix, the `custom:<name>` provider-slug
canonicalization (upstream converged on parts of the slug handling but not the
bare-endpoint collapse or the validate-probe fix), and the mem0
embedder-bearer scoping. Added in `v0.20.5`: a stored session's model is
restored as a per-chat pin only when the user chose it (a chat that merely
inherited the default follows a later `config.yaml` model change — the
served-model rename case — with a status line saying so); the desktop's Model
settings **Apply** against a deployment-manager endpoint (`model.provider:
custom` + `base_url`) is idempotent, keeps the endpoint credential across the
bare-custom ↔ named-entry rename, registers the named entry with the key
reference the config already holds, and collapses an undeclared
`custom:<name>` back to the routable bare endpoint; auxiliary slots on
`provider: main` read as inheriting (not "still run on main"); the settings
panel writes agent defaults from a fresh config read; the ComfyUI skill
honours `COMFYUI_HOST`. `v0.20.6`: the ComfyUI skill's WebSocket monitor sends
`X-API-Key` on the upgrade for a local host too (a fronted ComfyUI's key guard
checks the upgrade like every other request; before, `/prompt` and `/history`
passed and the monitor 401'd). `v0.20.7`: streaming TTS through the built-in
`openai` provider is hardened for local OpenAI-compatible servers (Kokoro): the
streamer's endpoint is `tts.openai.base_url` only (it no longer falls back to
`OPENAI_BASE_URL`, the LLM custom-endpoint override), `speed` and `language`
(`lang_code`) are honoured on the streaming path as on the whole-file path, a
provider failure mid-session sends an `error` frame so the desktop falls back
to the POST path (or finishes what played) instead of going silent, and
`hermes doctor --live` probes `tts.openai.base_url` with `tts.openai.api_key`.
It also retires the `cli-*` image (see [Runtime images](#runtime-images) above).
And it gives the Linux packages an app icon: `build.linux.icon` now
points at an icon *set* (`apps/desktop/assets/icons/`), because a single-PNG
icon made electron-builder install one `hicolor/1024x1024` entry — a directory
the freedesktop icon theme does not index, so the installed deb and rpm showed
no icon at all (Ubuntu 26.04, Fedora 44). `v0.20.8`: replies can be split for
speech at three granularities via `tts.streaming.chunking` —
`punctuation` (per sentence, the default), `paragraphs` (per line), or `none`
(the whole reply as one utterance, synthesized when the reply completes —
still streamed as PCM, so barge-in keeps working). The cut happens server-side
in `SentenceChunker`, so voice mode, read-aloud, CLI/TUI voice, and gateway
streaming all honour it, and the desktop Settings → Voice page gets a "Speech
Chunking" dropdown. It also moves past three dependency advisories: h2 4.4.1
(CVE-2026-71554, request smuggling), nanoid 3.3.18 (CVE-2026-67213), and
Electron 40.10.6 (CVE-2026-70606).

## Supported platforms and signing state

- Images: `linux/amd64`.
- Desktop: Linux + macOS (ad-hoc signed, not notarized) + Windows (unsigned, x64).

See [Platform compatibility](../reference/compatibility.md) for the full matrix
and [Compatibility](./compatibility.md) for the version mapping.

## Related

- [How ForgeGuard relates to upstream](./upstream.md)
- [Migrating from upstream](./migration-from-upstream.md)
- [Compatibility](./compatibility.md)
