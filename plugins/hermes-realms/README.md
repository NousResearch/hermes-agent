# Realms — optional bundled Hermes plugin

Each conversation can use its own Linux GUI desktop while you keep working on yours. Private labwc/Xwayland compositors keep parallel agents from fighting over the host's windows, pointer, clipboard and focus. This is especially useful on tiling desktops such as Hyprland. **It is GUI separation, not a VM or a security sandbox:** ordinary realm terminals retain project filesystem and network access.

For work that needs a *real* Omarchy — shell plugins, Hyprland, system-level changes — there is a second kind, **Omarchy VM**, which is a disposable Omarchy guest in QEMU/KVM with its own kernel, disk and desktop. See [Omarchy VM realms](#omarchy-vm-realms) below. The labwc realm remains the default.

## Proposal dependency

This follow-up requires [NousResearch/hermes-agent#103690](https://github.com/NousResearch/hermes-agent/pull/103690) to merge first, including its generic consented native-plugin setup lifecycle. Do not advertise compatibility with Hermes versions lacking that lifecycle or its generic session ownership, execution-context and desktop viewer APIs.

The implementation is adopted from [hermes-realms](https://github.com/Zeus-Deus/hermes-realms) at `fbea3060e63621c1c73b7a6d5520dc6ff58e3227`. Runtime, desktop, dashboard and skill live together here. Desktop's build discovers this same `desktop/plugin.js`; there is no second implementation, downloader stub, or dependency on a separately installed `realms` package. Local adaptations add native CLI registration, an explicit desktop opt-in, writable profile-local driver placement, and rejection of unsupported platforms before runtime registration. Text I/O uses explicit UTF-8; Linux-only ownership and process primitives remain intact. Original MIT attribution and all vendored noVNC/pako notices are preserved.

## Off by default

- Native discovery inventories `kind: standalone` but does not import its Python until `plugins.enabled` includes `hermes-realms`. Explicit `plugins.disabled` wins.
- Its hooks, model tool, slash command, administrative CLI and skill are absent while disabled. No binary download, compositor, systemd scope, viewer listener or desktop contribution starts on discovery.
- The dashboard API follows the same opt-in gate, including revocation of already-mounted routes. Enabling the Python plugin does **not** enable the desktop UI, or vice versa.
- The desktop half appears in Settings → Plugins, with `defaultEnabled: false`; its decision persists using the native plugin inventory. Backend activation is per profile, with no config inheritance from other profiles.

The two settings are separate layers, not alternative global/profile switches:

- **Desktop plugins → Realms** loads viewing controls throughout this desktop app. It does not enable tools, change any gateway's configuration, or start a desktop.
- **Agent plugins → Applies to → hermes-realms** enables the runtime for the selected profile on the connected gateway. Enable it only for the profiles that should use Realms; leave it off elsewhere.

Enabling the agent plugin first reviews the pinned Cua release, checksums and destination. **Set up and enable** explicitly authorizes the profile-local install and readiness checks. Cancel leaves enablement unchanged. A download, checksum, permissions, missing-system-package or user-manager failure keeps the previous setting and displays a setup error; it never authorizes host fallback. Retry after correcting the reported prerequisite.

Session badges appear only after a successful lookup confirms an owned realm. A missing or disabled backend does not label ordinary sessions as unavailable. During a temporary read failure, an existing badge keeps its last-known count and marks the status as unavailable on hover.

## Omarchy VM realms

A labwc realm structurally cannot host `omarchy-shell`: that needs Hyprland and one shell per session. The `omarchy-vm` kind runs a throwaway Omarchy guest instead, so an agent can build and prove a shell plugin, change a theme or break a Hyprland config without any of it reaching your machine.

The guest is installed by **Omarchy's own GPG-verified release ISO**, through a pinned vendored copy of [`omarchy vm`](https://github.com/omacom/omarchy/pull/10977) (see `realms/vendor/VENDOR.md`). We do not build, host or modify an image.

### What isolation means here, honestly

- **Yes, isolated.** Separate kernel, disk and desktop. A guest crash, a bad plugin, an `rm -rf`, a broken Hyprland config — all of it stays in the qcow2 and is gone when the conversation ends.
- **Files are copied, never mounted.** `/realm push SOURCE [GUEST_PATH]` copies work in; `/realm pull GUEST_PATH LOCAL_PATH` takes results out. Nothing leaves the guest unless you ask for it. There is no shared folder.
- **Network is on by default**, because plugins need `git` and `pacman`. QEMU user-mode networking also lets the guest reach host loopback services via `10.0.2.2`; set `plugins.realms.vm.network: false` for `restrict=on` when that matters. As with the labwc realm, this is **not** a hostile-code sandbox against a determined attacker.
- **SSH agent forwarding is off**, and so is X11 forwarding — pinned per connection, so a `Host *` block in your `~/.ssh/config` cannot re-enable either. The guest has passwordless sudo and no disk encryption, so forwarding your agent into it would let anything running there authenticate as you. Keep secrets out of the guest; it is disposable.
- **The guest disk has no encryption and passwordless sudo.** That is fine because it is disposable — but never put a credential, token or key in it.
- **The host's environment does not cross the boundary.** Only terminal presentation variables are forwarded; this profile's API keys are not.

### Setup (once per profile)

Needs `qemu-full`, `edk2-ovmf`, `mtools`, access to `/dev/kvm` and an `~/.ssh/id_ed25519.pub`. Install system packages with your distribution's package manager — the plugin never installs them.

```sh
hermes realms vm doctor     # what is missing
hermes realms vm install    # signed ISO (~5 GB) + unattended install, ~3 min
hermes realms vm status     # base image, storage, running realms
hermes realms vm settings   # the Desktop panel's block, as JSON
hermes realms vm settings --check-updates   # also ask GitHub for the newest release
```

`hermes realms vm clean` removes superseded ISOs and orphaned session disks; `hermes realms vm remove-base` deletes the base image.

The Desktop panel (Settings → Plugins → Realms) reads the same data over
`GET /realms/vm/settings` and drops the same files over `POST /realms/vm/clean`.
The update check is opt-in (`?check_updates=true`) because it costs a network
round trip: when GitHub cannot be reached, `update.available` is `null` —
"not checked", never a fabricated "up to date".

### Using it

`/realm on omarchy` (or `realm` with `kind: omarchy-vm`) selects the kind for that conversation; the guest starts lazily at the next terminal action. Every chat gets a **copy-on-write clone of the base image** — 196 KiB and hundredths of a second, booting in about fifteen seconds — so breakage never carries over between conversations. The session badge reads **Omarchy VM** and hover shows what it costs.

A guest holds its whole `plugins.realms.vm.memory` figure in host RAM while it runs (no balloon device; the guest touches all of it), so an idle realm is reaped on the same `idle_ttl` as a labwc one, and a finished conversation takes its guest with it.

Computer use inside the guest is not wired up yet, so `app: screen` capture and clicking are unavailable there; `/realm shot` captures the guest's own desktop with `grim` over SSH, and `/realm watch` views it over the same noVNC bridge the labwc realm uses.

### Settings (`plugins.realms` in `config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `default_kind` | `realm` | Which kind a conversation uses when it does not ask |
| `vm.memory` | `3072` | Guest RAM in MiB; applies at the next guest boot |
| `vm.network` | `true` | `false` adds `restrict=on` and refuses the guest's outbound routes |
| `vm.disk_size` | `40G` | Virtual disk size for the base image |
| `vm.boot_timeout` | `180` | Seconds to wait for a guest to answer SSH |
| `vm.omarchy_vm_path` | *(vendored)* | Absolute path to your own already-headless `omarchy vm` |

### Never escaping to the host

Realms strip the host's display, bus and input handles from every command's environment. Putting them back — `env WAYLAND_DISPLAY=… hyprctl`, `export DISPLAY=:0`, re-pointing `XDG_RUNTIME_DIR` — drives your real desktop from inside a realm, which has genuinely happened. Terminal commands that do this are now refused with a message telling the agent to ask you or use `/realm off`. This is an agent-judgment guardrail, not a containment boundary; the VM kind is what makes the boundary real.

## Enable and prepare (Linux only)

The runtime needs labwc, Xwayland, WayVNC, grim, wlr-randr, bubblewrap, D-Bus, the AT-SPI bus launcher/registry, a working systemd **user** manager and a render node. Install these explicitly with your distribution's package manager. This plugin does not install system packages, services, or alter/reload Hyprland configuration.

**The pinned driver installer supports Linux x86-64 only.** macOS, Windows and Linux ARM do not have a supported installer. Do not enable native Realms on unsupported hosts. Enabling without prerequisites does not authorize host fallback.

For the profile you intend to use:

```sh
hermes plugins enable hermes-realms
hermes realms --help
hermes realms doctor
```

The CLI enable command also asks for setup consent (default **No**). Noninteractive callers must supply the exact profile/key/revision JSON shown in the refusal through `--setup-consent`; install/force flags do not authorize setup. The approved setup reuses the pinned installer, verifies archive and executable SHA-256, runs the installed binary's `--version`, and requires a successful doctor check before recording completion. A valid existing installation avoids another download. Driver verification alone is not completed setup.

`hermes realms install-driver` remains the **explicit** repair/export action for already-enabled profiles. It installs cua-driver 0.23.2 only to the active profile's `plugin-data/hermes-realms/bin/cua-driver`. Native registration never invokes it. Sealed/packaged/Nix source directories need not be writable. `--archive /path/to/approved-release.tar.gz` uses a previously downloaded release without networking; `--target` is an administrative export override, not runtime configuration. Normal runtime uses the profile-local path. The global Cua binary is not modified. Profile-local directory symlink redirects are rejected; a damaged or symlinked binary leaf is replaced without changing its referent. A non-executable filesystem or binary execution failure is not reported as ready.

Use `hermes -p NAME ...` for each command when targeting a named profile. After enabling/disabling Python, restart that profile's backend to load/unload its hooks and API routes. Start a new conversation rather than changing an existing conversation's cached tool schema. Enable **Realms** separately in the desktop Plugins settings if you want session badges and viewing controls. A UI-only enable has no backend authority.

## Use and disable

The default mode **after opting in** is `realm`: eligible tools start a desktop lazily. No desktop is started merely by plugin enablement. Use `hermes config set plugins.realms.default_mode ask` to require a per-session choice, or `realm` for private-by-default. Existing approvals are unchanged.

- `/realm on`: select the private desktop; the next eligible tool starts it.
- `/realm status`: inspect mode and actual readiness.
- `/realm size 1280x720`: resize the private desktop.
- `/realm watch`: get a short-lived view-only viewer. Human takeover pauses agent input until returned.
- `/realm stop`: stop this session's owned processes without selecting host mode.
- `/realm off`: **only by explicit user request**, stop the realm and return subsequent tools to the ordinary host route.

Normal turn completion does not destroy the realm; actual session finalization does. Watch/Pop out require a local connection: a remote loopback URL is not a tunnel. Never persist or log viewer tickets. Do not fall back to host display, input, accessibility buses or capture after a realm error.

To remove the feature, stop active realms first, disable its desktop UI, run `hermes plugins disable hermes-realms`, and restart the owning backend. This leaves profile data and the explicitly installed binary in place; it does not delete another profile's data.

## Verification scope

Ordinary host tests execute native discovery with temporary homes and audit hooks forbidding process/network actions, including the real root CLI parser's separation of `realms` dispatch from child argv without desktop executables. Package tests build actual source and wheel artifacts, make the plugin tree read-only, and exercise discovery, registration, identity hooks, administrative CLI, invalid-archive rejection and serving vendored viewer assets. Desktop tests execute the actual bundled loader and real plugin export through off → on → off, including inventory curation. These ordinary tests do **not** prove native desktop behavior.

The Linux native tests in the files below are marked `integration`, excluded by the ordinary suite's default `-m 'not integration'`. In an **explicitly approved, disposable Linux test environment** with the system prerequisites above, run from the Hermes checkout:

```sh
HOME="$(mktemp -d)" scripts/run_tests.sh \
  tests/plugins/test_bundled_realms_cli.py \
  tests/plugins/test_realms_import_runtime.py \
  tests/plugins/test_realms_setup.py \
  -m integration --file-retries 0 -j 1 -W error
```

The runner forwards `-m integration` to pytest, overriding the default marker selection; `--include-integration` alone only changes directory discovery and does not select marked tests. Native coverage preserves real CLI stdout/stderr and exit 0/7, captured/FD jobs, a generated contained launcher running `/usr/bin/true`, PNG capture, foreign-module isolation, and runtime/process/systemd-unit/registry cleanup. Setup coverage downloads the actual pinned archive, executes the binary, checks offline reuse and repair, and rejects cross-profile redirects, invalid archives, invalid config and failed readiness-record invalidation. Native Desktop consent/cancel/failure/retry, private Cua capture, Watch/Pop out and window behavior require separate approved DEV acceptance. Nix source wiring is included; a full Nix build is a separate lane.

## License

Original code: [MIT](LICENSE). Vendored assets: [third-party notices](realms/web/THIRD_PARTY.md), including all original license and author files. The MIT license does not replace the component licenses.
