# Realms — optional bundled Hermes plugin

Each conversation can use its own Linux GUI desktop while you keep working on yours. Private labwc/Xwayland compositors keep parallel agents from fighting over the host's windows, pointer, clipboard and focus. This is especially useful on tiling desktops such as Hyprland. **It is GUI separation, not a VM or a security sandbox:** ordinary realm terminals retain project filesystem and network access.

## Proposal dependency

This follow-up requires [NousResearch/hermes-agent#103690](https://github.com/NousResearch/hermes-agent/pull/103690) to merge first. Historical standalone integration used `fa6e9ac63155780dc368bf02f86c34413d441788`; this bundled proposal uses the refreshed prerequisite `6f60473db6be55d362c0cc5ab09b018a1f5c0b84`. Do not advertise compatibility with Hermes versions lacking those generic session ownership, execution-context and desktop viewer APIs.

The implementation is adopted from [hermes-realms](https://github.com/Zeus-Deus/hermes-realms) at `fbea3060e63621c1c73b7a6d5520dc6ff58e3227`. Runtime, desktop, dashboard and skill live together here. Desktop's build discovers this same `desktop/plugin.js`; there is no second implementation, downloader stub, or dependency on a separately installed `realms` package. Local adaptations add native CLI registration, an explicit desktop opt-in, writable profile-local driver placement, and rejection of unsupported platforms before runtime registration. Text I/O uses explicit UTF-8; Linux-only ownership and process primitives remain intact. Original MIT attribution and all vendored noVNC/pako notices are preserved.

## Off by default

- Native discovery inventories `kind: standalone` but does not import its Python until `plugins.enabled` includes `hermes-realms`. Explicit `plugins.disabled` wins.
- Its hooks, model tool, slash command, administrative CLI and skill are absent while disabled. No binary download, compositor, systemd scope, viewer listener or desktop contribution starts on discovery.
- The dashboard API follows the same opt-in gate, including revocation of already-mounted routes. Enabling the Python plugin does **not** enable the desktop UI, or vice versa.
- The desktop half appears in Settings → Plugins, with `defaultEnabled: false`; its decision persists using the native plugin inventory. Backend activation is per profile, with no config inheritance from other profiles.

## Enable and prepare (Linux only)

The runtime needs labwc, Xwayland, WayVNC, grim, wlr-randr, bubblewrap, D-Bus, the AT-SPI bus launcher/registry, a working systemd **user** manager and a render node. Install these explicitly with your distribution's package manager. This plugin does not install system packages, services, or alter/reload Hyprland configuration.

**The pinned driver installer supports Linux x86-64 only.** macOS, Windows and Linux ARM do not have a supported installer. Do not enable native Realms on unsupported hosts. Enabling without prerequisites does not authorize host fallback.

For the profile you intend to use:

```sh
hermes plugins enable hermes-realms
hermes realms --help
hermes realms install-driver
hermes realms doctor
```

`install-driver` is the **explicit** network/install action. It downloads cua-driver 0.23.2, verifies the archive and executable SHA-256 pins, and installs only to the active profile's `plugin-data/hermes-realms/bin/cua-driver`. Native registration never invokes it. Sealed/packaged/Nix source directories need not be writable. `--archive /path/to/approved-release.tar.gz` uses a previously downloaded release without networking; `--target` is an administrative export override, not runtime configuration. Normal runtime uses the profile-local path. The global Cua binary is not modified.

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

The two Linux native tests in the files below are marked `integration`, excluded by the ordinary suite's default `-m 'not integration'`. In an **explicitly approved, disposable Linux test environment** with the system prerequisites above, run from the Hermes checkout:

```sh
HOME="$(mktemp -d)" scripts/run_tests.sh \
  tests/plugins/test_bundled_realms_cli.py \
  tests/plugins/test_realms_import_runtime.py \
  -m integration --file-retries 0 -j 1 -W error
```

The runner forwards `-m integration` to pytest, overriding the default marker selection; `--include-integration` alone only changes directory discovery and does not select marked tests. This exact native selection was exercised against the bundled proposal on the refreshed prerequisite: **2 passed, no failures or skips**. It preserves real CLI stdout/stderr and exit 0/7, captured/FD jobs, a generated contained launcher running `/usr/bin/true`, PNG capture, foreign-module isolation, and runtime/process/systemd-unit/registry cleanup. It does not install or exercise the pinned Cua binary or send input. Full private Cua, native Watch/Pop out and window behavior require separate approved DEV acceptance. Nix source wiring is included; a full Nix build is a separate lane.

## License

Original code: [MIT](LICENSE). Vendored assets: [third-party notices](realms/web/THIRD_PARTY.md), including all original license and author files. The MIT license does not replace the component licenses.
