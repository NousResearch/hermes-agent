# Native Rust port of Hermes Desktop

The Rust target lives inside the official desktop application. The source baseline is NousResearch/hermes-agent commit `0284ce17d9bf9ed2f1832ad8950dfac2394d45ab`. The original renderer, Electron host, and their tests remain in this checkout as executable references during migration. The normal upstream release still uses Electron. The native development build uses Iced and runs the agent in Rust.

The working native features and outstanding behaviors are recorded against the public v0.23.3 release. [Feature parity audit](FEATURE_PARITY.md)

## Run

From `apps/desktop`, run `cargo run --locked -p hermes-desktop`. To start a new Nous device sign-in, append `-- --sign-in`. On macOS, `scripts/package-macos.sh` builds an application bundle containing the native desktop and gateway binaries. The development bundle has its own application identifier and uses the original Hermes icon.

A valid access token from an existing Hermes CLI Nous session can be used read-only. A fresh desktop sign-in stores its own rotating OAuth credentials with owner-only file permissions. The app requests inference access. The model selector accepts a Nous model ID. The free model used for live verification was `qwen/qwen4-max:free`. To select it for one launch, set `HERMES_RUST_MODEL=qwen/qwen4-max:free`.

By default the port stores its native catalog, transcripts and provider preferences in its own OS application-data directory. `HERMES_RUST_DATA_DIR` overrides it. Importing an upstream SQLite transcript creates a native copy. The original and imported conversations then have independent histories.

## Source and behavior mapping

| Original implementation | Native implementation | Verification and scope |
| --- | --- | --- |
| `src/app`, `src/components/chat/intro.tsx`, `wordmark.tsx`, `src/styles.css` | `crates/desktop/src/{main,ui,welcome}.rs` | Native desktop shell, chat, streaming transcript and composer. The original first-launch screenshot is the visual reference. Full visual, keyboard, accessibility, and internationalization parity remain open. |
| `src/app/settings` and CLI Nous authorization | `crates/desktop/src/settings.rs`, `crates/engine/src/portal.rs` | Nous device code, inference-scoped access, token refresh, and provider selection. Other settings pages have additional controls to port. |
| Upstream provider streaming and agent tool loop | `crates/engine/src/{provider,lib,tools}.rs` | Rust model requests, streamed tool arguments, explicit approval, actual execution, model follow-up, durable transcript. Tests cover denial, cancellation and interrupted streams. |
| `apps/shared/src/gateway-contract.openrpc.json`, `tui_gateway` | `crates/gateway` | Real local HTTP/WebSocket server, scoped session RPCs, tickets, events, approvals, and replay. See the gateway crate README for the supported method list. |
| Upstream session SQLite schema and project/session navigation | `crates/library`, `crates/desktop/src/imports.rs` | Native catalog and session switching plus read-only import from upstream. Synthetic database fixtures cover WAL, compression, and source preservation. Live cross-client upstream database writes remain open. |
| `src/app/right-sidebar/files`, `src/app/right-sidebar/review` | `crates/desktop/src/files.rs`, `git.rs` | Project-scoped local reads and previews, real Git status and diff. Editing, publishing and advanced previews remain open. The local terminal uses a native pseudoterminal. |
| MCP tools | `crates/engine/src/mcp.rs` | Optional explicitly configured Streamable HTTP server via `HERMES_RUST_MCP_URL` and optional `HERMES_RUST_MCP_TOKEN`. Discovery and invocation use protocol negotiation. Stdio server management remains open. |

The original application also includes messaging, bots and groups, scheduled jobs, webhooks, plugin pages, cloud and remote connections, voice, attachments, advanced previews, profile switching, signed updates, and platform integrations. These are not all available in the Rust build. A working chat turn does not establish parity for these features.

## Tests

Run `cargo test --workspace --locked` and `cargo clippy --workspace --all-targets --locked`. Local integration tests require permission to bind loopback sockets. The live desktop interaction test is opt-in. It requires a fresh, explicitly assigned data directory and a valid Nous login. It sends a synthetic clock request to the free Qwen model, approves only `get_current_time`, verifies the timestamp in the reply, and checks the transcript after reopening the native desktop state.

The reference application's existing Playwright boot suite was also run against a disposable official build of the pinned commit. Its 1220 × 800 logical-pixel `boot-ready-actual.png` provides the baseline used during this port.
