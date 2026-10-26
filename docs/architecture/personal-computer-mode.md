# Personal Computer Mode

**Status:** initial implementation in progress. The desktop entry point and persistent task ledger are in this branch; cross-channel task creation, Android UI, and end-to-end automation are not yet wired.

## Product goal

Give each Hermes profile a persistent, visual computer that can be opened from a dedicated **Computer** section in the desktop app. Hermes should be able to operate a Chromium-family browser with visible pointer/keyboard interaction, while the person can watch, take over, and approve consequential actions. The same profile remains the authority for its existing terminal, configuration, memory, and `SOUL.md`; this feature must not silently create a second agent identity or pretend that two different execution environments are the same.

## Existing code to reuse

- `tools/computer_use/` and `plugins/computer_use/`: existing cross-platform computer-use provider contract and CUA backend.
- `hermes_cli/browser_connect.py`, `hermes_cli/browser_runtime.py`, and `plugins/browser/`: Chromium discovery, CDP connection, and browser providers.
- `tools/bot_desktop/` and the gateway `display.*` RPCs: the current headless Linux display, authenticated noVNC stream, and human/agent input lease.
- `apps/desktop/src/plugins/hermes-bots/screen-pane.tsx`: existing live screen viewer, start/stop controls, clipboard handling, and human takeover.
- The current terminal tool and active gateway/profile: the computer view should reuse the same host/profile and filesystem where supported, rather than launching an unrelated sandbox.

The existing Bot Desktop screen is a headless Linux display on the gateway; it is not a native GUI stream for every host. The computer-use provider supports additional platform-specific paths, but they need a separate UI and real platform tests. Do not label Linux-only display controls as available on macOS, Windows, or Android.

## Changes in this branch

1. Adds `/computer` as a first-class route and sidebar entry in the desktop app's existing Hermes Bots plugin.
2. Reuses the existing Screen pane to open a profile's live desktop; it does not fork or duplicate the noVNC/control-lease implementation.
3. Adds `hermes_cli/computer_mode.py`, a SQLite task ledger with stable task IDs, persistence, status transitions, audit events, and explicit approval states.
4. Adds tests for persistence, transitions, approval, and denied-approval bypass prevention.

The task ledger is deliberately not an execution engine: it does not run shell commands or click websites by itself. It must be connected to the gateway/API and messaging adapters before it can coordinate real work.

## Target architecture

- **Computer UI:** a dedicated tab/page inside Hermes Desktop; later reuse the same view model in the Android app and other clients.
- **Computer backend:** existing browser and computer-use providers. The active profile's terminal and browser must be reused where supported.
- **Task control API:** create/list/inspect/pause/resume/cancel/approve tasks, with one durable task ID across clients.
- **Persistent task store:** SQLite initially, located in Hermes' private data directory; keep prompts/results appropriately private and avoid storing credentials.
- **Messaging adapters:** a configured WhatsApp or other channel creates/inspects tasks via the same API. It must authenticate the sender and never use an incoming message alone as permission to send mail, make purchases, or alter important files.
- **Cross-platform clients:** desktop first, then Android, then other clients against the same API. A client should report unsupported display capabilities clearly rather than showing a fake live computer.

## Safety requirements

- Human takeover and backend-enforced input leases, not just a disabled UI button.
- Require approval before sending messages, submitting forms, purchases, destructive file changes, account/security changes, or other consequential external side effects.
- Record approval decisions and task-state changes in an inspectable audit log.
- Pause/stop/revoke controls must be available from the UI.
- Use least-privilege credentials and keep secrets out of screenshots, task logs, and the APK.
- Browser navigation must use existing URL/SSRF safeguards and must not expose loopback or private services to an untrusted remote requester.

## Next implementation phases

1. **Gateway API:** expose the task ledger through authenticated RPC/HTTP methods and tests; ensure idempotency and ownership checks.
2. **Computer interactions:** connect the Computer page to start/stop/status and the existing authenticated display stream; make browser actions go through the existing browser/CDP backend.
3. **Approval inbox and notifications:** present pending approvals and reminders; prepare email drafts but do not send without approval.
4. **WhatsApp and other channels:** map authorized senders to profiles and use the same task IDs and permissions.
5. **Android client:** add the Computer section as a client of the same API; do not try to embed the whole desktop agent in the APK before its Python dependency strategy is verified.
6. **macOS/Windows/Linux validation:** use the native computer-use provider where available; explicitly document which hosts support a live desktop stream versus browser-only automation.
7. **Upstream contribution:** split reusable generic changes into small PRs to NousResearch/hermes-agent with tests, screenshots/video of an end-to-end run, compatibility notes, and a concise rationale. Keep fork-specific branding and UI customization separate where possible.

## Acceptance criteria

- A task started from a supported messaging channel can be inspected from desktop and Android with the same task ID.
- A restart does not erase queued tasks or approval history.
- The user can see the pointer and live browser interaction when the selected backend supports a display.
- Terminal and browser operate under the intended profile/host, with no false claim that a separate display is the same as a local desktop.
- A denied approval cannot be bypassed by changing task status.
- Each platform has CI coverage and a smoke test before being called supported.
