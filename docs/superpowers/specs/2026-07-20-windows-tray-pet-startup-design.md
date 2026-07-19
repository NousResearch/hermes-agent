# Windows tray and pet startup design

## Goal

Give Hermes Desktop a safe Windows notification-area lifecycle and two independent startup preferences:

1. Start Hermes with the main window hidden in the tray.
2. Automatically pop the configured pet out to its desktop overlay after startup.

The tray must remain a reliable way to restore or quit Hermes. Hiding the primary window must not stop the desktop-managed backend, active turns, secondary session windows, cron or gateway work, or an already popped-out pet.

## Scope

This design applies only to Windows. It extends the existing first-party Electron pet overlay and does not create a desktop plugin or a second pet implementation.

Included:

- Safe close-to-tray behavior.
- A tray menu for current window and pet actions.
- Persistent, independent startup preferences for starting hidden and automatically popping out the pet.
- Explicit tray cleanup during a real shutdown.
- Safe fallback when the tray or pet overlay is unavailable.
- Automated lifecycle and preference tests.

Excluded:

- Launching Hermes automatically when Windows signs in.
- Linux tray behavior or a macOS menu-bar implementation.
- Pet speech containing full assistant responses.
- Fullscreen-aware pet hiding.
- Tray badges, unread counters, agent-status icons, or plugin-contributed tray commands.

## User experience

The Windows tray context menu contains:

```text
Show Hermes / Hide Hermes
Show desktop pet / Return pet to Hermes
──────────────────────────
✓ Start in tray
✓ Pop out pet on startup
──────────────────────────
Quit Hermes
```

The first two labels reflect current state. The startup preferences are checkboxes and affect future launches only. Toggling either preference must not immediately hide the current main window or move the current pet.

- **Start in tray** means Hermes initializes normally but does not show the primary window after startup. The tray is available immediately as the recovery surface.
- **Pop out pet on startup** means the renderer invokes the existing pet pop-out flow after it is ready and the configured pet is available.
- The two preferences are independent. A user may show the main window and the desktop pet together, hide only the main window, or disable either startup behavior.
- Closing the primary window with the title-bar X hides it when a usable tray exists. It is not a real application quit.
- **Quit Hermes**, updater handoff, uninstaller handoff, and fatal restart paths retain the existing full-shutdown behavior.

## Architecture

### 1. Windows-local desktop preferences

Store the two preferences in a small desktop-owned JSON file under Electron's local `userData` directory rather than `config.yaml`:

```json
{
  "startInTray": false,
  "popOutPetOnStartup": false
}
```

Reasons:

- These settings control the Windows desktop shell, not the agent core.
- They must not sync through the shared Hermes profile to another operating system.
- Tray checkbox updates need a small synchronous or atomic local write without invoking the general Hermes configuration pipeline.

The preference module must:

- supply `false` defaults;
- tolerate a missing, malformed, or partially written file;
- ignore unknown keys;
- write atomically through a temporary file and replacement;
- expose small pure functions that can be tested independently.

### 2. Main-process tray lifecycle

The Electron main process owns:

- the `Tray` instance;
- main-window show/hide behavior;
- close-to-tray policy;
- startup visibility;
- real-quit state;
- updater/uninstaller handoff exceptions;
- tray-menu rendering and preference persistence;
- explicit tray destruction during shutdown.

Tray creation occurs before deciding whether to reveal the primary window. `startInTray` is honored only when tray creation succeeded. If no icon exists or `new Tray(...)` throws, Hermes shows the main window and retains the existing normal close lifecycle. It must never create an unreachable hidden application.

The main process still creates the renderer while starting in the tray. This preserves backend startup, session state, gateway connections, pet state production, and normal initialization.

### 3. Pet command bridge

The renderer remains the source of truth for pet state and invokes the existing pop-out and return flows. The Electron tray must not create an independent pet window with incomplete state.

Add a narrow desktop IPC contract:

- main process requests `show desktop pet` or `return pet to Hermes`;
- renderer reports whether the pet is enabled and currently popped out;
- renderer performs the existing pop-out/return action;
- renderer reports updated state so the tray menu label can refresh.

After renderer readiness:

1. Load normal pet state and configuration.
2. Report current availability and overlay state to the main process.
3. If `popOutPetOnStartup` is enabled and the pet is available but not already popped out, invoke the existing pop-out action once.
4. Report success or failure without blocking desktop startup.

The startup request must be idempotent. Reloading the renderer or receiving duplicate readiness events must not create duplicate overlays.

### 4. Dynamic tray menu

Rebuild or update the tray context menu when any of these values change:

- main window visible/hidden;
- pet available/unavailable;
- pet popped in/out;
- either startup preference changes.

When the pet is unavailable, the current-action pet item is disabled. The preference checkbox may remain configurable for the next launch, allowing it to take effect after a pet is installed or enabled.

## Lifecycle rules

### Normal launch

- Create tray on Windows.
- Load desktop preferences.
- Create the main window and initialize the renderer/backend.
- Show the main window unless `startInTray` is enabled and tray creation succeeded.
- Once renderer pet state is ready, optionally pop out the pet.

### Closing the primary window

Hide the primary window only when all are true:

- the platform is Windows;
- a usable tray exists;
- the app is not performing a real quit;
- no updater/uninstaller handoff is active.

The popped-out pet, backend, active turns, and secondary windows remain alive.

### Real quit

Before a real quit:

- mark the application as quitting;
- close the pet overlay through the existing shutdown path;
- dispose terminal sessions and stop desktop-managed backends as today;
- explicitly destroy the tray and clear its reference;
- allow all windows to close.

### Update and uninstall

Updater, swap, and uninstall handoffs bypass close-to-tray. Their PID wait must not be blocked by a hidden desktop process or surviving tray.

### Failure behavior

- Tray setup failure: log it, show the main window, and use normal exit behavior.
- Preference read failure: use defaults and continue.
- Preference write failure: leave the in-memory value unchanged, log the failure, and keep the menu usable.
- Pet unavailable: skip automatic pop-out and disable the immediate pet action.
- Pet command timeout or renderer reload: keep Hermes running, refresh state after the next renderer-ready report, and do not duplicate the overlay.

## Testing strategy

Use focused Electron Vitest tests for extracted policy, preference, and tray-action modules, plus an integration-oriented seam around renderer/tray commands.

Required behavior tests:

- Normal Windows close hides the main window only when a tray exists.
- No tray means the main window appears and the last window closes normally.
- Tray construction failure does not prevent `createWindow()`.
- Existing main window is restored and focused; a destroyed one is recreated.
- Quit marks real-quit state before calling `app.quit()`.
- Updater/uninstaller handoff is never intercepted.
- Real shutdown destroys the tray exactly once.
- Non-Windows close semantics remain unchanged.
- Missing or malformed preference files return safe defaults.
- Preference writes preserve independent checkbox values and use an atomic replacement.
- `startInTray` is ignored when no tray exists.
- Pet startup action runs only after renderer readiness and only once.
- Pet unavailable or already popped out produces no duplicate action.
- Tray pet commands reach the renderer and refresh the menu after state changes.

Verification gates:

```text
npm run lint
npm run typecheck
npm run test:desktop:platforms
npm run build
```

A Windows manual check must verify launch visible/hidden combinations, X-to-hide, tray restore, pet pop-out/return, real quit, and update handoff.

## Repository and contribution boundary

The working tree also contains unrelated QQ Bot edits. Every stage, test, diff, commit, and future PR operation for this feature must be path-scoped. Never use `git add -A` in this checkout.

The implementation should build on the existing local Windows tray lifecycle files, but it must close the remaining gaps above before being treated as contribution-ready. The feature is broader than another generic tray PR: its contribution boundary is safe startup visibility, first-party pet coordination, backend preservation, and updater-compatible shutdown.
