---
name: drive-electron-apps
description: "Use when driving Electron desktop apps (Obsidian, Slack)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, windows, linux]
metadata:
  hermes:
    tags: [electron, cdp, desktop, automation, browser]
    category: desktop
    related_skills: [computer-use]
---

# Drive Electron Desktop Apps over CDP

Electron apps (Obsidian, Slack, VS Code, Discord, Notion, Figma desktop, …)
are Chromium renderers in a desktop shell. Chromium **drops synthetic
pointer events into occluded, unfocused renderers**, which is why
`computer_use` background clicks often come back `suspected_noop` on these
apps. The higher-fidelity route is a **CDP attach**: the user exposes the
app's DevTools protocol port, you drive the DOM directly through
`browser_exec` — exact selectors, real JS, zero focus steal, no window
flash.

## The one-command flow

```
hermes browser attach            # USER runs this, not you
```

It scans running processes for Electron apps, finds or provisions a CDP
endpoint, and registers it as a named `browser_exec` session. Then you
drive the app with:

```
browser_exec(code="print(page_info())", session="obsidian")
```

Everything `browser_exec` offers works against the app: `js()`, `cdp()`,
`fill_input()`, the workspace, screenshots.

`hermes browser list` shows attached sessions; `hermes browser detach
<name>` removes one.

## Consent — hard rule

A CDP attach exposes **everything the app can access** (Slack DMs, vault
contents, editor buffers, tokens in localStorage) to local debugger
connections. That grant belongs to the user:

- **Never** relaunch the user's app with `--remote-debugging-port` yourself
  via terminal, and never open debug ports on their apps.
- If no endpoint is attached, **ask the user to run `hermes browser
  attach`**. Its relaunch confirmation prompt is the consent moment.
- Same doctrine as cua-driver's `grant_existing_profile` gate — this route
  is the desktop-app equivalent and gets the same respect.

## When to use which tool

- **Endpoint attached** (session exists in `hermes browser list`, or the
  user says so) → `browser_exec(session=…)` is the preferred way to drive
  that app's UI.
- **No endpoint, quick task** → `computer_use` background ladder first; it
  often works (AX-routed input reaches many Electron controls). Do NOT
  predict failure from the app being Electron — escalate only on returned
  signals (`suspected_noop`, refusals), per the computer-use skill.
- **Escalation arrives with `alternative: "cdp_attach"`** → that is the
  runtime telling you the renderer refused background input and the target
  is Electron. Ask the user to run `hermes browser attach`, then switch to
  `browser_exec`.
- **App state that lives in files** (Obsidian notes, VS Code settings.json)
  → file tools remain better than any UI automation.

## Detecting Electron yourself (when scanning is unavailable)

- `resources/app.asar` (or `resources/app/package.json`) next to the
  executable — on macOS under `Foo.app/Contents/Resources/`.
- Child processes carrying `--type=renderer` / `--type=gpu-process`.
- A crashpad handler child process.
- Real browsers (Chrome, Brave, Edge) ship none of these — they are NOT
  Electron and already have first-class routes.

## Relaunch recipe (for reference — the attach command does this)

Chromium's **single-instance lock** means the app must FULLY quit first
(including tray/background instances), or the new invocation forwards to
the old process and exits without opening the port.

- macOS: quit the app (AppleEvent), then
  `open -a <App> --args --remote-debugging-port=<port>`
- Windows/Linux: quit, then `<exe> --remote-debugging-port=<port>`
- Pick a free port that is NOT 9222 (9222 belongs to the `/browser
  connect` debug-Chrome flow; colliding steals its slot).
- Verify with `curl http://127.0.0.1:<port>/json/version`.

## Target picking

`GET /json/list` on the endpoint enumerates targets. Drive entries with
`"type": "page"`; skip `devtools://`, `chrome-extension://`, and
service-worker targets. Multi-window apps expose one page target per
window — match on `title`/`url`. With `browser_exec` this is mostly
handled for you; `ensure_real_tab()` recovers from landing on an internal
target.

## Framework input patterns (the part that saves you an hour)

Electron apps are mostly React/Vue with controlled inputs — DOM
`element.value = x` does NOT register with the framework's state.

**React controlled input — use the native value setter, then fire the event:**

```python
js("""(() => {
  const el = document.querySelector('input[aria-label="Search"]');
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, 'my query');
  el.dispatchEvent(new Event('input', { bubbles: true }));
})()""")
```

For `<textarea>`, use `HTMLTextAreaElement.prototype`. For
contenteditable surfaces (Slack/Discord composers), prefer focusing the
element and using `cdp('Input.insertText', text='…')` — it goes through
the real input pipeline.

**Radix/Headless-UI menus (Slack, Linear, many Electron apps): the menu
closes on focus loss between evals — do open + click in ONE `js()` call:**

```python
js("""(() => {
  document.querySelector('[data-state][aria-haspopup]').click();
  const item = [...document.querySelectorAll('[role="menuitem"]')]
    .find(e => e.textContent.includes('Settings'));
  if (item) item.click();
  return item ? 'clicked' : 'menu item not found';
})()""")
```

**Keyboard shortcuts** that the app binds globally: dispatch via
`cdp('Input.dispatchKeyEvent', …)` rather than DOM KeyboardEvent — apps
often listen at the Electron/webContents level.

## Pitfalls

- **Electron rejects `Target.createTarget` ("Not supported")** — you cannot
  open new tabs in an app the way you can in Chrome. `new_tab()` fails;
  Hermes routes app sessions onto the harness's attach-to-existing-page
  path automatically, but anything that tries to create a target will
  error. Work within the app's existing page target(s).
- **Don't `goto_url()` in an app session** — that would navigate the app's
  own window away from its UI (Obsidian is `app://obsidian.md/index.html`).
  Work within the existing target; `page_info()` + `js()` are your
  primitives.
- Many Electron apps expose a global app object (`window.app` in Obsidian)
  — often far richer than DOM scraping. One `js()` call against it beats
  ten selector queries; explore it with
  `js("Object.keys(window.app ?? {}).join()")` first.
- The app's window is a real window on the user's desktop: DOM clicks via
  CDP don't steal focus, but anything that opens NATIVE surfaces (file
  dialogs, context menus, notifications) leaves CDP's reach — hand those
  to `computer_use`.
- Some apps gate DevTools in production builds; if `/json` never answers
  after a debug-port relaunch, the app may strip the switch — report that
  honestly instead of retrying.
- The registry entry outlives the app process: if the app was restarted
  WITHOUT the debug port, the session's endpoint is dead. Symptom:
  browser_exec connection errors. Fix: user re-runs `hermes browser attach`.
