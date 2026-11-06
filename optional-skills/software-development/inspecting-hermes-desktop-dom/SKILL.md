---
name: inspecting-hermes-desktop-dom
description: "Read the live Hermes desktop DOM/CSS over CDP."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [desktop, electron, cdp, dom, ui-verification, self-inspection]
    related_skills: [node-inspect-debugger, systematic-debugging, dogfood]
---

# Inspecting the live Hermes desktop DOM

## Overview

When you are developing `apps/desktop` and the user is running that same app
(`hgui` / `pnpm run dev`), you can read the **live rendered DOM** of the window
they are looking at — computed styles, geometry, which CSS rule actually won,
console output — instead of inferring it from `.tsx` and being wrong.

Dev-server runs open a Chrome DevTools Protocol port on `127.0.0.1:9222`
automatically. The renderer is a Chromium page, so everything DevTools can read,
a script can read.

**This does not replace looking at it.** CDP answers *factual* questions ("what
is the computed padding", "did this element render", "which selector matches").
It cannot tell you whether the result looks good. Colour balance, spacing feel,
and "is this ugly" still need the user's eyes or a screenshot. Answer facts with
CDP; hand aesthetics to the user.

## When to Use

- Verifying a UI change actually took effect in the running app
- "Why is this element still X?" — find the winning rule before editing anything
- Locating a stable selector for a component you're about to change
- Checking a design token's computed value on a real node
- Reading renderer console errors the user mentions but can't copy out

**Don't use for:** perf profiling or heap work (`node-inspect-debugger`, the
`apps/desktop/scripts/perf/` harness), or anything where the real question is "does this
look right".

## The port

Open on `127.0.0.1:9222` for any dev-server run. Closed in exactly two cases
(`apps/desktop/electron/dev-cdp.ts`):

- **packaged builds** — always, and no environment value overrides it;
- **no `HERMES_DESKTOP_DEV_SERVER`** — an unpackaged `electron .` against
  `dist/` is how the packaged app gets smoke tested, so it behaves like one.

`HERMES_DESKTOP_CDP_PORT` moves the port (`=9333`) or disables it (`=off`).

### Packaged builds: explicit user opt-in through `desktop.electron_flags`

`dev-cdp.ts` never opens the port in a packaged build, but it does not strip a
Chromium switch the user passes in. When the user explicitly wants their own
packaged window inspected, the supported launcher path is the list-valued
`desktop.electron_flags` setting (a plain string is rejected):

```bash
# Linux/Wayland: --ozone-platform=wayland is required, see below
hermes config set desktop.electron_flags \
  '["--ozone-platform=wayland", "--remote-debugging-port=9222", "--remote-debugging-address=127.0.0.1"]'
# Revert when done (the port lets any local process drive the app and read sessions)
hermes config set desktop.electron_flags '[]'
```

The user quits and relaunches the app themselves. Do not tell them to start
the binary under `release/` directly: the window may never appear, because the
launcher's environment is missing.

**Wayland relaunch pitfall.** Without an explicit `--ozone-platform`, the first
process (`electron/entry.ts` → `wslg-launch.ts`) re-executes itself with
`--ozone-platform=…` and stays alive without windows. It receives the debugging
flags too and binds the port first. The real app then logs
`bind() failed … (98)` / `Cannot start http server for devtools` in
`logs/desktop-chromium.log`, and the port answers with an empty `/json/list`.
Putting `--ozone-platform` into the flags skips the relaunch. Confirm the
owner with `ss -ltnp | grep 9222`.

Check before doing anything else:

```bash
curl -s --max-time 3 http://127.0.0.1:${HERMES_DESKTOP_CDP_PORT:-9222}/json/version
```

Empty → no port. Do not guess another port silently. A port that answers but
lists no `page` target belongs to the wrong process (see the Wayland pitfall).

**Never relaunch the user's app on your own to get a port.** That destroys their
session and their state. Launch your own isolated instance instead (below), or
use the packaged opt-in above with the user's agreement and let them restart.

## Reading the DOM

`apps/desktop/scripts/eval.mjs` is the one-liner:

```bash
cd apps/desktop
node scripts/eval.mjs "document.querySelectorAll('[data-slot]').length"
```

For multi-step work use the shared client — it has target discovery and
promise-aware eval:

```js
import { CDP, SELECTORS } from './scripts/perf/lib/cdp.mjs'

const cdp = await CDP.connect({ port: 9222, match: '5174' })
const out = await cdp.eval(`JSON.stringify({
  radius: getComputedStyle(document.documentElement).getPropertyValue('--radius-scalar').trim(),
  composer: !!document.querySelector('[data-slot="composer-rich-input"]')
})`)
cdp.close()
```

`SELECTORS` in `scripts/perf/lib/cdp.mjs` holds the stable `data-slot` hooks
(composer, thread viewport, assistant message, turn pair, profile rail). Prefer
them over inventing a `querySelector` — they are updated as a unit when
components move.

## The question this is best at: which rule won?

Editing every call site because a style "isn't applying" is the classic waste.
Read the real node first:

```js
const el = document.querySelector('[data-slot="aui_assistant-message-root"] a')
JSON.stringify({
  ownClasses: el.className,
  weight: getComputedStyle(el).fontWeight,
  parents: (() => {
    const out = []
    let n = el
    while ((n = n.parentElement) && out.length < 6) out.push(n.className)
    return out
  })()
})
```

If the node carries no class of its own, the value is **inherited** — sweeping
call sites will not fix it, and you need the ancestor rule. A plugin stylesheet
(e.g. `@tailwindcss/typography`'s `prose a { font-weight: 500 }`) routinely beats
a utility class; override on the shared class, not at each usage.

## Driving the user's window

Only drive input after the user agrees, and stop when they are working in it.

- Check before and after every step that the layout is the one you expect: for
  two session tabs in one stack, exactly one `[data-slot="pane-tab"]` with
  `aria-selected="true"`. Abort on any drift. A session switch does not change
  `location.hash`, and a stable message count is not proof that a switch happened.
- `element.click()` on a pane tab does not activate it. Synthetic CDP mouse
  presses on a tab can start a drag (`onEngage`) and move the tab into another
  split, which changes the user's layout. Activate tabs from the keyboard instead:
  `focus()` the tab, then `Input.dispatchKeyEvent` Enter (`pane-tab.tsx` handles
  Enter/Space).

## Your own isolated instance

When there is no port, or you must not disturb the user's window:

```bash
cd apps/desktop
HERMES_HOME=$HOME/.hermes/cache/scratch/cdp-probe-home \
HERMES_DESKTOP_DEV_SERVER=http://127.0.0.1:5174 \
HERMES_DESKTOP_CDP_PORT=9333 \
  pnpm exec electron . --user-data-dir=$HOME/.hermes/cache/scratch/cdp-probe-userdata
```

The separate `--user-data-dir` dodges Electron's single-instance lock, so it
cannot collide with a running `hgui`; the separate `HERMES_HOME` keeps it away
from real sessions. Pick a port other than 9222 for the same reason. Run it in
the background and kill it when done.

`pnpm run perf:serve` does the same with a temp `HERMES_HOME` baked in, if you
also want the perf harness.

## Pitfalls

- **Never kill the user's dev server or app to "free" anything.** A mid-serve
  kill nukes Chromium's socket pool, and the resulting `ERR_NETWORK_CHANGED`
  gets blamed on whatever you just changed.
- **A throwaway `HERMES_HOME` has no backend.** The app logs `ECONNREFUSED` for
  `hermes:api` and may exit on its own. The renderer still mounts and the DOM is
  readable — read promptly, and don't mistake a self-exited probe for a broken
  port. Chromium logs `DevTools listening on ws://127.0.0.1:<port>/…` when it
  binds; that line is the proof the port opened.
- **Poll, don't probe once.** A just-launched app needs a second or two before
  the port answers.
- **Never dump the whole DOM.** The desktop renders hundreds of nodes and
  `outerHTML` will bury your context. Project down to a small JSON object inside
  the evaluated expression.
- **Pass `match` to `CDP.connect`.** Without it you may attach to the pet
  overlay, quick-entry window, or a devtools target instead of the main window.
- **`cdp.eval` returns the value; raw `Runtime.evaluate` double-nests it**
  (`.result.result.value`). Use the wrapper.
- **`import.meta.env.DEV` is `true` under `vite dev`** in this repo. The note in
  `apps/desktop/scripts/profile-typing-lag.md` claiming otherwise is stale.
