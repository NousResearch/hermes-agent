# Native titlebar click regression (macOS)

From `apps/desktop`, with the checkout's pinned Node/pnpm and existing workspace dependencies:

```sh
node electron/titlebar-click-native-fixture/run.mjs
```

Requires a macOS GUI session, Xcode Command Line Tools, and Node's `include/node`
headers (or `NODE_INCLUDE_DIR` pointing to them). Other platforms print an explicit
**SKIP**, not a native-test pass. Nothing is downloaded or installed. The runner
uses `TMPDIR` for its generated bundle, compiled N-API addon, isolated homes,
Electron storage, JSON results and logs; it prints that artifact directory.

The fixture bundles the **actual `LayoutTreeRoot`, `usePanelTitlebar`,
`useResizeObserver`, and `mirrorTreeHorizontal`** from this checkout. The bundler
replaces only unrelated shell services, stores, and expensive child content with
`./shell-seam.mjs`; the same atom drives the real root and the native Swap action.
There is no duplicated root effect, fixture chrome-change dispatch, repair flag,
maximize/restore, fullscreen transition, or native-window resize.

Each of two fresh windows runs ten same-size keyed panel mirrors. Native events
must still activate Settings, Layout, HUD, both sidebar controls, Swap and a body
control. The test also checks retained DOM node identities, unchanged panel/window
sizes and that drag regions do not overlap control centers. Full per-click counts
and geometry are in each run's `results.json`; failures retain `failure.json`.

`native-pointer.mm` posts mouse-down/up `NSEvent`s through the application's Cocoa
event queue. They traverse native window/drag-region hit testing, unlike DOM
`.click()`, CDP or `webContents.sendInputEvent`. The mandatory negative control
places a native drag region beneath a higher-z-index no-drag button: Cocoa must
swallow its click while DOM and Electron renderer-input clicks still arrive. A
runner that accidentally bypasses native hit testing therefore fails this test.

This is an in-process native-window regression, **not external OS pointer
injection or a packaged whole-app end-to-end test**. It requests/changes no
Accessibility or other permissions, does not activate the fixture or move the
cursor, and asserts that the frontmost app and cursor are unchanged. It never
imports the desktop main process, connects a gateway, touches an installed Hermes
profile, or registers a protocol handler. Normal desktop Vitest runs do not open
these opt-in native windows.

Renderer regression coverage lives in
`src/components/pane-shell/tree/renderer/titlebar-tree-commit.test.tsx`: the real
root plus real hook, no ResizeObserver delivery or resize, pre-paint measurement,
non-mirror three-panel reorders, `titlebar=false`, and unrelated rerenders.
