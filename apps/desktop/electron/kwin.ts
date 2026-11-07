// kwin.ts — window enumeration for KDE Plasma on Wayland, by asking KWin.
//
// `read_window_below` normally enumerates through `get-windows`, which on Linux
// shells out to `xprop` and reads `_NET_CLIENT_LIST_STACKING`. That is an X11
// protocol, and Wayland deliberately refuses to tell one application about
// another's windows — so on a Wayland session the tool has nothing to work
// with. Under XWayland it is arguably worse than nothing: it enumerates the few
// legacy X11 clients and silently misses every native Wayland window, which on
// a Plasma desktop is all of them. A KDE Wayland session therefore answers with
// an empty list while the compositor knows every window on screen, and an empty
// list is indistinguishable from "nothing is underneath" — the failure is
// silent rather than reported.
//
// Hyprland answers over its own IPC socket (see hyprland.ts). KWin has no
// socket, but it exposes the same information through its scripting interface
// on the session bus: a script loaded with `org.kde.kwin.Scripting.loadScript`
// runs inside the compositor and can read `workspace.stackingOrder`, which is
// the real window stacking order, bottom to top.
//
// The script reports back over D-Bus rather than through `print()`, because a
// script's stdout never reaches the journal — neither the user's nor
// plasma-kwin_wayland's. This module publishes a tiny object at
// `/probe` and the script calls it; no external tool (`dbus-monitor`,
// `kdotool`, `xdotool`) is involved, which matters because the desktop app must
// not depend on software the user has not installed.
//
// Two details of the KWin scripting host cost real time and are load-bearing
// below, so they are asserted by the tests:
//
//   - The script body cannot use `for…of` over a KWin list or an arrow
//     function. Doing so fails to parse: `loadScript` still reports success and
//     `start()` still returns, but the script sends nothing at all. From the
//     outside that is indistinguishable from a desktop with no windows, which
//     is the exact failure this module exists to fix. The body therefore uses
//     C-style loops and one plain message instead of several.
//   - `JSON.stringify` of an object built out of a *string literal* of quoted
//     keys does not survive either. Building the JSON from a plain object
//     literal works and keeps the payload in one message, so the receiving side
//     has a single, complete value to parse.
//
// Scripts persist in the compositor once loaded, and a later `start()` re-runs
// every loaded script, so each call unloads its own script in a `finally` and
// uses a fresh plugin name.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { createClient, type InterfaceDescriptor } from 'dbus-native'

import type { EnumeratedWindow } from './window-below'

const KWIN_SERVICE = 'org.kde.KWin'
const SCRIPTING_PATH = '/Scripting'
const SCRIPTING_IFACE = 'org.kde.kwin.Scripting'
const PROBE_PATH = '/probe'
const PROBE_IFACE = 'org.hermes.KWinWindowProbe'

/**
 * The one method KWin's script calls back on: one string in, nothing out.
 * Declared as its own typed constant because the classic descriptor is a
 * positional tuple — `[out, in, outNames, inNames]` — and a bare two-element
 * array is not one.
 */
const PROBE_DESCRIPTOR: InterfaceDescriptor = {
  methods: { Payload: ['', 's', [], []] },
  name: PROBE_IFACE,
  signals: {}
}

// KWin answers in milliseconds when it answers at all; a session where the
// scripting interface is not reachable must not stall the tool call, because
// the caller has a working enumerator to fall back to.
const REQUEST_TIMEOUT_MS = 2000

let requestSeq = 0

// One request at a time: the payload arrives on the connection this module
// owns, so two overlapping calls would each see the other's answer. The HUD's
// watcher and a tool call can overlap, hence the queue rather than a bare flag.
let inFlight: Promise<EnumeratedWindow[] | null> = Promise.resolve(null)

/**
 * Whether this session is one KWin could be running in: Wayland with Plasma as
 * the desktop. The check is a gate, not proof — `readKwinWindows` still has to
 * get an answer out of KWin, and answers null if it cannot.
 *
 * X11 is excluded on purpose. There the X11 enumerator is correct and cheaper,
 * and asking KWin as well would only add a round trip to a path that already
 * works.
 */
export function kwinSessionLikely(env: NodeJS.ProcessEnv): boolean {
  const session = env.XDG_SESSION_TYPE
  // `XDG_SESSION_TYPE` is authoritative when it is set. The `WAYLAND_DISPLAY`
  // fallback covers launchers that don't propagate it, but it must not override
  // an explicit `x11` — a Plasma X11 session with XWayland in the environment
  // would otherwise send us to KWin for an answer the X11 path already has.
  // `DISPLAY` is deliberately not consulted: a Plasma Wayland session normally
  // has it set, so treating it as evidence of X11 would exclude the very case
  // this module exists for.
  const onWayland = session === 'wayland' || (session !== 'x11' && Boolean(env.WAYLAND_DISPLAY))
  const plasma = (env.XDG_CURRENT_DESKTOP ?? '').toUpperCase().includes('KDE')
  const hyprland = Boolean(env.HYPRLAND_INSTANCE_SIGNATURE)

  return onWayland && plasma && !hyprland
}

/**
 * The script KWin runs. It reads the real stacking order and reports every
 * window on it; what counts as a window is decided on this side, in
 * `parseKwinWindows`, so that the interesting logic is unit-testable and the
 * script stays a dumb reader.
 *
 * `stackingOrder` is bottom-to-top, which is the reverse of the front-to-back
 * order the caller wants — the same correction `get-windows` needs on Linux,
 * and it is applied here rather than in the script so it can be tested.
 */
export const KWIN_PROBE_SCRIPT = `function __hermesSend(json) {
    callDBus("${PROBE_IFACE}", "${PROBE_PATH}", "${PROBE_IFACE}", "Payload", json);
}
try {
    var rows = [];
    var order = workspace.stackingOrder;
    for (var i = 0; i < order.length; i++) {
        var w = order[i];
        var f = w.frameGeometry;
        var desktops = [];
        var ds = w.desktops || [];
        for (var d = 0; d < ds.length; d++) { desktops.push(String(ds[d].id)); }
        rows.push({
            class: String(w.resourceClass), pid: w.pid, id: String(w.internalId),
            stackingOrder: w.stackingOrder, caption: String(w.caption), active: w.active,
            x: f.x, y: f.y, width: f.width, height: f.height,
            desktopWindow: w.desktopWindow, dock: w.dock, toolbar: w.toolbar, menu: w.menu,
            popupWindow: w.popupWindow, popupMenu: w.popupMenu, dropdownMenu: w.dropdownMenu,
            tooltip: w.tooltip, notification: w.notification,
            criticalNotification: w.criticalNotification, appletPopup: w.appletPopup,
            onScreenDisplay: w.onScreenDisplay, comboBox: w.comboBox, dndIcon: w.dndIcon,
            minimized: w.minimized, hidden: w.hidden, deleted: w.deleted,
            onAllDesktops: w.onAllDesktops, desktops: desktops
        });
    }
    var current = workspace.currentDesktop ? String(workspace.currentDesktop.id) : "";
    __hermesSend(JSON.stringify({ current: current, rows: rows }));
} catch (e) {
    __hermesSend(JSON.stringify({ error: String(e) }));
}
`

interface KwinWindow {
  active?: boolean
  appletPopup?: boolean
  caption?: string
  class?: string
  comboBox?: boolean
  criticalNotification?: boolean
  deleted?: boolean
  desktopWindow?: boolean
  desktops?: string[]
  dock?: boolean
  dndIcon?: boolean
  dropdownMenu?: boolean
  height?: number
  hidden?: boolean
  id?: string
  menu?: boolean
  minimized?: boolean
  notification?: boolean
  onAllDesktops?: boolean
  onScreenDisplay?: boolean
  pid?: number
  popupMenu?: boolean
  popupWindow?: boolean
  stackingOrder?: number
  toolbar?: boolean
  tooltip?: boolean
  width?: number
  x?: number
  y?: number
}

interface KwinPayload {
  current?: string
  error?: string
  rows?: KwinWindow[]
}

/**
 * KWin's chrome — the desktop background, panels, tooltips, popups, the
 * on-screen display — is on the same stacking order as real windows and, on a
 * Plasma desktop, usually forms the majority of it. Every flag below is a type
 * KWin itself exposes on a managed window, so this is a list of what the
 * compositor calls furniture rather than a guess at window names.
 */
const FURNITURE: Array<keyof KwinWindow> = [
  'appletPopup',
  'comboBox',
  'criticalNotification',
  'desktopWindow',
  'dndIcon',
  'dock',
  'dropdownMenu',
  'menu',
  'notification',
  'onScreenDisplay',
  'popupMenu',
  'popupWindow',
  'toolbar',
  'tooltip'
]

const isFurniture = (w: KwinWindow): boolean => FURNITURE.some(flag => w[flag] === true)

/**
 * `internalId` is a UUID string, and the field it feeds is a number that only
 * ever travels back to the model as an opaque handle, so a cheap truncation is
 * enough — same posture as the Hyprland provider's pointer parse.
 */
const opaqueId = (internalId: string): number =>
  Number.parseInt(internalId.replace(/[{}-]/g, '').slice(0, 8), 16) || 0

/**
 * A `stackingOrder` payload → the windows that could be underneath us,
 * front-to-back.
 *
 * Three corrections turn KWin's answer into the one `pickWindowBelow` wants:
 *
 *   - Order. `stackingOrder` runs bottom-to-top and the caller reads
 *     front-to-back, so the list is sorted by the index descending. Unlike
 *     Hyprland's focus history this is true stacking order, which is why our
 *     own windows stay in the list here: with real z-order the caller can walk
 *     past them and take the next one, which is what "below" means, and it
 *     still answers correctly when the HUD is not the frontmost window.
 *   - Desktop. Windows on a virtual desktop you cannot see occupy the same
 *     coordinates as the ones you can, so left in they win the overlap test
 *     against our bounds and the tool confidently reports a window on another
 *     desktop. Our own window names the visible desktop; if we cannot find
 *     ourselves we keep everything, which is no worse than the X11 path.
 *   - Furniture. See `FURNITURE` above.
 */
export function parseKwinWindows(payload: string, selfPid: number): EnumeratedWindow[] {
  let raw: KwinPayload

  try {
    raw = JSON.parse(payload) as KwinPayload
  } catch {
    return []
  }

  if (!raw || !Array.isArray(raw.rows)) {
    return []
  }

  const rows = raw.rows
  const ours = rows.find(r => r.pid === selfPid)
  const desktop = ours?.desktops?.[0] ?? raw.current ?? ''

  const visible =
    desktop === ''
      ? rows
      : rows.filter(r => r.onAllDesktops === true || (r.desktops ?? []).includes(desktop))

  return visible
    .filter(r => r.deleted !== true && r.hidden !== true && r.minimized !== true)
    .filter(r => !isFurniture(r) && (r.width ?? 0) > 0 && (r.height ?? 0) > 0)
    .sort((a, b) => (b.stackingOrder ?? -1) - (a.stackingOrder ?? -1))
    .map(r => ({
      app: r.class ?? '',
      bounds: {
        x: r.x ?? 0,
        y: r.y ?? 0,
        width: r.width ?? 0,
        height: r.height ?? 0
      },
      id: opaqueId(r.id ?? ''),
      pid: r.pid ?? 0,
      title: r.caption ?? ''
    }))
}

/** Where the probe script is written for KWin to read. */
export function kwinScriptPath(): string {
  return path.join(os.tmpdir(), `hermes-kwin-window-probe-${process.pid}.js`)
}

/**
 * Every window KWin can see, front-to-back, or null when this isn't a session
 * KWin answers in — which is the caller's cue to fall back to the X11 path.
 */
export function readKwinWindows(
  selfPid: number,
  env: NodeJS.ProcessEnv = process.env
): Promise<EnumeratedWindow[] | null> {
  if (!kwinSessionLikely(env)) {
    return Promise.resolve(null)
  }

  const run = inFlight.then(() => queryKwin(selfPid), () => queryKwin(selfPid))

  inFlight = run.catch(() => null)

  return run
}

async function queryKwin(selfPid: number): Promise<EnumeratedWindow[] | null> {
  const scriptPath = kwinScriptPath()
  const pluginName = `hermes-window-probe-${process.pid}-${++requestSeq}`

  const bus = createClient({
    busAddress:
      process.env.DBUS_SESSION_BUS_ADDRESS ||
      `unix:path=${process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid?.() ?? 0}`}/bus`,
    authMethods: ['EXTERNAL'],
    direct: true,
    timeout: REQUEST_TIMEOUT_MS
  })

  let settle: (value: null | string) => void = () => {}

  const answered = new Promise<null | string>(resolve => {
    settle = resolve
  })

  const call = (member: string, signature: string, body: unknown[]) =>
    bus.invokeDbus<unknown>(
      {
        body,
        destination  : KWIN_SERVICE,
        interface: SCRIPTING_IFACE,
        member,
        path: SCRIPTING_PATH,
        signature
      },
      { timeout: REQUEST_TIMEOUT_MS }
    )

  try {
    fs.writeFileSync(scriptPath, KWIN_PROBE_SCRIPT, 'utf8')

    await bus.invokeDbus({ member: 'Hello' })

    // Without the name KWin's script call has nothing to deliver to, and the
    // answer would arrive as an error this module cannot read. Losing the race
    // to an older instance of Hermes is a normal "not our turn" outcome.
    if ((await bus.requestName(PROBE_IFACE, 0)) !== 1) {
      return null
    }

    bus.exportInterface({ Payload: (json: string) => settle(json) }, PROBE_PATH, PROBE_DESCRIPTOR)

    await call('loadScript', 'ss', [scriptPath, pluginName])
    await call('start', '', [])

    const timer = new Promise<null>(resolve => setTimeout(() => resolve(null), REQUEST_TIMEOUT_MS))
    const payload = await Promise.race([answered, timer])

    if (!payload) {
      return null
    }

    const parsed = JSON.parse(payload) as KwinPayload

    if (parsed.error) {
      return null
    }

    const windows = parseKwinWindows(payload, selfPid)

    return windows.length > 0 ? windows : null
  } catch {
    return null
  } finally {
    // Scripts survive in the compositor, and the next `start()` runs every
    // loaded one — so leaving this behind would fire a stale probe on a later
    // call. Losing the connection releases the name with it.
    try {
      await call('unloadScript', 's', [pluginName])
    } catch {
      // The connection may already be gone; nothing left to unload then.
    }

    try {
      fs.unlinkSync(scriptPath)
    } catch {
      // Already removed, or never written.
    }

    bus.connection.stream.destroy()
  }
}
