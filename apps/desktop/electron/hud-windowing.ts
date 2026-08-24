/**
 * HUD windowing profile — one object for every operation that used to
 * special-case OS / Ozone / compositor.
 *
 * The Ozone backend (not the login session) is the normalizer: Electron 20+
 * prefers a native Wayland surface on a Wayland session, so `setBounds`
 * position is a no-op there and ignore-mouse restore works. X11 is the
 * inverse. macOS and Windows keep the click-through + renderer-drag design.
 *
 * Capabilities are derived once. Callers ask the profile, they do not
 * re-derive "linux && click-through".
 */

export type HudOzoneBackend = 'wayland' | 'x11'
export type HudWindowingBackend = 'cocoa' | 'win32' | HudOzoneBackend
export type HudInput = 'click-through' | 'solid'
export type HudMove = 'native-drag' | 'renderer'

export interface HudWindowing {
  backend: HudWindowingBackend
  /** Whether the window may ignore the mouse (restore is known to work). */
  ignoreMouse: boolean
  input: HudInput
  move: HudMove
  /** Client can set global x/y. Native Wayland cannot. */
  clientPlacement: boolean
  /** Immediate Ctrl+primary-button grab (X11 renderer drag). */
  controlDrag: boolean
  /** Temporary all-workspaces during a renderer grab (X11/KWin). */
  workspaceTransfer: boolean
  /** Main must poll the cursor; `{ forward: true }` is unavailable. */
  cursorFeed: boolean
}

/** Slice the renderer is allowed to see. */
export interface HudWindowingView {
  clientPlacement: boolean
  controlDrag: boolean
  nativeDrag: boolean
  workspaceTransfer: boolean
}

function requestedOzonePlatform(env: NodeJS.ProcessEnv, argv: readonly string[]): null | string {
  let explicit: null | string = null
  let hint: null | string = null

  for (const arg of argv) {
    const match = /^--ozone-platform(-hint)?=(.+)$/.exec(arg)

    if (!match) {
      continue
    }

    if (match[1]) {
      hint = match[2].toLowerCase()
    } else {
      explicit = match[2].toLowerCase()
    }
  }

  return explicit ?? hint ?? env.ELECTRON_OZONE_PLATFORM_HINT?.toLowerCase() ?? null
}

function sessionIsWayland(env: NodeJS.ProcessEnv): boolean {
  return env.XDG_SESSION_TYPE === 'wayland' || (Boolean(env.WAYLAND_DISPLAY) && !env.DISPLAY)
}

/**
 * The Ozone platform Electron will actually use on Linux.
 *
 * Unset / `auto` follows the session. An explicit `--ozone-platform=x11`
 * (or `desktop.ozone_platform_hint: x11`) is the COSMIC always-on-top escape
 * hatch and lands on the solid / renderer-drag path.
 */
export function linuxOzoneBackend(env: NodeJS.ProcessEnv, argv: readonly string[]): HudOzoneBackend {
  const requested = requestedOzonePlatform(env, argv)

  if (requested === 'x11') {
    return 'x11'
  }

  if (requested === 'wayland') {
    return 'wayland'
  }

  return sessionIsWayland(env) ? 'wayland' : 'x11'
}

function desktopWindowing(backend: 'cocoa' | 'win32'): HudWindowing {
  return {
    backend,
    ignoreMouse: true,
    input: 'click-through',
    move: 'renderer',
    clientPlacement: true,
    controlDrag: false,
    workspaceTransfer: false,
    cursorFeed: false
  }
}

export function resolveHudWindowing(
  platform: string,
  env: NodeJS.ProcessEnv,
  argv: readonly string[]
): HudWindowing {
  if (platform === 'darwin') {
    return desktopWindowing('cocoa')
  }

  if (platform === 'win32') {
    return desktopWindowing('win32')
  }

  const backend = linuxOzoneBackend(env, argv)
  const wayland = backend === 'wayland'

  return {
    backend,
    ignoreMouse: wayland,
    input: wayland ? 'click-through' : 'solid',
    move: wayland ? 'native-drag' : 'renderer',
    clientPlacement: !wayland,
    controlDrag: !wayland,
    workspaceTransfer: !wayland,
    cursorFeed: wayland
  }
}

export function hudWindowingView(windowing: HudWindowing): HudWindowingView {
  return {
    clientPlacement: windowing.clientPlacement,
    controlDrag: windowing.controlDrag,
    nativeDrag: windowing.move === 'native-drag',
    workspaceTransfer: windowing.workspaceTransfer
  }
}
