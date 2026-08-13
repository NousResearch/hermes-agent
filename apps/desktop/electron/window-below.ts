// window-below.ts — which OS window sits directly underneath a Hermes window.
//
// Backs the desktop-gated `read_window_below` tool: the renderer receives
// `window.read.request` from the gateway, asks main over IPC, and answers
// with this module's serialized result. Enumeration uses `get-windows`
// (front-to-back z-order on macOS/Windows/Linux-X11); the picking logic is a
// pure function so the OS-specific part stays a thin provider. Where that
// provider can't run at all, the answer is why — see `enumerationFailureNote`.
//
// Privacy contract (matches the tool schema): metadata only — app, title,
// bounds. Never pixels. On macOS, window titles require the Screen Recording
// permission; we pass titles through only when that permission is ALREADY
// granted and never trigger the prompt for it.

import { readHyprlandWindows } from './hyprland'
import { readCosmicWindows } from './cosmic'

export interface EnumeratedWindow {
  app: string
  bounds: { x: number; y: number; width: number; height: number }
  id: number
  pid: number
  title: string
}

export interface WindowBelowResult {
  frontmost: { app: string; title: string } | null
  note?: string
  platform: string
  window: {
    app: string
    bounds: { x: number; y: number; width: number; height: number }
    id: number
    title: string
  } | null
}

export interface WindowBelowUnavailable {
  error: string
  platform: string
}

/**
 * Why enumeration just failed, in terms the user can act on.
 *
 * The generic "could not determine the window underneath" this replaces is a
 * dead end on Linux, where the two ways it fails have opposite fixes and
 * neither is guessable: a Wayland session withholds window identity from
 * applications outright, and an X11 session needs `xprop`/`xwininfo` present
 * because that is what the enumerator shells out to.
 *
 * A session with both `WAYLAND_DISPLAY` and `DISPLAY` is Wayland running
 * XWayland, where `xprop` can still answer — so it is treated as X11 and gets
 * the tooling advice rather than being told to change session type.
 */
export function enumerationFailureNote(platform: string, env: NodeJS.ProcessEnv): string {
  if (platform !== 'linux') {
    return 'Could not enumerate windows on this system.'
  }

  // Hyprland is asked over its own IPC, so reaching here means the socket
  // didn't answer — telling a Hyprland user to go and install xprop, or to
  // abandon Wayland, would send them in exactly the wrong direction.
  if (env.HYPRLAND_INSTANCE_SIGNATURE) {
    return (
      'Could not enumerate windows: Hyprland did not answer on its IPC socket. ' +
      'Check that `hyprctl clients` works from the same session Hermes is ' +
      'running in.'
    )
  }

  // COSMIC (cosmic-comp) exposes no native window-enumeration IPC, so Hermes
  // can only read other windows when it runs under XWayland. Reaching here on
  // a native-Wayland COSMIC session means that hasn't been enabled yet — point
  // at the opt-in rather than the generic "log into X11" advice, which COSMIC
  // barely supports. (See issue #84011 / PR #84013.)
  const isCosmic =
    (env.XDG_CURRENT_DESKTOP ?? '').toLowerCase().includes('cosmic') ||
    (env.XDG_SESSION_DESKTOP ?? '').toLowerCase().includes('cosmic')

  if (isCosmic) {
    return (
      'Could not enumerate windows: COSMIC does not expose other apps\u2019 windows ' +
      'to Hermes over native Wayland. Enable XWayland so the window-under awareness ' +
      'works — in Hermes config.yaml set `desktop: { ozone_platform_hint: x11 }` ' +
      '(or launch with ELECTRON_OZONE_PLATFORM_HINT=x11 hermes desktop). See issue #84011.'
    )
  }

  const wayland = env.XDG_SESSION_TYPE === 'wayland' || (Boolean(env.WAYLAND_DISPLAY) && !env.DISPLAY)

  if (wayland) {
    return (
      'Could not enumerate windows: this is a Wayland session, and Wayland does ' +
      'not let an application see other applications\u2019 windows. Log in to an ' +
      'X11/Xorg session, or run Hermes under XWayland with DISPLAY set.'
    )
  }

  return (
    'Could not enumerate windows: this needs the xprop and xwininfo commands ' +
    '(the x11-utils package on Debian/Ubuntu, xorg-x11-utils on Fedora).'
  )
}

const overlaps = (a: EnumeratedWindow['bounds'], b: EnumeratedWindow['bounds']): boolean =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height

/**
 * Pick the window directly underneath ours from a front-to-back window list.
 *
 * Walks past every window owned by our own process (all Hermes windows share
 * the main process pid), then takes the first other-process window whose
 * bounds overlap ours — "underneath" means visually behind, not merely next
 * in z-order on some other display. `frontmost` is the first other-process
 * window regardless of overlap: the app the user was last working in.
 */
export function pickWindowBelow(
  windows: EnumeratedWindow[],
  selfPid: number,
  selfBounds: EnumeratedWindow['bounds']
): { below: EnumeratedWindow | null; frontmost: EnumeratedWindow | null } {
  const others = windows.filter(w => w.pid !== selfPid)
  const frontmost = others[0] ?? null

  const selfIndex = windows.findIndex(w => w.pid === selfPid)
  const behind = selfIndex === -1 ? others : windows.slice(selfIndex + 1)
  const below = behind.find(w => w.pid !== selfPid && overlaps(w.bounds, selfBounds)) ?? null

  return { below, frontmost }
}

type GetWindowsModule = {
  openWindows: (options?: { accessibilityPermission?: boolean; screenRecordingPermission?: boolean }) => Promise<
    Array<{
      bounds?: { height?: number; width?: number; x?: number; y?: number }
      id?: number
      owner?: { name?: string; processId?: number }
      title?: string
    }>
  >
}

let getWindowsModule: Promise<GetWindowsModule> | null = null

const loadGetWindows = (): Promise<GetWindowsModule> => {
  getWindowsModule ??= import('get-windows')

  return getWindowsModule
}

/**
 * Enumerate windows and serialize the one underneath `selfBounds`.
 *
 * `titlesAvailable` is the macOS Screen Recording grant (pass true on other
 * platforms, where titles are free). When enumeration itself is unavailable
 * (Wayland, missing xprop, addon load failure) this answers with the reason
 * rather than nothing, so the agent can tell the user what to fix instead of
 * reporting a blank failure.
 */
async function enumerateViaGetWindows(titlesAvailable: boolean): Promise<EnumeratedWindow[] | null> {
  let raw

  try {
    const { openWindows } = await loadGetWindows()
    raw = await openWindows(
      process.platform === 'darwin'
        ? { accessibilityPermission: false, screenRecordingPermission: titlesAvailable }
        : undefined
    )
  } catch {
    return null
  }

  if (!Array.isArray(raw)) {
    return null
  }

  // get-windows documents openWindows() as front-to-back, and macOS/Windows
  // honor that (CGWindowList / EnumWindows order). Its lib/linux.js, however,
  // iterates `_NET_CLIENT_LIST_STACKING` in raw xprop order, which EWMH
  // defines as bottom-to-top — so the Linux list arrives back-to-front and
  // must be reversed to match. (Verified against get-windows 9.3.0.)
  const ordered = process.platform === 'linux' ? [...raw].reverse() : raw

  return ordered.map(w => ({
    app: w.owner?.name ?? '',
    bounds: {
      x: w.bounds?.x ?? 0,
      y: w.bounds?.y ?? 0,
      width: w.bounds?.width ?? 0,
      height: w.bounds?.height ?? 0
    },
    id: w.id ?? 0,
    pid: w.owner?.processId ?? 0,
    title: w.title ?? ''
  }))
}

export async function readWindowBelow(
  selfPid: number,
  selfBounds: EnumeratedWindow['bounds'],
  titlesAvailable: boolean
): Promise<WindowBelowResult | WindowBelowUnavailable> {
  // Hyprland first, and only ever on Hyprland — its own IPC sees native Wayland
  // windows, which the X11 enumerator below cannot, and it answers null
  // everywhere else so the established path stays the default.
  //
  // COSMIC next (only on COSMIC): it has no native window IPC, so its only
  // viable enumeration is the X11 one, which works once Hermes runs under
  // XWayland (desktop.ozone_platform_hint: x11, issue #84011). On a
  // native-Wayland COSMIC session readCosmicWindows returns null and the
  // tailored COSMIC note in enumerationFailureNote takes over.
  const windows =
    (await readHyprlandWindows(selfPid)) ??
    (await readCosmicWindows(selfPid, titlesAvailable, process.env, enumerateViaGetWindows)) ??
    (await enumerateViaGetWindows(titlesAvailable))

  if (!windows) {
    return {
      error: enumerationFailureNote(process.platform, process.env),
      platform: process.platform
    }
  }

  const { below, frontmost } = pickWindowBelow(windows, selfPid, selfBounds)

  const result: WindowBelowResult = {
    frontmost: frontmost ? { app: frontmost.app, title: frontmost.title } : null,
    platform: process.platform,
    window: below ? { app: below.app, bounds: below.bounds, id: below.id, title: below.title } : null
  }

  if (process.platform === 'darwin' && !titlesAvailable) {
    result.note =
      'Window titles are hidden: macOS reveals other apps\u2019 titles only with the ' +
      'Screen Recording permission, which Hermes does not request for this.'
  }

  return result
}
