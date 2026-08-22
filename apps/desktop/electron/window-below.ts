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

let getWindowsModule: Promise<GetWindowsModule | null> | null = null

/**
 * Why enumeration last came up empty, in the enumerator's own words.
 *
 * Every way this can fail used to be swallowed by a bare `catch` and reported
 * as the same generic "Could not enumerate windows on this system." — which is
 * indistinguishable from a platform that genuinely cannot answer, and sends
 * anyone debugging it looking at permissions when the real fault was an exec
 * that never got off the ground. The reason is appended to the error the tool
 * returns, so it reaches the agent transcript (and `agent.log` on whichever
 * host runs the backend) without needing the desktop's own log.
 *
 * Holds the most recent cause until an enumeration succeeds. Import failures
 * stay permanent by construction: the module import is memoized, so a failed
 * load can never be followed by a success, and its reason survives to be
 * reported on every later call. A transient runtime failure is cleared as soon
 * as `openWindows()` returns a real list, so a later unexplained null carries
 * no cause at all and a stale reason can never be attached to it.
 */
let lastEnumerationFailure: string | null = null

export const describeThrown = (error: unknown): string => {
  const message = error instanceof Error ? error.message : String(error)
  const code = (error as { code?: string } | null)?.code

  return code ? `${code}: ${message}` : message
}

const noteEnumerationFailure = (reason: string): void => {
  lastEnumerationFailure = reason
  console.warn(`[window-below] ${reason}`)
}

/**
 * Redirect a resolved module specifier out of `app.asar` so its files exist on
 * the real filesystem. `/app.asar/` only ever appears as a complete path
 * segment in a packaged build (electron-builder names the archive exactly
 * that), so in dev — where nothing is archived — this is a no-op.
 */
export const resolveOutsideAsar = (specifier: string): string =>
  specifier.replace('/app.asar/', '/app.asar.unpacked/')

const loadGetWindows = (): Promise<GetWindowsModule | null> => {
  // get-windows is an optionalDependency: `npm ci` can skip it when its native
  // install fails, including Linux and Windows ARM64 where 9.3.0 has no
  // prebuilt. A missing module is therefore a normal state on those targets,
  // so the lazy import resolves to null instead of rejecting; enumeration then
  // degrades to the failure note instead of an uncaught error.
  //
  // The import is redirected out of app.asar in a packaged build. get-windows
  // does its real work by exec'ing a helper binary whose path it derives from
  // its own module URL, and execFile cannot run a file that lives inside the
  // archive — the kernel sees app.asar as a file, so the spawn fails ENOTDIR.
  // Electron does rewrite asar paths for child_process, but only once that
  // module has been pulled through the CJS loader, which this ESM main process
  // never does (every import here is `from 'node:child_process'`). Importing
  // the unpacked copy directly gives get-windows a real path to derive from
  // and does not depend on that patch ever being installed. Outside a packaged
  // build the specifier has no /app.asar/ segment and the replace is a no-op.
  getWindowsModule ??= (async () => {
    let specifier

    try {
      specifier = resolveOutsideAsar(import.meta.resolve('get-windows'))
    } catch (error) {
      noteEnumerationFailure(`import.meta.resolve('get-windows') threw ${describeThrown(error)}`)

      return null
    }

    try {
      return await import(specifier)
    } catch (error) {
      noteEnumerationFailure(`import('${specifier}') threw ${describeThrown(error)}`)

      return null
    }
  })()

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
    const getWindows = await loadGetWindows()

    if (!getWindows) {
      return null
    }

    const { openWindows } = getWindows
    raw = await openWindows(
      process.platform === 'darwin'
        ? { accessibilityPermission: false, screenRecordingPermission: titlesAvailable }
        : undefined
    )
  } catch (error) {
    noteEnumerationFailure(`openWindows() threw ${describeThrown(error)}`)

    return null
  }

  if (!Array.isArray(raw)) {
    noteEnumerationFailure(`openWindows() returned ${typeof raw}, expected an array`)

    return null
  }

  // A real list proves both the loader and the enumerator work, so any cause
  // recorded earlier in this process describes a condition that has passed.
  // Import failures keep their permanence here by construction: a memoized
  // failed import returns null above and never reaches this line.
  lastEnumerationFailure = null

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
  const windows = (await readHyprlandWindows(selfPid)) ?? (await enumerateViaGetWindows(titlesAvailable))

  if (!windows) {
    return {
      error: lastEnumerationFailure
        ? `${enumerationFailureNote(process.platform, process.env)} (${lastEnumerationFailure})`
        : enumerationFailureNote(process.platform, process.env),
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
