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

import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { app } from 'electron'

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
 *
 * macOS and Windows have no environmental fork like that, so their note is
 * whatever the enumerator actually said. That `detail` is the whole point: a
 * bare "could not enumerate windows on this system" is what a real report came
 * back with (macOS 26, packaged app), and neither the tool result nor the
 * desktop log said whether the module failed to load, the helper failed to
 * spawn, or the OS answered with nothing — three failures with three different
 * fixes, all collapsed into one sentence.
 */
export function enumerationFailureNote(platform: string, env: NodeJS.ProcessEnv, detail?: string): string {
  if (platform !== 'linux') {
    return detail ? `Could not enumerate windows: ${detail}` : 'Could not enumerate windows on this system.'
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

/** Enumeration couldn't run at all, and why. Distinct from an empty list,
 *  which is a real answer meaning "nothing else is on screen". */
export interface EnumerationFailure {
  reason: string
}

export const enumerationFailed = <T>(result: EnumerationFailure | T): result is EnumerationFailure =>
  typeof result === 'object' && result !== null && 'reason' in result

const describeError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error ?? 'unknown error')

/**
 * Redirect a resolved module specifier out of `app.asar` so its files exist on
 * the real filesystem. `/app.asar/` only ever appears as a complete path
 * segment in a packaged build (electron-builder names the archive exactly
 * that), so in dev — where nothing is archived — this is a no-op.
 */
export const resolveOutsideAsar = (specifier: string): string =>
  specifier.replace('/app.asar/', '/app.asar.unpacked/')

let getWindowsModule: Promise<GetWindowsModule | EnumerationFailure> | null = null

const loadGetWindows = (): Promise<GetWindowsModule | EnumerationFailure> => {
  // get-windows is an optionalDependency: `npm ci` can skip it when its native
  // install fails, including Linux and Windows ARM64 where 9.3.0 has no
  // prebuilt. A missing module is therefore a normal state on those targets,
  // so the lazy import resolves to null instead of rejecting; enumeration then
  // degrades to the failure note instead of an uncaught error.
  //
  // The STAGED copy is tried first, and it is not a dev-only nicety.
  // `import('get-windows')` resolves out of node_modules, whose lib/windows.js
  // locates its binding through `preGyp.find()` — by HOST platform. When the
  // tree was installed on a different OS than Electron is running on (a
  // WSL-hosted dev run driving a win32 Electron is the everyday case here),
  // pre-gyp picks the host's slot, ignores the win32 binding sitting beside it,
  // and upstream's fail-soft path hands back no-op stubs. Enumeration then
  // reports "unavailable" on a machine that answers perfectly well, which is
  // what silently disabled both read_window_below and the HUD's game overlay.
  // scripts/stage-native-deps.mjs writes a staged lib/windows.js that requires
  // the binding directly, so it is the more reliable of the two everywhere.
  getWindowsModule ??= (async () => {
    // Both specifiers are redirected out of app.asar in a packaged build.
    // get-windows does its real work by exec'ing a helper binary whose path it
    // derives from its own module URL, so a module imported through an
    // `/app.asar/` path derives an in-archive helper path and the spawn fails
    // ENOTDIR: the kernel sees app.asar as a file. Electron rewrites asar
    // paths for child_process only once that module has been pulled through
    // the CJS loader, which this ESM main process never does (every import
    // here is `from 'node:child_process'`). Pointing the import at the
    // unpacked copy gives get-windows a real path to derive from.
    const staged = resolveOutsideAsar(
      path.join(app.getAppPath(), 'dist', 'node_modules', 'get-windows', 'index.js')
    )
    let stagedError = 'not staged in this build'

    if (fs.existsSync(staged)) {
      try {
        return (await import(pathToFileURL(staged).href)) as GetWindowsModule
      } catch (error) {
        stagedError = describeError(error)
      }
    }

    try {
      return (await import(resolveOutsideAsar(import.meta.resolve('get-windows')))) as GetWindowsModule
    } catch (error) {
      return {
        reason:
          'the get-windows module could not be loaded ' +
          `(staged copy: ${stagedError}; node_modules copy: ${describeError(error)})`
      }
    }
  })()

  return getWindowsModule
}

/**
 * Every window `get-windows` can see, front-to-back, or why it could not look.
 *
 * `titlesAvailable` is the macOS Screen Recording grant (pass true on other
 * platforms, where titles are free). The three ways this can fail — the module
 * not loading, the enumerator throwing, the enumerator answering with
 * something that isn't a list — each say so, because they have three different
 * fixes and the caller has no other way to tell them apart.
 */
async function enumerateViaGetWindows(titlesAvailable: boolean): Promise<EnumeratedWindow[] | EnumerationFailure> {
  const getWindows = await loadGetWindows()

  if (enumerationFailed(getWindows)) {
    return getWindows
  }

  let raw

  try {
    raw = await getWindows.openWindows(
      process.platform === 'darwin'
        ? { accessibilityPermission: false, screenRecordingPermission: titlesAvailable }
        : undefined
    )
  } catch (error) {
    // On macOS this is the helper binary failing to spawn — a missing or
    // non-executable `main`, or the OS refusing to run it — which is invisible
    // from the outside and used to surface as the generic note.
    return { reason: `the window enumerator failed: ${describeError(error)}` }
  }

  if (!Array.isArray(raw)) {
    return { reason: 'the window enumerator returned no window list' }
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

/**
 * Front-to-back window enumeration, or why the platform could not answer.
 *
 * Hyprland first, and only ever on Hyprland — its own IPC sees native Wayland
 * windows, which the X11 enumerator cannot, and it answers null everywhere
 * else so the established path stays the default. Shared by the
 * read_window_below tool and the HUD's game-overlay watch, so the two can
 * never disagree about what the screen looks like.
 */
export async function enumerateWindowsFrontToBack(
  selfPid: number,
  titlesAvailable: boolean
): Promise<EnumeratedWindow[] | EnumerationFailure> {
  return (await readHyprlandWindows(selfPid)) ?? (await enumerateViaGetWindows(titlesAvailable))
}

export async function readWindowBelow(
  selfPid: number,
  selfBounds: EnumeratedWindow['bounds'],
  titlesAvailable: boolean
): Promise<WindowBelowResult | WindowBelowUnavailable> {
  const windows = await enumerateWindowsFrontToBack(selfPid, titlesAvailable)

  if (enumerationFailed(windows)) {
    return {
      error: enumerationFailureNote(process.platform, process.env, windows.reason),
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
