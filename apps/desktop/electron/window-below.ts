// window-below.ts — which OS window sits directly underneath a Hermes window.
//
// Backs the desktop-gated `read_window_below` tool: the renderer receives
// `window.read.request` from the gateway, asks main over IPC, and answers
// with this module's serialized result. Enumeration uses `get-windows`
// (front-to-back z-order on macOS/Windows/Linux-X11); the picking logic is a
// pure function so the OS-specific part stays a thin provider.
//
// Privacy contract (matches the tool schema): metadata only — app, title,
// bounds. Never pixels. On macOS, window titles require the Screen Recording
// permission; we pass titles through only when that permission is ALREADY
// granted and never trigger the prompt for it.

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

const overlaps = (
  a: EnumeratedWindow['bounds'],
  b: EnumeratedWindow['bounds']
): boolean =>
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
  openWindows: (options?: {
    accessibilityPermission?: boolean
    screenRecordingPermission?: boolean
  }) => Promise<
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
 * platforms, where titles are free). Returns null only when enumeration
 * itself is unavailable (Wayland, missing xprop, addon load failure) — the
 * caller turns that into an empty tool answer.
 */
export async function readWindowBelow(
  selfPid: number,
  selfBounds: EnumeratedWindow['bounds'],
  titlesAvailable: boolean
): Promise<WindowBelowResult | null> {
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

  const windows: EnumeratedWindow[] = ordered.map(w => ({
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

  const { below, frontmost } = pickWindowBelow(windows, selfPid, selfBounds)

  const result: WindowBelowResult = {
    frontmost: frontmost ? { app: frontmost.app, title: frontmost.title } : null,
    platform: process.platform,
    window: below
      ? { app: below.app, bounds: below.bounds, id: below.id, title: below.title }
      : null
  }

  if (process.platform === 'darwin' && !titlesAvailable) {
    result.note =
      'Window titles are hidden: macOS reveals other apps\u2019 titles only with the ' +
      'Screen Recording permission, which Hermes does not request for this.'
  }

  return result
}
