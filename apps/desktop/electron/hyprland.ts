// hyprland.ts — window enumeration for Hyprland, via the compositor's own IPC.
//
// `read_window_below` normally enumerates through `get-windows`, which on Linux
// shells out to `xprop` and reads `_NET_CLIENT_LIST_STACKING`. That is an X11
// protocol, and Wayland deliberately refuses to tell one application about
// another's windows — so on a Wayland session the tool has nothing to work
// with. Under XWayland it is arguably worse than nothing: it enumerates the few
// legacy X11 clients and silently misses every native Wayland window, which on
// a Hyprland desktop is most of them.
//
// Hyprland answers the question directly. `j/clients` returns every window with
// its class, title, position, size, pid and focus history, which is richer than
// anything the X11 path provides. This module is the provider for that; the
// picking logic stays in window-below.ts, unchanged and shared.
//
// Sway and the other wlroots compositors expose the same information through
// `swaymsg -t get_tree`. That is a different shape (a tree, not a list) and is
// deliberately not attempted here rather than shipped untested.

import net from 'node:net'

import type { EnumeratedWindow } from './window-below'

// Hyprland evaluates this socket synchronously and freezes for a five-second
// timeout on an unclosed connection, so every path below destroys the socket.
// One request per tool call, never polled.
const REQUEST_TIMEOUT_MS = 1000

/**
 * Path to the running instance's command socket, or null when Hyprland is not
 * the compositor.
 *
 * The instance signature is what makes this per-instance — two Hyprlands on one
 * machine have two sockets. The `/run/user/<uid>` fallback mirrors Hyprland's
 * own client code for sessions that don't export `XDG_RUNTIME_DIR`.
 */
export function hyprlandSocketPath(env: NodeJS.ProcessEnv, uid: number): null | string {
  const signature = env.HYPRLAND_INSTANCE_SIGNATURE

  if (!signature) {
    return null
  }

  const runtime = env.XDG_RUNTIME_DIR ? `${env.XDG_RUNTIME_DIR}/hypr` : `/run/user/${uid}/hypr`

  return `${runtime}/${signature}/.socket.sock`
}

interface HyprlandClient {
  address?: string
  at?: [number, number]
  class?: string
  pid?: number
  size?: [number, number]
  title?: string
  workspace?: { id?: number }
}

/**
 * `j/clients` payload → the windows that could be underneath us, most recently
 * used first.
 *
 * Three corrections turn Hyprland's answer into the one `pickWindowBelow`
 * wants:
 *
 *   - Order. The list arrives in the compositor's internal order, not z-order.
 *     `focusHistoryID` is 0 for the focused window, 1 for the one before it, and
 *     so on, which is the ordering the caller actually means by "underneath":
 *     the app the user was last working in.
 *   - Ourselves. Dropped, and this is load-bearing rather than tidiness. On X11
 *     the list is true stacking order, so walking past our own window and
 *     taking the next one is what "below" means. Focus history is not stacking
 *     order: the HUD floats always-on-top while the user works in the app
 *     beneath it, so the app we want to report is the focused one and WE are
 *     further down the list. Slicing after ourselves would skip straight past
 *     the answer and name something the user last touched ten minutes ago.
 *     Leaving ourselves out puts `pickWindowBelow` on its overlap-only path,
 *     which reads a focus-ordered list correctly.
 *   - Workspace. `clients` spans every workspace, and windows on a workspace
 *     you cannot see occupy the same coordinates as the ones you can. Left in,
 *     they win the overlap test against our bounds and the tool confidently
 *     reports a window on another desktop. Our own window pins which workspace
 *     is the visible one — the one use we have for it before dropping it; if we
 *     can't find ourselves we keep everything, which is no worse than the X11
 *     path.
 */
export function parseHyprlandClients(payload: string, selfPid: number): EnumeratedWindow[] {
  let raw: unknown

  try {
    raw = JSON.parse(payload)
  } catch {
    return []
  }

  if (!Array.isArray(raw)) {
    return []
  }

  const clients = raw as Array<HyprlandClient & { focusHistoryID?: number }>
  const ours = clients.find(c => c.pid === selfPid)
  const workspace = ours?.workspace?.id
  const visible = workspace === undefined ? clients : clients.filter(c => c.workspace?.id === workspace)

  return visible
    .filter(c => c.pid !== selfPid && (c.size?.[0] ?? 0) > 0 && (c.size?.[1] ?? 0) > 0)
    .sort((a, b) => (a.focusHistoryID ?? Number.MAX_SAFE_INTEGER) - (b.focusHistoryID ?? Number.MAX_SAFE_INTEGER))
    .map(c => ({
      app: c.class ?? '',
      bounds: {
        x: c.at?.[0] ?? 0,
        y: c.at?.[1] ?? 0,
        width: c.size?.[0] ?? 0,
        height: c.size?.[1] ?? 0
      },
      // Addresses are pointers rendered as hex; they only travel back to the
      // model as an opaque handle, so a failed parse costs nothing.
      id: Number.parseInt(c.address ?? '', 16) || 0,
      pid: c.pid ?? 0,
      title: c.title ?? ''
    }))
}

/** One request on the command socket, always closed, never left hanging. */
function request(socketPath: string, command: string): Promise<null | string> {
  return new Promise(resolve => {
    let body = ''
    let settled = false

    const socket = net.createConnection(socketPath)

    const finish = (result: null | string) => {
      if (settled) {
        return
      }

      settled = true
      socket.destroy()
      resolve(result)
    }

    socket.setTimeout(REQUEST_TIMEOUT_MS, () => finish(null))
    socket.on('error', () => finish(null))
    socket.on('connect', () => socket.write(command))
    socket.on('data', chunk => {
      body += chunk.toString('utf8')
    })
    socket.on('end', () => finish(body))
  })
}

/**
 * Every window Hyprland can see, front-to-back, or null when this isn't a
 * Hyprland session or the compositor didn't answer — which is the caller's cue
 * to fall back to the X11 enumerator.
 */
export async function readHyprlandWindows(
  selfPid: number,
  env: NodeJS.ProcessEnv = process.env,
  uid: number = process.getuid?.() ?? 0
): Promise<EnumeratedWindow[] | null> {
  const socketPath = hyprlandSocketPath(env, uid)

  if (!socketPath) {
    return null
  }

  const payload = await request(socketPath, 'j/clients')

  if (!payload) {
    return null
  }

  const windows = parseHyprlandClients(payload, selfPid)

  return windows.length > 0 ? windows : null
}
