/**
 * Hyprland (Omarchy, etc.) tiles a new Electron toplevel by default.
 * `alwaysOnTop` is compositor-owned on native Wayland, and `xdg_toplevel.move`
 * — the native HUD drag path — is ignored on a tiled window. The bar then
 * sits in the layout instead of over the user's work, which is the whole HUD.
 *
 * We already speak Hyprland's command socket for `read_window_below`. After
 * the HUD maps, ask that same socket to float and pin this title. Pin is the
 * always-on-top analogue (all workspaces, above tiled clients).
 *
 * Hyprland 0.55+ on a Lua config evaluates `dispatch` as Lua, so the classic
 * `setfloating` / `pin` strings fail there and the `hl.dsp.*` forms fail on a
 * hyprlang session. Which grammar a session speaks is only discoverable by
 * trying; we probe once and cache it.
 */

import { hyprlandRequest, hyprlandSocketPath } from './hyprland'

export type HyprlandDispatchReply = 'ok' | 'wrong-syntax' | 'failed'

export type HudHyprlandClient = {
  address: string
  floating: boolean
  pinned: boolean
}

type HyprlandDispatchSyntax = 'legacy' | 'lua'

export type HyprlandRequestFn = (socketPath: string, command: string) => Promise<null | string>

const DEFAULT_ATTEMPTS = 8
const DEFAULT_DELAY_MS = 50

let cachedSyntax: null | HyprlandDispatchSyntax = null

export function resetHyprlandDispatchSyntax(): void {
  cachedSyntax = null
}

/**
 * Hyprland's textual reply to a `dispatch`. "ok" applied. A Lua parse error or
 * "Invalid dispatcher" means we used the other grammar. Anything else is the
 * right grammar aimed at a missing window — retry the lookup, not the syntax.
 *
 * Reply strings are what Hyprland 0.54 (hyprlang) and 0.56 (Lua config) actually
 * write; they are the probe, not a guess.
 */
export function classifyHyprlandDispatchReply(reply: string): HyprlandDispatchReply {
  const text = reply.trim()

  if (text === 'ok') {
    return 'ok'
  }

  if (text.startsWith('Invalid dispatcher') || text.startsWith('error:')) {
    return 'wrong-syntax'
  }

  return 'failed'
}

function windowAddress(raw: string): string {
  return raw.startsWith('0x') || raw.startsWith('0X') ? raw : `0x${raw}`
}

/** First mapped client whose title is exactly the HUD's. */
export function parseHudHyprlandClient(payload: string, title: string): HudHyprlandClient | null {
  let raw: unknown

  try {
    raw = JSON.parse(payload)
  } catch {
    return null
  }

  if (!Array.isArray(raw)) {
    return null
  }

  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    const client = entry as { address?: unknown; floating?: unknown; pinned?: unknown; title?: unknown }

    if (client.title !== title || typeof client.address !== 'string' || client.address.length === 0) {
      continue
    }

    return {
      address: windowAddress(client.address),
      floating: client.floating === true,
      pinned: client.pinned === true
    }
  }

  return null
}

export function hudOverlayCommands(address: string, action: 'float' | 'pin'): { legacy: string; lua: string } {
  const selector = `address:${address}`

  if (action === 'float') {
    return {
      legacy: `dispatch setfloating ${selector}`,
      lua: `dispatch hl.dsp.window.float({ action = "enable", window = "${selector}" })`
    }
  }

  return {
    legacy: `dispatch pin ${selector}`,
    lua: `dispatch hl.dsp.window.pin({ action = "enable", window = "${selector}" })`
  }
}

async function dispatchWithFallback(
  request: HyprlandRequestFn,
  socketPath: string,
  pair: { legacy: string; lua: string }
): Promise<HyprlandDispatchReply> {
  const order: HyprlandDispatchSyntax[] = cachedSyntax === 'lua' ? ['lua', 'legacy'] : ['legacy', 'lua']

  for (const syntax of order) {
    const command = syntax === 'legacy' ? pair.legacy : pair.lua
    const reply = await request(socketPath, command)

    if (reply == null) {
      return 'failed'
    }

    const kind = classifyHyprlandDispatchReply(reply)

    if (kind === 'wrong-syntax') {
      continue
    }

    cachedSyntax = syntax

    return kind
  }

  return 'wrong-syntax'
}

async function defaultSleep(ms: number): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Float and pin the HUD on Hyprland. No-op on every other compositor.
 * Fire-and-forget from the HUD reveal path; retries because `j/clients` can
 * lag the first frame.
 */
export async function promoteHudOnHyprland(options: {
  title: string
  attempts?: number
  delayMs?: number
  env?: NodeJS.ProcessEnv
  request?: HyprlandRequestFn
  sleep?: (ms: number) => Promise<void>
  uid?: number
}): Promise<boolean> {
  const env = options.env ?? process.env
  const uid = options.uid ?? process.getuid?.() ?? 0
  const socketPath = hyprlandSocketPath(env, uid)

  if (!socketPath) {
    return false
  }

  const request = options.request ?? hyprlandRequest
  const sleep = options.sleep ?? defaultSleep
  const attempts = options.attempts ?? DEFAULT_ATTEMPTS
  const delayMs = options.delayMs ?? DEFAULT_DELAY_MS

  for (let attempt = 0; attempt < attempts; attempt++) {
    if (attempt > 0 && delayMs > 0) {
      await sleep(delayMs)
    }

    const payload = await request(socketPath, 'j/clients')

    if (!payload) {
      continue
    }

    const client = parseHudHyprlandClient(payload, options.title)

    if (!client) {
      continue
    }

    if (!client.floating) {
      const floated = await dispatchWithFallback(request, socketPath, hudOverlayCommands(client.address, 'float'))

      if (floated !== 'ok') {
        continue
      }
    }

    if (!client.pinned) {
      const pinned = await dispatchWithFallback(request, socketPath, hudOverlayCommands(client.address, 'pin'))

      if (pinned !== 'ok') {
        continue
      }
    }

    return true
  }

  return false
}
