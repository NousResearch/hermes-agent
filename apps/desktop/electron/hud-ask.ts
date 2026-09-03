/**
 * HUD "ask about this" — the pointer-targeted action sheet.
 *
 * A global chord (or, when the optional input hook is installed and the user
 * opted in, Ctrl + right-click anywhere) captures what is under the OS cursor
 * and hands it to the HUD, which asks what to do with it. Electron cannot see
 * clicks in other apps on its own, so the chord is the always-available door
 * and the right-click is the upgrade.
 *
 * Pure pieces live here — the preference record, the crop math, the
 * window-under-cursor pick, and the chord controller — so the main-process
 * glue in main.ts stays thin and each decision is tested on its own.
 */

import { type GlobalShortcutLike, parseQuickEntryShortcut } from './quick-entry'
import type { EnumeratedWindow } from './window-below'

export const DEFAULT_HUD_ASK_SHORTCUT = 'CommandOrControl+Alt+H'

/** Longest edge of the crop handed to the model, in image pixels. */
export const HUD_ASK_CROP_SIZE = { width: 960, height: 600 }

export interface HudPrefs {
  /** Lazy follow-the-pointer mode (see hud-follow.ts). */
  follow: boolean
  /** Global chord that opens the action sheet at the cursor. */
  askShortcut: string
  /** Ctrl + right-click anywhere opens the sheet — needs the optional input
   *  hook (uiohook-napi); a missing hook keeps this a documented no-op. */
  askOnRightClick: boolean
  /** Pixel pets patrolling the strip above the bar. */
  pets: boolean
  /** Pet per agent, keyed by lower-cased profile name. */
  petByAgent: Record<string, HudPetChoice>
}

export type HudPetChoice = 'avatar' | 'hank' | 'mina' | 'none'

const HUD_PET_CHOICES = new Set<string>(['avatar', 'hank', 'mina', 'none'])

export const DEFAULT_HUD_PREFS: HudPrefs = {
  // On by default: following the pointer is what makes the HUD a companion
  // rather than a window you go and find. The Settings row turns it off.
  follow: true,
  askShortcut: DEFAULT_HUD_ASK_SHORTCUT,
  askOnRightClick: false,
  pets: true,
  petByAgent: {}
}

function sanitizePetByAgent(raw: unknown): Record<string, HudPetChoice> {
  const out: Record<string, HudPetChoice> = {}

  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return out
  }

  for (const [profile, choice] of Object.entries(raw as Record<string, unknown>)) {
    const key = profile.trim().toLowerCase()

    if (key && typeof choice === 'string' && HUD_PET_CHOICES.has(choice)) {
      out[key] = choice as HudPetChoice
    }
  }

  return out
}

/** Raw persisted JSON → usable prefs. Same posture as Quick Entry's settings:
 *  anything malformed falls back to the shipped default for that field only. */
export function sanitizeHudPrefs(raw: unknown): HudPrefs {
  const record = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const parsed = parseQuickEntryShortcut(record.askShortcut)

  return {
    follow: typeof record.follow === 'boolean' ? record.follow : DEFAULT_HUD_PREFS.follow,
    askShortcut: parsed.ok ? parsed.accelerator : DEFAULT_HUD_ASK_SHORTCUT,
    askOnRightClick: record.askOnRightClick === true,
    pets: typeof record.pets === 'boolean' ? record.pets : DEFAULT_HUD_PREFS.pets,
    petByAgent: sanitizePetByAgent(record.petByAgent)
  }
}

interface Point {
  x: number
  y: number
}

interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/**
 * The crop rectangle, in IMAGE pixels, centred on the cursor.
 *
 * The capture is a thumbnail of the whole display whose pixel size need not
 * equal the display's DIP size times its scale factor (desktopCapturer sizes
 * thumbnails to fit the requested box), so the cursor is mapped through the
 * display bounds proportionally rather than multiplied by the scale factor.
 * The rect is then shifted, not shrunk, to stay inside the image — a cursor in
 * a corner still gets the full crop size, just off-centre.
 */
export function cropAroundCursor(
  cursor: Point,
  display: Rect,
  image: { width: number; height: number },
  size: { width: number; height: number }
): Rect {
  const width = Math.min(size.width, image.width)
  const height = Math.min(size.height, image.height)

  const fx = display.width > 0 ? (cursor.x - display.x) / display.width : 0.5
  const fy = display.height > 0 ? (cursor.y - display.y) / display.height : 0.5

  const cx = Math.round(Math.min(Math.max(fx, 0), 1) * image.width)
  const cy = Math.round(Math.min(Math.max(fy, 0), 1) * image.height)

  const x = Math.min(Math.max(cx - Math.round(width / 2), 0), image.width - width)
  const y = Math.min(Math.max(cy - Math.round(height / 2), 0), image.height - height)

  return { x, y, width, height }
}

/**
 * The topmost window, excluding Hermes' own, whose bounds contain the cursor.
 * The list is front-to-back (see enumerateWindowsFrontToBack), so the first
 * hit is the one the user is looking at.
 */
export function windowUnderCursor(
  windows: readonly EnumeratedWindow[],
  cursor: Point,
  selfPid: number
): EnumeratedWindow | null {
  for (const candidate of windows) {
    if (candidate.pid === selfPid) {
      continue
    }

    const { x, y, width, height } = candidate.bounds

    if (width <= 0 || height <= 0) {
      continue
    }

    if (cursor.x >= x && cursor.x < x + width && cursor.y >= y && cursor.y < y + height) {
      return candidate
    }
  }

  return null
}

/** What the HUD renderer receives. `imagePath` is a PNG on THIS machine in
 *  the composer-images folder, exactly what a pasted screenshot produces, so
 *  the ordinary image-attachment path carries it to the model. */
export interface HudAskPayload {
  app: string
  /** Where the sheet was invoked, screen DIP. */
  cursor: Point
  /** Small data URL for the sheet's preview — never the full capture. */
  imagePath: string
  thumbnail: string
  title: string
  /** 'shortcut' | 'right-click' — surfaced in the sheet's eyebrow. */
  via: 'right-click' | 'shortcut'
}

/** What Settings and the HUD see: the prefs plus the ground truth main
 *  holds about them (whether the chord is actually registered, whether the
 *  optional hook loaded, whether this platform can place the window). */
export interface HudPrefsStatus extends HudPrefs {
  askError: 'invalid' | 'taken' | null
  askHookAvailable: boolean
  askHookReason: null | string
  askRegistered: boolean
  followSupported: boolean
}

/** Agents and rooms the HUD's switchers list — the same rows Quick Entry's
 *  launcher shows, cached in main from the primary renderer's state push. */
export interface HudLaunchOptions {
  agents: {
    color?: string
    displayName: string
    emoji?: string
    image?: string
    profile: string
    reachable: boolean
    title?: string
  }[]
  groups: { displayName: string; groupId: string; memberCount?: number; reachable: boolean }[]
}

export interface HudAskShortcutController {
  /** (Re)register the chord currently in prefs. Returns false when another
   *  app owns it — logged by the caller, never silent. */
  register(accelerator: string): boolean
  dispose(): void
  current(): null | string
}

export function createHudAskShortcut(globalShortcut: GlobalShortcutLike, onAsk: () => void): HudAskShortcutController {
  let active: null | string = null

  const release = () => {
    if (active) {
      try {
        globalShortcut.unregister(active)
      } catch {
        // Best effort — a dead accelerator must not block re-register.
      }

      active = null
    }
  }

  return {
    register(accelerator) {
      release()

      const parsed = parseQuickEntryShortcut(accelerator)

      if (!parsed.ok) {
        return false
      }

      let ok = false

      try {
        ok = globalShortcut.isRegistered(parsed.accelerator)
          ? false
          : globalShortcut.register(parsed.accelerator, onAsk)
      } catch {
        ok = false
      }

      active = ok ? parsed.accelerator : null

      return ok
    },
    dispose() {
      release()
    },
    current() {
      return active
    }
  }
}
