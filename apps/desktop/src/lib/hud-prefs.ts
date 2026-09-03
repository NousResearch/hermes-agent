/**
 * HUD follow / ask prefs and payloads — the renderer-visible mirror of the
 * main-process contract in electron/hud-ask.ts. Kept in step by hand; main
 * sanitizes every value it stores, so a drift here is a type error in the
 * bridge, never a runtime surprise.
 */

/** Which pet an agent gets on the HUD. */
export type HudPetChoice = 'avatar' | 'hank' | 'mina' | 'none'

export const HUD_PET_CHOICES: readonly HudPetChoice[] = ['hank', 'mina', 'avatar', 'none']

export interface HudPrefs {
  /** Lazy follow-the-pointer mode. */
  follow: boolean
  /** Global chord that opens the ask sheet at the cursor. */
  askShortcut: string
  /** Ctrl/⌘ + right-click anywhere opens the sheet (needs the optional hook). */
  askOnRightClick: boolean
  /** Pixel pets patrolling the strip above the bar. */
  pets: boolean
  /** Pet per agent, keyed by normalised profile name; missing = default. */
  petByAgent: Record<string, HudPetChoice>
}

export interface HudPrefsStatus extends HudPrefs {
  askError: 'invalid' | 'taken' | null
  askHookAvailable: boolean
  askHookReason: null | string
  askRegistered: boolean
  followSupported: boolean
}

export interface HudAskPayload {
  app: string
  cursor: { x: number; y: number }
  /** PNG crop around the cursor, on this machine (composer-images). Empty
   *  when the capture failed — the sheet still opens, without the picture. */
  imagePath: string
  /** Small data URL of the same crop for the sheet's preview. */
  thumbnail: string
  title: string
  via: 'right-click' | 'shortcut'
}

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

export const DEFAULT_HUD_ASK_SHORTCUT = 'CommandOrControl+Alt+H'

/** One line of a room's log as the HUD shows it. */
export interface HudRoomEntry {
  at: number
  /** Speaker: the member's profile name, or "You". */
  author: string
  id: string
  kind: 'member' | 'user'
  text: string
}

/** A room's recent log, pushed from the app window where the room engine
 *  runs. `turn` is the member currently replying, when a round is running. */
export interface HudRoomFeed {
  entries: HudRoomEntry[]
  groupId: string
  members: string[]
  running: boolean
  turn: null | string
}
