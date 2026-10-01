/**
 * Raise a bot's Screen tab the moment the bot starts driving its desktop, so
 * you watch it work instead of finding out afterwards. Opt-in per bot
 * (`BotMeta.screenAutoOpen`, row context menu): the Desktop rule is "offer,
 * don't hijack", and a tab appearing because a background bot touched its
 * screen is a hijack unless the user asked for exactly that.
 *
 * What counts as "started driving": a live `tool.start` for a screen tool
 * (computer_use, the browser tools) on a session the bot owns. Live means
 * arriving on the socket now; replayed history never opens anything — the
 * reconnect replay in `@hermes/shared` re-dispatches parked frames, so a
 * seq-gated ring is not enough and the wake is also rate-limited per bot.
 *
 * Fencing rules, in order:
 *   - the tab is already open → nothing (never re-front over what you moved to)
 *   - a manual Close holds until the NEXT fresh screen use, never re-raises
 *     mid-burst: a user closing the tab while the bot clicks around has said
 *     "not now", and the next tool.start of the same run must respect that
 *   - one wake per bot per 30 s
 * Opening never moves keyboard focus: `openBotScreen` reveals the pane, and the
 * viewer only grabs keys when the human takes over.
 */

import { host } from '@hermes/plugin-sdk'
import type { RpcEvent } from '@hermes/plugin-sdk'

import { $botMeta, $lastRoster, botSelectionKey } from './data'
import { botRosterMeta, resolveBotConnectionRoute } from './routing'
import { openBotScreen, screenPaneId } from './screen-open'
import type { RosterRow } from './types'

/** Tools whose first call means the bot is now using its screen. Browser tools count:
 *  on a Bot Screen host the headed Chromium is on that desktop. */
const SCREEN_TOOL_PREFIXES = ['computer_use', 'browser_']

export const AUTO_RAISE_COOLDOWN_MS = 30_000

/** Per-bot wake bookkeeping, keyed by `botSelectionKey`. */
interface RaiseState {
  lastRaisedAt: number
  /** Set when the user closed the tab; cleared by a quiet gap, so the next burst may raise again. */
  closedAt: number
}

const raiseState = new Map<string, RaiseState>()
/** Bot keys whose Screen tab is open right now (mirrors `screen-open`'s tab map without coupling to it). */
const openTabs = new Set<string>()

export function isScreenTool(name: string | undefined): boolean {
  return typeof name === 'string' && SCREEN_TOOL_PREFIXES.some(prefix => name.startsWith(prefix))
}

/** Resolve the roster row that owns an event: same source connection (a local/legacy
 *  event carries no tag and matches a `local` route) and the event's profile. */
export function botForEvent(roster: RosterRow[], event: Pick<RpcEvent, 'connectionId' | 'profile'>): RosterRow | null {
  const profile = (event.profile ?? '').trim()

  if (!profile) {
    return null
  }

  for (const bot of roster) {
    const resolved = resolveBotConnectionRoute(bot)

    if (resolved.status === 'owner_removed') {
      continue
    }

    const route = resolved.route
    const botProfile = route ? route.targetProfile || route.profile : bot.name
    const expected = route?.connectionId ?? null
    const actual = event.connectionId ?? null

    const sameSource =
      expected === actual || (expected === 'local' && actual === null) || (expected === null && actual === 'local')

    if (sameSource && botProfile === profile) {
      return bot
    }
  }

  return null
}

/** Pure decision: should a live screen tool.start for `bot` raise its tab now? */
export function shouldAutoRaise(
  bot: RosterRow,
  now: number,
  state: RaiseState | undefined,
  tabOpen: boolean,
  meta = $botMeta.get()
): boolean {
  if (!botRosterMeta(bot, meta)?.screenAutoOpen || tabOpen) {
    return false
  }

  if (!state) {
    return true
  }

  // A Close during a burst holds through the burst: the gap must be at least one cooldown.
  if (state.closedAt && now - state.closedAt < AUTO_RAISE_COOLDOWN_MS) {
    return false
  }

  return now - state.lastRaisedAt >= AUTO_RAISE_COOLDOWN_MS
}

export function noteScreenTabOpened(bot: RosterRow): void {
  openTabs.add(botSelectionKey(bot))
}

export function noteScreenTabClosed(bot: RosterRow, now = Date.now()): void {
  const key = botSelectionKey(bot)

  openTabs.delete(key)
  raiseState.set(key, { lastRaisedAt: raiseState.get(key)?.lastRaisedAt ?? 0, closedAt: now })
}

export function handleScreenToolStart(event: RpcEvent, now = Date.now()): boolean {
  const payload = event.payload as { name?: string } | undefined

  if (!isScreenTool(payload?.name)) {
    return false
  }

  const bot = botForEvent($lastRoster.get(), event)

  if (!bot) {
    return false
  }

  const key = botSelectionKey(bot)

  if (!shouldAutoRaise(bot, now, raiseState.get(key), openTabs.has(key))) {
    return false
  }

  raiseState.set(key, { lastRaisedAt: now, closedAt: 0 })
  openBotScreen(bot, botRosterMeta(bot, $botMeta.get()) ?? null)

  return true
}

/** Wire the listener; returns the disposer. Feature-detected on older shells. */
export function startScreenAutoRaise(): () => void {
  if (typeof host.onEvent !== 'function') {
    return () => undefined
  }

  return host.onEvent('tool.start', (event: RpcEvent) => {
    handleScreenToolStart(event)
  })
}

/** Test seam. */
export function resetScreenAutoRaise(): void {
  raiseState.clear()
  openTabs.clear()
}

export function screenTabPaneId(bot: RosterRow): string {
  return screenPaneId(bot)
}
