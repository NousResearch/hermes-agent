import { PALETTE_AREA } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'

import { $botMeta, $lastRoster, botRosterKey, cachedUnionRoster } from './data'
import { isBotHidden } from './hidden-bots'
import { displayName } from './labels'
import { openRosterBot } from './roster-actions'
import { botRosterMeta } from './routing'
import type { RosterRow } from './types'

/** The palette opens the same canonical chat as a click in the Bots roster. */
export function registerBotChatPalette(ctx: PluginContext): () => void {
  const entries = new Map<string, { label: string; dispose: () => void }>()
  const roster = (): RosterRow[] => cachedUnionRoster()?.profiles ?? $lastRoster.get()

  const sync = () => {
    const meta = $botMeta.get()
    const bots = roster().filter(bot => bot?.name && !isBotHidden(bot, meta))
    const names = new Map<string, number>()

    for (const bot of bots) {
      const name = displayName(bot, botRosterMeta(bot, meta))
      names.set(name, (names.get(name) || 0) + 1)
    }

    const current = new Set<string>()

    for (const bot of bots) {
      const key = botRosterKey(bot)
      const name = displayName(bot, botRosterMeta(bot, meta))
      const label = `Chat with ${name}${names.get(name)! > 1 ? ` (${bot.connectionLabel || bot.connectionId || bot.name})` : ''}`
      current.add(key)

      if (entries.get(key)?.label === label) {
        continue
      }

      entries.get(key)?.dispose()

      const dispose = ctx.register({
        id: `chat-${key}`,
        area: PALETTE_AREA,
        data: {
          id: `hermes-bots.chat-${key}`,
          label,
          keywords: ['bot', 'profile', 'agent', 'chat', bot.name, name],
          run: () => {
            const latest = roster().find(row => botRosterKey(row) === key)

            if (latest) {
              void openRosterBot(latest)
            }
          }
        }
      })

      entries.set(key, { label, dispose })
    }

    for (const [key, entry] of entries) {
      if (current.has(key)) {
        continue
      }

      entry.dispose()
      entries.delete(key)
    }
  }

  const unbindRoster = $lastRoster.listen(sync)
  const unbindMeta = $botMeta.listen(sync)
  ctx.onDispose(() => {
    unbindRoster()
    unbindMeta()

    for (const entry of entries.values()) {
      entry.dispose()
    }

    entries.clear()
  })
  sync()

  return sync
}
