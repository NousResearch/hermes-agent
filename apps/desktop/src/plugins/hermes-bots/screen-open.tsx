/**
 * Open a bot's Screen as a main-window workspace tab (host.openWorkspace),
 * one tab per bot; a second open refocuses the existing tab.
 */

import { host } from '@hermes/plugin-sdk'

import { ensureBotMetadata } from './canonical-chat'
import { botSelectionKey, saveBotMeta } from './data'
import { botsText } from './i18n'
import { displayName } from './labels'
import { noteScreenTabClosed, noteScreenTabOpened } from './screen-autoraise'
import { displayRequest, type DisplayStatus } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { setScreenStatus } from './screen-state'
import { ID } from './shared'
import type { BotMeta, RosterRow } from './types'

const openTabs = new Map<string, () => void>()

export function screenPaneId(bot: RosterRow): string {
  return `plugin-workspace:${ID}:screen:${botSelectionKey(bot)}`
}

export function openBotScreen(bot: RosterRow, meta?: BotMeta | null): void {
  if (typeof host.openWorkspace !== 'function') {
    host.notify({ kind: 'info', message: botsText().screen.openNeedsUpdate })

    return
  }

  const key = botSelectionKey(bot)

  if (openTabs.has(key)) {
    host.revealPane(screenPaneId(bot))

    return
  }

  const close = host.openWorkspace(`${ID}:screen:${key}`, {
    title: `${displayName(bot, meta ?? null)} · ${botsText().screen.title}`,
    minWidth: '28rem',
    render: () => <BotScreenPane bot={bot} />,
    onClose: () => {
      openTabs.delete(key)
      noteScreenTabClosed(bot)
    }
  })

  openTabs.set(key, close)
  noteScreenTabOpened(bot)
}

/** Flip "open Screen when the bot uses it" for one bot, from what is stored for it, and say which
 *  way it went. */
export async function toggleBotScreenAutoOpen(bot: RosterRow): Promise<void> {
  const b = botsText()

  try {
    const current = await ensureBotMetadata(bot)
    const next = !current.screenAutoOpen

    void saveBotMeta(bot, { screenAutoOpen: next })
    host.notify({
      kind: 'info',
      message: (next ? b.screen.autoOpenOnToast : b.screen.autoOpenOffToast)(displayName(bot, current))
    })
  } catch (error) {
    host.notifyError?.(error, b.bot.metadataLoadFailed)
  }
}

/** Start a bot's screen: the request its pane makes, with the status the backend answers taken as
 *  what every surface shows. A screen that will not start rejects with the backend's reason. */
export async function startBotScreen(bot: RosterRow): Promise<void> {
  setScreenStatus(bot, await displayRequest<DisplayStatus>(bot, 'display.start'))
}
