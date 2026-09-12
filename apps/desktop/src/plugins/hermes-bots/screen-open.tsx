/**
 * Open a bot's Screen as a main-window workspace tab (host.openWorkspace),
 * one tab per bot; a second open refocuses the existing tab.
 */

import { host } from '@hermes/plugin-sdk'

import { botSelectionKey } from './data'
import { displayName } from './labels'
import { BotScreenPane } from './screen-pane'
import { ID } from './shared'
import type { BotMeta, RosterRow } from './types'

const openTabs = new Map<string, () => void>()

export function screenPaneId(bot: RosterRow): string {
  return `plugin-workspace:${ID}:screen:${botSelectionKey(bot)}`
}

export function openBotScreen(bot: RosterRow, meta?: BotMeta | null): void {
  if (typeof host.openWorkspace !== 'function') {
    host.notify({ kind: 'info', message: 'Update Hermes Desktop to open bot screens.' })

    return
  }

  const key = botSelectionKey(bot)

  if (openTabs.has(key)) {
    host.revealPane(screenPaneId(bot))

    return
  }

  const close = host.openWorkspace(`${ID}:screen:${key}`, {
    title: `${displayName(bot, meta ?? null)} · Screen`,
    minWidth: '28rem',
    render: () => <BotScreenPane bot={bot} />,
    onClose: () => {
      openTabs.delete(key)
    }
  })

  openTabs.set(key, close)
}
