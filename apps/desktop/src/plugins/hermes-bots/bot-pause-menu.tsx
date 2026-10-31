/**
 * The bot row's Pause pieces: the paused glyph beside the name, the face dim,
 * and the Pause / Resume item plus the Notifications submenu in its context
 * menu. Kept out of `bot-row.tsx` so the row stays one readable component.
 */

import {
  Codicon,
  ContextMenuCheckboxItem,
  ContextMenuItem,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  Tip
} from '@hermes/plugin-sdk'

import {
  BOT_NOTIFICATION_LEVEL_LABELS,
  BOT_NOTIFICATION_LEVELS,
  setBotNotificationLevel,
  toggleBotPaused
} from './bot-pause'
import { useBots } from './i18n'
import type { BotNotificationLevel, RosterRow } from './types'

/** A paused bot's face reads as idle, like an unreachable one. */
export const pausedFaceClass = (paused: boolean) => (paused ? 'grayscale opacity-60' : '')

export function BotPausedGlyph({ paused }: { paused: boolean }) {
  const b = useBots()

  if (!paused) {
    return null
  }

  return (
    <Tip label={b.pause.pausedBadge}>
      <Codicon
        aria-label={b.pause.pausedBadge}
        className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)"
        name="debug-pause"
      />
    </Tip>
  )
}

interface BotPauseMenuProps {
  bot: RosterRow
  level: BotNotificationLevel
  paused: boolean
}

/** Pause is enforced by the bot's backend (routines, teammate DMs); Hide,
 *  beside it in the menu, stays display-only. */
export function BotPauseMenu({ bot, level, paused }: BotPauseMenuProps) {
  const b = useBots()

  return (
    <>
      <ContextMenuItem onSelect={() => void toggleBotPaused(bot)}>
        <Codicon className="mr-1.5" name={paused ? 'debug-continue' : 'debug-pause'} />
        {paused ? b.pause.resume : b.pause.pause}
      </ContextMenuItem>
      <ContextMenuSub>
        <ContextMenuSubTrigger>{b.pause.notifications}</ContextMenuSubTrigger>
        <ContextMenuSubContent>
          {BOT_NOTIFICATION_LEVELS.map(option => (
            <ContextMenuCheckboxItem
              checked={level === option}
              key={option}
              onSelect={() => void setBotNotificationLevel(bot, option)}
            >
              {b.pause[BOT_NOTIFICATION_LEVEL_LABELS[option]]}
            </ContextMenuCheckboxItem>
          ))}
        </ContextMenuSubContent>
      </ContextMenuSub>
    </>
  )
}
