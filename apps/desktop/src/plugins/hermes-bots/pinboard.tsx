/**
 * The iMessage-style bot pinboard that sits above the Sessions list.
 *
 * It is a launcher, not a second session browser: each tile represents one
 * source-qualified bot profile and opens that profile's canonical Bot Chat.
 */

import { cn, Codicon, host, SessionStatusDot, Tip, useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import { $botChatFocused, $focusedBotOwner, $selectedRosterKey, focusedRosterOwner } from './bot-state'
import {
  $botAttention,
  $botMeta,
  botActivitySession,
  botRosterKey,
  botSelectionKey,
  botSourceStatus,
  useRoster
} from './data'
import { $groupChatWorkspace } from './group-chat'
import { useBots } from './i18n'
import { displayName } from './labels'
import { pinboardMeta, pinnedBotRows } from './pinboard-model'
import { openRosterBot } from './roster-actions'
import { botCanonicalSessionId, botRowOwnsWorkspace, workerActiveAt } from './row-helpers'
import type { RosterRow } from './types'

/** The pinboard owns this much vertical space; its inner grid scrolls when the
 * roster grows instead of pushing the Sessions list out of the sidebar. */
export const BOT_PINBOARD_HEIGHT = '128px'

function botSourceLabel(bot: RosterRow): string | null {
  if (!bot.remoteSource && !bot.sourceScoped) {
    return null
  }

  const connectionLabel = bot.connectionLabel?.trim()

  if (connectionLabel) {
    return connectionLabel
  }

  const connectionId = bot.connectionId?.trim()

  if (connectionId && connectionId !== 'local') {
    return connectionId
  }

  return bot.remoteSource ? 'Remote source' : 'This device'
}

function openLabel(
  bot: RosterRow,
  meta: ReturnType<typeof pinboardMeta>,
  openBotChat: string,
  sourceLabel: string | null
): string {
  const label = displayName(bot, meta)

  return `${openBotChat}: ${label}${sourceLabel ? ` (${sourceLabel})` : ''}`
}

function warmBot(bot: RosterRow) {
  if (bot.sourceScoped && typeof host.warmAgent === 'function') {
    try {
      host.warmAgent(bot.connectionId, bot.name)
    } catch {
      /* warming is best-effort */
    }

    return
  }

  if (typeof host.warmProfile !== 'function') {
    return
  }

  try {
    host.warmProfile(bot.name)
  } catch {
    /* warming is best-effort */
  }
}

interface BotPinboardItemProps {
  active: boolean
  attention: null | { at: number; message: string; reason: string }
  bot: RosterRow
  gatewayState: string
  meta: ReturnType<typeof pinboardMeta>
  openBotChat: string
  pinnedLabel: string
}

function BotPinboardItem({
  active,
  attention,
  bot,
  gatewayState,
  meta,
  openBotChat,
  pinnedLabel
}: BotPinboardItemProps) {
  const activeProfile = useValue(host.state.profile)
  const sourceStatus = botSourceStatus(bot)
  const { color, image, shape } = botAppearance(bot.name, meta)
  const photo = Boolean(image && !isBackfilledFacePng(image))
  const activity = botActivitySession(bot)
  const workerActive = workerActiveAt(bot)
  const isGatewayHome = !bot.remoteSource && bot.name === activeProfile
  const mood = workerActive || (isGatewayHome && gatewayState === 'busy') ? 'work' : 'idle'
  const canonicalSessionId = botCanonicalSessionId(bot)
  const label = displayName(bot, meta)
  const sourceLabel = botSourceLabel(bot)

  const tooltip = [meta?.pinned ? pinnedLabel : null, label, sourceLabel, sourceStatus.label, activity?.preview]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className="min-w-0" role="gridcell">
      <Tip label={tooltip || label}>
        <button
          aria-current={active ? 'true' : undefined}
          aria-label={openLabel(bot, meta, openBotChat, sourceLabel)}
          className={cn(
            'flex w-full min-w-0 flex-col items-center gap-1 rounded-lg px-1.5 py-1.5 text-center transition-colors',
            'hover:bg-(--chrome-action-hover) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-(--ui-accent)',
            active && 'bg-(--ui-row-active-background)',
            !sourceStatus.available && 'opacity-65'
          )}
          data-bot-pin={botRosterKey(bot)}
          data-slot="bot-pinboard-item"
          onClick={() => void openRosterBot(bot)}
          onPointerEnter={() => warmBot(bot)}
          type="button"
        >
          <span
            className={cn(
              'relative grid size-14 shrink-0 place-items-center overflow-hidden rounded-full bg-(--chrome-action-hover) ring-1 ring-(--ui-stroke-tertiary)',
              active && 'ring-2 ring-(--ui-accent)',
              !sourceStatus.available && 'grayscale'
            )}
          >
            <BotFace
              color={avatarColor(color, bot.name)}
              image={photo ? image : null}
              mood={mood}
              name={bot.name}
              shape={shape}
              size={56}
            />
            <SessionStatusDot
              className="absolute bottom-0.5 right-0.5 rounded-full ring-2 ring-(--ui-bg-primary)"
              storedSessionId={canonicalSessionId}
            />
            {attention ? (
              <Tip label={attention.message || 'Needs attention'}>
                <span
                  aria-label="Needs attention"
                  className="absolute right-0.5 top-0.5 flex size-4 items-center justify-center rounded-full bg-(--ui-bg-primary) text-amber-600 dark:text-amber-300"
                  role="status"
                >
                  <Codicon name="warning" />
                </span>
              </Tip>
            ) : null}
          </span>
          <span className="max-w-full truncate text-[0.6875rem] font-medium text-(--ui-text-secondary)">{label}</span>
          {sourceLabel ? (
            <span className="max-w-full truncate text-[0.5625rem] text-(--ui-text-quaternary)">{sourceLabel}</span>
          ) : null}
        </button>
      </Tip>
    </div>
  )
}

export function BotPinboard() {
  const b = useBots()
  const { data, error } = useRoster()
  const allMeta = useValue($botMeta)
  const attentionByKey = useValue($botAttention)
  const focusedOwner = focusedRosterOwner(useValue($focusedBotOwner))
  const selectedRosterKey = useValue($selectedRosterKey)
  const botChatFocused = useValue($botChatFocused)
  const activeGroup = useValue($groupChatWorkspace)
  const gatewayState = useValue(host.state.gateway)
  const liveRoster = Array.isArray(data?.profiles) ? data.profiles : null
  // React Query retains the last successful data while a refresh error is
  // present, so this launcher stays useful without maintaining another cache.
  const roster = liveRoster || []
  const rows = pinnedBotRows(roster, allMeta)

  if (!rows.length) {
    const waiting = !data && !error

    return (
      <section
        aria-busy={waiting}
        aria-label={b.roster.botsOnly}
        className="flex min-h-0 flex-col border-b border-(--ui-stroke-secondary)"
        data-slot="bot-pinboard"
        style={{ maxHeight: BOT_PINBOARD_HEIGHT, minHeight: BOT_PINBOARD_HEIGHT }}
      >
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-3 py-2 text-center">
          <span className="text-[0.6875rem] font-medium text-(--ui-text-secondary)">
            {waiting ? b.roster.waitingForGateway : b.roster.emptyTitle}
          </span>
          {!waiting ? (
            <span className="mt-0.5 text-[0.625rem] text-(--ui-text-quaternary)">{b.roster.emptyDesc}</span>
          ) : null}
        </div>
      </section>
    )
  }

  return (
    <section
      aria-label={b.roster.botsOnly}
      className="flex min-h-0 flex-col border-b border-(--ui-stroke-secondary)"
      data-slot="bot-pinboard"
      style={{ maxHeight: BOT_PINBOARD_HEIGHT, minHeight: BOT_PINBOARD_HEIGHT }}
    >
      <div
        aria-label={b.roster.botsOnly}
        className="grid min-h-0 grid-cols-3 gap-x-1 gap-y-2 overflow-y-auto px-2 pb-2 pt-2"
        data-slot="bot-pinboard-grid"
        role="grid"
      >
        {rows.map(bot => {
          const meta = pinboardMeta(bot, allMeta)

          const attention =
            attentionByKey[botSelectionKey(bot)] ||
            attentionByKey[botRosterKey(bot)] ||
            attentionByKey[`${bot?.connectionId || 'local'}::${bot?.name || 'default'}`] ||
            null

          const active = botRowOwnsWorkspace(bot, activeGroup, botChatFocused, focusedOwner, selectedRosterKey)

          return (
            <BotPinboardItem
              active={active}
              attention={attention}
              bot={bot}
              gatewayState={gatewayState}
              key={botRosterKey(bot)}
              meta={meta}
              openBotChat={b.bot.openBotChat}
              pinnedLabel={b.roster.pinned}
            />
          )
        })}
      </div>
    </section>
  )
}
