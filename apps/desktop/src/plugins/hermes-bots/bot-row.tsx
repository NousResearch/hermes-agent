/**
 * The two rows the Bots rail lists: one bot, and one group chat.
 *
 * Rows only. Both take their bot or room and their context-menu callbacks as
 * props and own no roster state, so the pane above them decides what to list
 * and the row decides only how it reads.
 */

import {
  cn,
  coarseElapsed,
  Codicon,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
  haptic,
  host,
  queryClient,
  RowButton,
  SessionStatusDot,
  SidebarRowLead,
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import {
  $botChatFocused,
  $focusedBotOwner,
  $selectedRosterKey,
  focusedRosterOwner,
  saveSelectedRosterBot
} from './bot-state'
import { ensureBotMetadata } from './canonical-chat'
import {
  $botAttention,
  $botMeta,
  $lastRoster,
  BOT_ATTENTION_HINTS,
  botActivitySession,
  botHandle,
  botRosterKey,
  botSelectionKey,
  botSourceStatus,
  isActiveRosterBot,
  isDefaultBot,
  newBotChat,
  ROSTER_KEY,
  saveBotMeta
} from './data'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import { botGroups, groupLastActivity } from './group-membership'
import { fallbackSelectionAfterHide, isBotHidden, isBotPinned } from './hidden-bots'
import { useBots } from './i18n'
import { displayName, stripPreviewMarkdown } from './labels'
import { duplicateBot } from './profile-ops'
import { openRosterBot } from './roster-actions'
import { botRosterMeta, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { A2A_PREFIX_RE, botCanonicalSessionId, botRowOwnsWorkspace, previewKind, workerActiveAt } from './row-helpers'
import type { GroupMember, RosterRow, SidebarRowLabels } from './types'

// ── bot row ──────────────────────────────────────────────────────────────────

/** Row age in the sidebar's compact form ("now", "52m", "3h", "18d").
 *  Deliberately the same `coarseElapsed` + suffix pair the session rows
 *  directly above use, so the two lists in one rail don't disagree about how
 *  an age is spelled. Not `relativeTime` — that's the bidirectional Intl form
 *  ("in 14 hr"), which belongs on a scheduled next-run. */
function rowAge(ms: number, r: SidebarRowLabels): string {
  const { unit, value } = coarseElapsed(Date.now() - ms)

  return unit === 'second' ? r.ageNow : `${value}${unit === 'day' ? r.ageDay : unit === 'hour' ? r.ageHour : r.ageMin}`
}

interface BotRowProps {
  bot: RosterRow
  onDelete: (bot: RosterRow) => void
  onEdit: (bot: RosterRow) => void
  onGroup: (bot: RosterRow) => void
  showHandle?: boolean
}

export function BotRow({ bot, onDelete, onEdit, onGroup, showHandle }: BotRowProps) {
  const { t } = useI18n()
  const b = useBots()
  const activeProfile = useValue(host.state.profile)
  const focusedOwner = focusedRosterOwner(useValue($focusedBotOwner))
  const selectedRosterKey = useValue($selectedRosterKey)
  const botChatFocused = useValue($botChatFocused)
  const activeGroup = useValue($groupChatWorkspace)
  const allMeta = useValue($botMeta)
  const meta = botRosterMeta(bot, allMeta)
  const hidden = isBotHidden(bot, allMeta)
  const pinned = isBotPinned(bot, allMeta)
  const sourceStatus = botSourceStatus(bot)
  const groups = botGroups(meta)
  const last = bot.last_session
  // Highlight follows the chat on screen (focused session's owner), not the
  // gateway socket's home — a focused tab doesn't swap the socket, and on the
  // old keying the wrong bot stayed highlighted while you read another's chat.
  // A selected group chat suppresses every bot-row highlight: the group row
  // owns the selection then (#88979).
  const activeConnectionId = String(host.state.connectionId?.get?.() || 'local').trim()
  // The highlight follows whoever owns the MAIN workspace. While a chat owns
  // it, that chat's profile wins (a stale roster click must not key the
  // highlight to a bot you are not reading). With no focused chat, the
  // source-qualified selection is the owner — and it is the only rule that
  // can highlight a remote row, which has no focusable local chat.
  const isActive = botRowOwnsWorkspace(bot, activeGroup, botChatFocused, focusedOwner, selectedRosterKey)

  // Turn-busy is a SOCKET fact: only the gateway-home profile can be mid-turn.
  const isGatewayHome =
    !bot.remoteSource &&
    bot.name === activeProfile &&
    isActiveRosterBot(bot, {
      name: activeProfile,
      connectionId: activeConnectionId
    })

  const { shape, color, image } = botAppearance(bot.name, meta)
  // Keep user photos/pets. Drop the 160px SVG backfill so the math face can move.
  const photo = Boolean(image && !isBackfilledFacePng(image))
  const gatewayState = useValue(host.state.gateway)
  // Preview identity must match click identity (#88200): when the backend
  // resolved the pinned canonical chat, preview THAT session — not the
  // profile's most recent (but unrelated) activity. Activity signals
  // (age label, pulse dot) follow the same rule via botActivitySession:
  // the canonical Bot Chat is hidden from last_session, so keying age off
  // last_session alone shows "6d ago" on a bot you just messaged.
  const previewSession = bot.canonical_session || last
  const activitySession = botActivitySession(bot)
  // A live kanban/tool worker counts as activity (#90268): fresh age while it
  // runs, falling back to chat activity when it ends.
  const workerActive = workerActiveAt(bot)

  const rowAgeTs = workerActive
    ? Math.max(activitySession?.last_active || 0, bot.worker_session?.last_active || 0)
    : activitySession?.last_active || 0

  const botMood = workerActive || (isGatewayHome && gatewayState === 'busy') ? 'work' : 'idle'
  // Status keys off the canonical Bot Chat — the very session this row opens,
  // so the dot and the click can never describe different conversations.
  const canonicalSessionId = botCanonicalSessionId(bot)
  // Needs-attention badge (#93091 item 3): background failures record under
  // the selection key (group turns) or the route key (relay deliveries) —
  // check both. Local/unannotated rows carry no connectionId, so their relay
  // failures live under `<activeConnectionId>::<name>` — resolve that shape
  // too or active-gateway bots never badge. Hidden bots keep their entry;
  // hiding is display-only.
  const attentionByKey = useValue($botAttention)

  const attention =
    attentionByKey[botSelectionKey(bot)] ||
    attentionByKey[botRosterKey(bot)] ||
    attentionByKey[`${bot?.connectionId || activeConnectionId}::${bot?.name || 'default'}`] ||
    null

  // WHO sent the last message (bot-to-bot DM vs human) — the full stored
  // history lives in the canonical chat, not inline.
  // Preview identity must match click identity (#88200): when the backend
  // resolved the pinned canonical chat, preview THAT session — not the
  // profile's most recent (but unrelated) activity. Liveness checks above
  // keep last_session semantics: any recent activity means the bot is alive.
  const { fromBot } = previewKind(previewSession?.preview)

  // DM previews read like DMs: strip the delivery prefix, keep the message.
  const displayPreview = stripPreviewMarkdown(
    fromBot ? (previewSession?.preview || '').replace(A2A_PREFIX_RE, '').trim() || '…' : previewSession?.preview || ''
  )

  const handle = botHandle(bot.name, bot)
  const gatewayLabel = bot.connectionLabel || (bot.connectionId === 'local' ? 'This device' : '')
  const showDetailsRow = Boolean(showHandle || displayPreview || fromBot)

  const rowTooltip = [displayName(bot, meta), `@${handle}`, gatewayLabel, sourceStatus.label]
    .filter(Boolean)
    .join(' · ')

  const warm = () => {
    // Multi-source row: pre-dial the agent's OWN source (feature-detected).
    if (bot.sourceScoped && typeof host.warmAgent === 'function') {
      try {
        host.warmAgent(bot.connectionId, bot.name)
      } catch {
        /* warm is best-effort */
      }

      return
    }

    if (typeof host.warmProfile !== 'function') {
      return
    }

    try {
      host.warmProfile(bot.name)
    } catch {
      /* warm is best-effort */
    }
  }

  // Rows and Active Now share the exact-owner open path; only that path may
  // activate a source and resolve the canonical Bot Chat.
  const open = () => void openRosterBot(bot)

  const row = (
    <RowButton
      aria-label={rowTooltip}
      className={cn(
        'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
        'hover:bg-(--chrome-action-hover)',
        isActive && 'bg-(--ui-row-active-background)'
      )}
      onClick={open}
      onPointerEnter={warm}
    >
      <div className={cn('shrink-0', !sourceStatus.available && 'grayscale opacity-60')}>
        <BotFace
          color={avatarColor(color, bot.name)}
          image={photo ? image : null}
          mood={botMood}
          name={bot.name}
          shape={shape}
          size={34}
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <div className="flex min-w-0 items-center gap-1.5">
            {/* The session row's own lead cell, so a bot's name sits on the
                same left edge as every session name above it in the rail. */}
            <SidebarRowLead>
              <SessionStatusDot storedSessionId={canonicalSessionId} />
            </SidebarRowLead>
            {pinned ? (
              <Tip label={b.roster.pinned}>
                <Codicon className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)" name="pinned" />
              </Tip>
            ) : null}
            {hidden ? (
              <Tip label={b.roster.hiddenFromRoster}>
                <Codicon className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)" name="eye-closed" />
              </Tip>
            ) : null}
            <Tip label={rowTooltip}>
              <span className="min-w-0 truncate text-[0.8125rem] font-medium">{displayName(bot, meta)}</span>
            </Tip>
          </div>
          {attention ? (
            <Tip label={BOT_ATTENTION_HINTS[attention.reason] || 'Needs attention'}>
              <Codicon
                aria-label={b.roster.needsAttention}
                className="shrink-0 text-[0.6875rem] text-amber-600 dark:text-amber-300"
                name="warning"
              />
            </Tip>
          ) : null}
          {rowAgeTs ? (
            <span className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)">
              {rowAge(rowAgeTs * 1000, t.sidebar.row)}
            </span>
          ) : null}
        </div>
        {showDetailsRow ? (
          <div className="flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)">
            {showHandle ? (
              <span className="shrink-0 font-mono text-[0.6875rem] text-(--ui-text-quaternary)">{`@${handle}`}</span>
            ) : null}
            {showHandle && displayPreview ? <span className="shrink-0 text-(--ui-text-quaternary)">·</span> : null}
            {displayPreview ? (
              <span className={cn('min-w-0 truncate', fromBot && 'italic')}>{displayPreview}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </RowButton>
  )

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{row}</ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => void openRosterBot(bot, { canonical: true })}>
          {b.bot.openBotChat}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() => {
            void ensureBotMetadata(bot)
              .then(current => {
                const pinned = Boolean(current.pinned)
                void saveBotMeta(bot, {
                  pinned: !pinned
                })
                host.notify({
                  kind: 'info',
                  message: `${displayName(bot, current)} ${pinned ? 'unpinned' : 'pinned to top'}`
                })
              })
              .catch(error => host.notifyError?.(error, 'Could not load bot metadata'))
          }}
        >
          {pinned ? 'Unpin' : 'Pin to top'}
        </ContextMenuItem>
        <ContextMenuItem
          onSelect={() => {
            void ensureBotMetadata(bot)
              .then(current => {
                const hidden = Boolean(current.hidden)
                void saveBotMeta(bot, {
                  hidden: !hidden
                })

                if (!hidden) {
                  fallbackSelectionAfterHide(botSelectionKey(bot))
                }

                host.notify({
                  kind: 'info',
                  message: hidden
                    ? `${displayName(bot, current)} is back in the roster`
                    : `${displayName(bot, current)} hidden — use the eye button in the Bots header to see hidden bots`
                })
              })
              .catch(error => host.notifyError?.(error, 'Could not load bot metadata'))
          }}
        >
          {hidden ? 'Unhide' : 'Hide'}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() =>
            void ensureBotMetadata(bot)
              .then(() => onEdit(bot))
              .catch(error => host.notifyError?.(error, 'Could not load bot'))
          }
        >
          {b.bot.editMenu}
        </ContextMenuItem>
        <ContextMenuItem
          onSelect={() =>
            void ensureBotMetadata(bot)
              .then(() => onGroup(bot))
              .catch(error => host.notifyError?.(error, 'Could not load bot groups'))
          }
        >
          {groups.length ? `Groups: ${groups.join(', ')}…` : 'Manage groups…'}
        </ContextMenuItem>
        <ContextMenuItem
          onSelect={() => {
            host.notify({
              kind: 'info',
              message: `Duplicating ${displayName(bot, meta)}…`
            })
            duplicateBot(bot, $lastRoster.get())
              .then(name => {
                queryClient.invalidateQueries({
                  queryKey: ROSTER_KEY
                })
                host.notify({
                  kind: 'success',
                  message: `Created ${name} — full copy of ${bot.name}`
                })
              })
              .catch(err => host.notifyError(err, b.bot.duplicateFailed))
          }}
        >
          {b.bot.duplicate}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() => {
            saveSelectedRosterBot(bot)
            setBotsWorkspaceOwner(botWorkspaceOwnerKey(bot), bot)
            newBotChat(bot)
          }}
        >
          {b.bot.newChatWith}
        </ContextMenuItem>
        {isDefaultBot(bot) ? null : <ContextMenuSeparator />}
        {isDefaultBot(bot) ? null : (
          <ContextMenuItem onSelect={() => onDelete(bot)} variant="destructive">
            {t.common.delete}
          </ContextMenuItem>
        )}
      </ContextMenuContent>
    </ContextMenu>
  )
}

/** One group chat as one quiet roster row. The room owns one visual identity;
 *  member details stay in its tooltip and workspace instead of competing
 *  with bot avatars in the narrow sidebar. */
interface GroupRowProps {
  active: boolean
  group: string
  members: GroupMember[]
  needsYou: boolean
  onDisband: (room: { members: GroupMember[]; name: string }) => void
  onOpen: (group: string) => void
}

export function GroupRow({ active, group, members, needsYou, onOpen, onDisband }: GroupRowProps) {
  const { t } = useI18n()
  const b = useBots()
  const rooms = useValue($groupChats)

  const room = rooms[group] || {
    log: []
  }

  const log = Array.isArray(room.log) ? room.log : []
  const last = log.length ? log[log.length - 1] : null
  const lastAt = groupLastActivity(room)
  // Room previews speak the same handle vocabulary as the roster, mentions
  // and the group prompt: the primary profile is @hermes, not @default.
  const lastFrom = last?.from?.name || ''

  const lastHandle = botHandle(
    lastFrom || 'bot',
    members.find(member => member?.name === lastFrom)
  )

  const preview = last
    ? `${last.from?.kind === 'user' ? 'You' : `@${lastHandle}`}: ${stripPreviewMarkdown(last.text) || '…'}`
    : `${members.length} bots`

  const availableMembers = members.filter(member => botSourceStatus(member).available).length
  const availabilityLabel = `${availableMembers} of ${members.length} available`

  const row = (
    <RowButton
      aria-label={`${group}, ${members.length} bots, ${availabilityLabel}`}
      className={cn(
        'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
        'hover:bg-(--chrome-action-hover)',
        active && 'bg-(--ui-row-active-background)'
      )}
      onClick={() => {
        haptic('tap')
        onOpen(group)
      }}
    >
      <div className="relative flex w-[34px] shrink-0 items-center justify-center">
        {room.image ? (
          <img
            alt=""
            className={cn(
              'size-8 rounded-md object-cover ring-1 ring-(--ui-stroke-tertiary)',
              availableMembers === 0 && 'grayscale opacity-60'
            )}
            src={room.image}
          />
        ) : (
          <span
            className={cn(
              'flex size-8 items-center justify-center rounded-md bg-(--chrome-action-hover) text-(--ui-text-tertiary)',
              availableMembers === 0 && 'opacity-60'
            )}
          >
            <Codicon name="organization" />
          </span>
        )}
        {members.length > 0 && availableMembers < members.length ? (
          <Tip label={availabilityLabel}>
            <span
              aria-label={availabilityLabel}
              className="absolute -bottom-0.5 -right-0.5 flex size-4 items-center justify-center rounded-full bg-(--ui-bg-primary) text-[0.625rem] text-amber-600 ring-1 ring-(--ui-stroke-tertiary) dark:text-amber-300"
            >
              <Codicon name="debug-disconnect" />
            </span>
          </Tip>
        ) : null}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium">{group}</span>
          {needsYou ? (
            <Tip label={b.group.needsYourInput}>
              <Codicon aria-label={b.roster.needsInput} className="shrink-0 text-(--ui-accent)" name="question" />
            </Tip>
          ) : null}
          {lastAt ? (
            <span className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)">
              {rowAge(lastAt, t.sidebar.row)}
            </span>
          ) : null}
        </div>
        <div className="min-w-0 truncate text-xs text-(--ui-text-tertiary)">{preview}</div>
      </div>
    </RowButton>
  )

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{row}</ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => onOpen(group)}>Open Group Chat</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          className="text-destructive focus:text-destructive"
          onSelect={() =>
            onDisband({
              name: group,
              members
            })
          }
        >
          {b.group.deleteAction}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
