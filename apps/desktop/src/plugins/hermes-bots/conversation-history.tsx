/**
 * Durable conversation history for one Bot profile.
 *
 * Bot workspace tabs are a presentation cache. This browser reads the profile's
 * state.db through the source-primary SDK door, so a closed/restarted Desktop
 * cannot strand a real conversation just because its tab is no longer open.
 */

import {
  Button,
  cn,
  coarseElapsed,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  GlyphSpinner,
  host,
  PanelEmpty,
  RowButton,
  SearchField,
  SessionStatusDot,
  SidebarRowLead,
  useI18n,
  useQuery
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useState } from 'react'

import { saveSelectedRosterBot } from './bot-state'
import { $botMeta, botRosterKey } from './data'
import { $groupChatWorkspace } from './group-chat'
import { closeGroupChatMainTab } from './group-panes'
import { useBots } from './i18n'
import { displayName } from './labels'
import { supersedeRosterOpen } from './roster-actions'
import {
  backendTargetProfile,
  botConnectionRoute,
  botRosterMeta,
  botWorkspaceOwnerKey,
  setBotsWorkspaceOwner
} from './routing'
import type { RosterRow } from './types'

const HISTORY_PAGE_SIZE = 200
const HISTORY_QUERY_KEY = ['hermes-bots', 'conversation-history'] as const

export interface BotConversationSummary {
  _lineage_ids?: null | string[]
  _lineage_root_id?: null | string
  _lineage_root_title?: null | string
  archived?: boolean
  id: string
  last_active?: number
  message_count?: number
  preview?: null | string
  source?: null | string
  started_at?: number
  title?: null | string
}

interface ConversationHistoryResult {
  errors?: Array<{ error: string; profile: string }>
  sessions?: BotConversationSummary[]
  total?: number
}

function sessionIds(session: BotConversationSummary): Set<string> {
  return new Set(
    [session.id, session._lineage_root_id, ...(Array.isArray(session._lineage_ids) ? session._lineage_ids : [])]
      .filter(Boolean)
      .map(String)
  )
}

/** Keep user conversations; Bot Mode's own forever-chat, room plumbing and
 * routine execution history already have dedicated surfaces. */
export function isBrowsableBotConversation(bot: RosterRow, session: BotConversationSummary): boolean {
  if (!session?.id || Number(session.message_count || 0) < 1 || session.source === 'cron') {
    return false
  }

  const ids = sessionIds(session)
  const canonicalIds = [bot.canonical_session?.id, bot.canonical_session?.resolved_id].filter(Boolean).map(String)

  if (canonicalIds.some(id => ids.has(id))) {
    return false
  }

  const rootTitle = String(session._lineage_root_title || '').trim()
  const title = String(session.title || '').trim()
  const plumbingTitle = rootTitle || title

  return plumbingTitle !== 'Bot Chat' && plumbingTitle !== 'Agent Inbox' && !plumbingTitle.startsWith('Group: ')
}

/** Read every page rather than treating the SDK's 500-row request ceiling as
 * a history ceiling. Offset advances by the requested window because pinned
 * rows can be back-filled past a page and are deduplicated by durable id. */
export async function loadBotConversationHistory(bot: RosterRow): Promise<BotConversationSummary[]> {
  if (typeof host.listPersistedSessions !== 'function') {
    throw new Error('Update Hermes Desktop to browse Bot conversation history.')
  }

  const route = botConnectionRoute(bot)
  const profile = backendTargetProfile(route, String(bot?.name || '').trim() || 'default')
  const conversations = new Map<string, BotConversationSummary>()
  let offset = 0

  for (;;) {
    const result = (await host.listPersistedSessions(route, {
      profile,
      limit: HISTORY_PAGE_SIZE,
      offset,
      minMessages: 1,
      archived: 'include',
      order: 'recent',
      includeHidden: true
    })) as ConversationHistoryResult

    if (Array.isArray(result.errors) && result.errors.length) {
      throw new Error(result.errors[0].error || `Could not read ${profile} conversations`)
    }

    const rows = Array.isArray(result.sessions) ? result.sessions : []

    for (const session of rows) {
      if (isBrowsableBotConversation(bot, session)) {
        conversations.set(session.id, session)
      }
    }

    const total = Number(result.total)
    offset += HISTORY_PAGE_SIZE

    if (!rows.length || (Number.isFinite(total) ? offset >= total : rows.length < HISTORY_PAGE_SIZE)) {
      break
    }
  }

  return [...conversations.values()].sort(
    (left, right) => Number(right.last_active || right.started_at || 0) - Number(left.last_active || left.started_at || 0)
  )
}

export async function openBotConversation(bot: RosterRow, session: BotConversationSummary): Promise<void> {
  if (typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions.')
  }

  supersedeRosterOpen()
  const route = botConnectionRoute(bot)
  const ownerKey = botWorkspaceOwnerKey(bot)
  const group = $groupChatWorkspace.get()

  if (group) {
    closeGroupChatMainTab(group)
    $groupChatWorkspace.set(null)
  }

  saveSelectedRosterBot(bot)
  setBotsWorkspaceOwner(ownerKey, bot)

  await host.openSession(session.id, {
    ...(route ? { route } : {}),
    profile: bot.name,
    intent: 'tab',
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey,
    tabTitle: String(session.title || session.preview || '').trim() || 'Conversation'
  })

  host.focusOpenWorkspaceSession?.(ownerKey, undefined, [session.id])
}

function historyAge(seconds: number, labels: ReturnType<typeof useI18n>['t']['sidebar']['row']): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return ''
  }

  const { unit, value } = coarseElapsed(Date.now() - seconds * 1000)

  return unit === 'second' ? labels.ageNow : `${value}${unit === 'day' ? labels.ageDay : unit === 'hour' ? labels.ageHour : labels.ageMin}`
}

function conversationTitle(session: BotConversationSummary, untitled: string): string {
  return String(session.title || '').trim() || String(session.preview || '').trim() || untitled
}

export function BotConversationHistoryDialog({
  bot,
  onOpenChange,
  open
}: {
  bot: null | RosterRow
  onOpenChange: (open: boolean) => void
  open: boolean
}) {
  const { t } = useI18n()
  const b = useBots()
  const allMeta = $botMeta.get()
  const [query, setQuery] = useState('')
  const [openingId, setOpeningId] = useState<null | string>(null)
  const botKey = bot ? botRosterKey(bot) : ''
  const botName = bot ? displayName(bot, botRosterMeta(bot, allMeta)) : ''

  const history = useQuery({
    queryKey: [...HISTORY_QUERY_KEY, botKey],
    queryFn: () => loadBotConversationHistory(bot!),
    enabled: open && Boolean(bot),
    staleTime: 5_000
  })

  useEffect(() => {
    if (!open) {
      setQuery('')
      setOpeningId(null)
    }
  }, [open])

  const sessions = useMemo(() => (Array.isArray(history.data) ? history.data : []), [history.data])

  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()

    if (!needle) {
      return sessions
    }

    return sessions.filter(session =>
      [session.title, session.preview, session.id].some(value => String(value || '').toLocaleLowerCase().includes(needle))
    )
  }, [query, sessions])

  const openConversation = async (session: BotConversationSummary) => {
    if (!bot || openingId) {
      return
    }

    setOpeningId(session.id)

    try {
      await openBotConversation(bot, session)
      onOpenChange(false)
    } catch (error) {
      host.notifyError?.(error, b.history.openFailed)
      setOpeningId(null)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="w-[min(40rem,94vw)] max-w-none overflow-hidden" data-slot="bot-conversation-history">
        <DialogHeader>
          <DialogTitle>{b.history.title(botName)}</DialogTitle>
          <DialogDescription>{b.history.description}</DialogDescription>
        </DialogHeader>

        {history.isLoading && !sessions.length ? (
          <div className="grid h-40 place-items-center">
            <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
          </div>
        ) : history.error && !sessions.length ? (
          <PanelEmpty
            action={
              <Button onClick={() => void history.refetch()} size="sm" variant="secondary">
                {t.common.retry}
              </Button>
            }
            description={b.history.readFailure}
            icon="warning"
            title={b.history.failedTitle}
          />
        ) : sessions.length === 0 ? (
          <PanelEmpty description={b.history.emptyDescription} icon="history" title={b.history.emptyTitle} />
        ) : (
          <div className="flex min-h-0 flex-col gap-2">
            <SearchField
              aria-label={b.history.search}
              containerClassName="w-full"
              inputClassName="w-full"
              onChange={setQuery}
              placeholder={b.history.searchPlaceholder}
              value={query}
            />
            <div className="max-h-[min(60vh,36rem)] overflow-y-auto overscroll-contain">
              <div className="grid gap-0.5">
                {filtered.map(session => {
                  const title = conversationTitle(session, b.history.untitled)
                  const age = historyAge(Number(session.last_active || session.started_at || 0), t.sidebar.row)
                  const opening = openingId === session.id

                  return (
                    <RowButton
                      aria-label={b.history.openConversation(title)}
                      className={cn(
                        'flex w-full min-w-0 items-start gap-2 rounded-md px-2 py-2.5 text-left transition-colors',
                        'hover:bg-(--chrome-action-hover)',
                        opening && 'pointer-events-none opacity-60'
                      )}
                      key={session.id}
                      onClick={() => void openConversation(session)}
                    >
                      <SidebarRowLead className="mt-0.5">
                        {opening ? (
                          <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
                        ) : (
                          <SessionStatusDot storedSessionId={session.id} />
                        )}
                      </SidebarRowLead>
                      <span className="min-w-0 flex-1">
                        <span className="flex min-w-0 items-baseline gap-2">
                          <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium">{title}</span>
                          {session.archived ? (
                            <span className="shrink-0 text-[0.625rem] uppercase tracking-wide text-(--ui-text-quaternary)">
                              {b.history.archived}
                            </span>
                          ) : null}
                          {age ? (
                            <span className="shrink-0 text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)">{age}</span>
                          ) : null}
                        </span>
                        {session.preview ? (
                          <span className="block truncate text-xs text-(--ui-text-tertiary)">{session.preview}</span>
                        ) : null}
                      </span>
                    </RowButton>
                  )
                })}
              </div>
            </div>
            {filtered.length === 0 ? (
              <div className="py-6 text-center text-xs text-(--ui-text-tertiary)">
                {b.history.noMatch(query.trim())}
              </div>
            ) : (
              <div className="text-right text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)">
                {b.history.count(filtered.length)}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
