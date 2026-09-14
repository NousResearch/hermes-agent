/**
 * The per-bot session browser as it renders inside the Bots rail: a
 * collapsible block directly under the bot's row listing every stored chat in
 * that bot's profile. Opened from the row's context menu ("Show sessions"),
 * one bot at a time.
 *
 * Read-only over `session.list`; opening a row goes through the same
 * workspace-aware open the canonical chat uses. Nothing here is persisted.
 */

import {
  cn,
  coarseElapsed,
  Codicon,
  GlyphSpinner,
  host,
  RowButton,
  SessionStatusDot,
  SidebarRowLead,
  Tip,
  useI18n,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'

import {
  $botSessionsOpen,
  closeBotSessions,
  isBotSessionsOpen,
  isCanonicalRow,
  listBotSessions,
  openBotSession,
  sessionActivity
} from './bot-sessions'
import type { BotSessionRow } from './bot-sessions'
import { $botMeta, botRosterKey } from './data'
import { useBots } from './i18n'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { RosterRow, SidebarRowLabels } from './types'

function rowAge(seconds: number, r: SidebarRowLabels): string {
  const { unit, value } = coarseElapsed(Date.now() - seconds * 1000)

  return unit === 'second' ? r.ageNow : `${value}${unit === 'day' ? r.ageDay : unit === 'hour' ? r.ageHour : r.ageMin}`
}

export const BOT_SESSIONS_QUERY_KEY = 'hermes-bots.sessions'

interface BotSessionsPanelProps {
  bot: RosterRow
}

/** Renders nothing unless this bot is the one expanded. Mounted under every
 *  row so the roster's virtualization/sections need no extra plumbing; the
 *  cost of a closed panel is one atom read. */
export function BotSessionsPanel({ bot }: BotSessionsPanelProps) {
  const open = useValue($botSessionsOpen)

  if (!isBotSessionsOpen(bot, open)) {
    return null
  }

  return <BotSessionsList bot={bot} />
}

function BotSessionsList({ bot }: BotSessionsPanelProps) {
  const { t } = useI18n()
  const b = useBots()
  const meta = botRosterMeta(bot, useValue($botMeta))
  const focused = useValue(host.state.focusedStoredSessionId)

  const { data, error, isLoading, refetch } = useQuery({
    queryKey: [BOT_SESSIONS_QUERY_KEY, botRosterKey(bot)],
    queryFn: () => listBotSessions(bot),
    refetchInterval: 15_000
  })

  const rows = data ?? []

  const openRow = (row: BotSessionRow) =>
    void openBotSession(bot, row).catch(err => host.notifyError?.(err, b.sessions.openFailed))

  return (
    <div
      aria-label={b.sessions.ariaLabel(displayName(bot, meta))}
      className="mb-1 ml-3 grid gap-0.5 border-l border-(--ui-stroke-tertiary) pl-1.5"
      data-slot="bot-sessions"
      role="region"
    >
      <div className="flex items-center gap-1 px-2 py-1 text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)">
        <Codicon className="shrink-0" name="history" />
        <span className="min-w-0 flex-1 truncate">{b.sessions.heading}</span>
        {rows.length ? <span className="shrink-0 font-normal tabular-nums">{rows.length}</span> : null}
        <Tip label={b.sessions.close}>
          <button
            aria-label={b.sessions.close}
            className="flex size-5 shrink-0 items-center justify-center rounded text-(--ui-text-quaternary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-secondary)"
            onClick={closeBotSessions}
            type="button"
          >
            <Codicon name="close" />
          </button>
        </Tip>
      </div>
      {isLoading && !rows.length ? (
        <div className="flex justify-center py-2">
          <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
        </div>
      ) : error && !rows.length ? (
        <div className="grid gap-1.5 px-2 py-1.5 text-xs text-(--ui-text-tertiary)">
          <span>{b.sessions.loadFailed(error instanceof Error ? error.message : 'gateway error')}</span>
          <button
            className="justify-self-start text-(--ui-accent) hover:underline"
            onClick={() => void refetch()}
            type="button"
          >
            {b.roster.retryNow}
          </button>
        </div>
      ) : rows.length === 0 ? (
        <div className="px-2 py-1.5 text-xs text-(--ui-text-tertiary)">{b.sessions.empty}</div>
      ) : (
        rows.map(row => {
          const canonical = isCanonicalRow(row)
          const storedId = String(row.resolved_id || row.id)
          const active = focused != null && String(focused) === storedId
          const title = canonical ? displayName(bot, meta) : String(row.title || '').trim() || b.sessions.untitled
          const age = sessionActivity(row)

          return (
            <RowButton
              aria-current={active ? 'true' : undefined}
              className={cn(
                'flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-(--chrome-action-hover)',
                active && 'bg-(--ui-row-active-background)'
              )}
              data-session-id={row.id}
              key={row.id}
              onClick={() => openRow(row)}
            >
              <SidebarRowLead>
                <SessionStatusDot storedSessionId={row.id} />
              </SidebarRowLead>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-1 text-[0.8125rem]">
                    {canonical ? (
                      <Tip label={b.sessions.canonicalTip}>
                        <Codicon className="shrink-0 text-[0.6875rem] text-(--ui-accent)" name="comment-discussion" />
                      </Tip>
                    ) : null}
                    <span className="min-w-0 truncate">{title}</span>
                  </span>
                  {age ? (
                    <span className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)">
                      {rowAge(age, t.sidebar.row)}
                    </span>
                  ) : null}
                </div>
                {row.preview ? (
                  <div className="min-w-0 truncate text-xs text-(--ui-text-tertiary)">{row.preview}</div>
                ) : null}
              </div>
            </RowButton>
          )
        })
      )}
    </div>
  )
}
