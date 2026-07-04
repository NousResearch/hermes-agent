import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Kanban, Loader2, MessageQuestion } from '@/lib/icons'
import { profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import {
  $kanbanBoard,
  $kanbanBoardError,
  refreshKanbanBoard,
  refreshKanbanBoards,
  setKanbanSelectedTaskId
} from '@/store/kanban'
import {
  $activeGatewayProfile,
  $profileColors,
  $profileOrder,
  $profiles,
  normalizeProfileKey,
  refreshProfiles,
  selectProfile
} from '@/store/profile'
import { $attentionSessionIds, $sessionProfileTotals, $sessions, $workingSessionIds } from '@/store/session'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { KANBAN_ROUTE } from '../routes'

import { buildProfileOverviewRows, summarizeKanbanBoard } from './overview-data'

interface OverviewPanelProps {
  onClose: () => void
  onNavigateRoute?: (path: string) => void
}

// Chip tones ride the ui-kit Badge variants (destructive/warn) so they stay in
// sync with every other status badge; only the hover shade is local.
const HOT_CHIP: Record<string, { hover: string; variant: 'destructive' | 'warn' }> = {
  blocked: { hover: 'hover:bg-destructive/20', variant: 'destructive' },
  review: { hover: 'hover:bg-amber-500/20', variant: 'warn' }
}

/** Cross-profile landing section: one activity row per profile (working /
 *  needs-input counts, last activity, quick switch) plus a summary strip for
 *  the shared kanban board. All session data folds from the already-fetched
 *  cross-profile aggregation — no extra sockets, kanban via its REST poller. */
export function OverviewPanel({ onClose, onNavigateRoute }: OverviewPanelProps) {
  const { t } = useI18n()
  const ov = t.commandCenterOverview
  const k = t.kanban

  const sessions = useStore($sessions)
  const profiles = useStore($profiles)
  const profileOrder = useStore($profileOrder)
  const profileColors = useStore($profileColors)
  const workingIds = useStore($workingSessionIds)
  const attentionIds = useStore($attentionSessionIds)
  const profileTotals = useStore($sessionProfileTotals)
  const activeGatewayProfile = useStore($activeGatewayProfile)

  const board = useStore($kanbanBoard)
  const boardError = useStore($kanbanBoardError)

  const refresh = () => {
    void refreshProfiles().catch(() => undefined)
    void refreshKanbanBoards()
    void refreshKanbanBoard()
  }

  // Refresh on mount: the session aggregation is already live in the stores,
  // so only the profile list and the kanban board need a poke.
  useEffect(refresh, [])

  useRefreshHotkey(refresh)

  const rows = useMemo(
    () =>
      buildProfileOverviewRows({
        sessions,
        profiles,
        profileOrder,
        workingIds,
        attentionIds,
        profileTotals
      }),
    [attentionIds, profileOrder, profileTotals, profiles, sessions, workingIds]
  )

  const summary = useMemo(() => (board ? summarizeKanbanBoard(board) : null), [board])

  const activeKey = normalizeProfileKey(activeGatewayProfile)

  // Older backends without the kanban plugin 404 the board route; the overview
  // stays useful as a profiles-only view instead of erroring the whole panel.
  const kanbanUnavailable = !board && Boolean(boardError)

  const switchToProfile = (key: string) => {
    selectProfile(key)
    onClose()
  }

  const openKanban = (taskId?: string) => {
    setKanbanSelectedTaskId(taskId ?? null)
    onNavigateRoute?.(KANBAN_ROUTE)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto pb-2">
      <section>
        <div className="mb-1.5 text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
          {ov.sectionLabel}
        </div>
        <ul>
          {rows.map(row => {
            const color = resolveProfileColor(row.isDefault ? null : row.key, profileColors)
            const active = row.key === activeKey
            const busy = row.workingCount > 0 || row.attentionCount > 0

            return (
              <li className="flex items-center gap-3 border-b border-(--ui-stroke-tertiary)/50 py-2.5" key={row.key}>
                <span
                  aria-hidden
                  className="size-2.5 shrink-0 rounded-full"
                  style={{
                    backgroundColor: color ?? 'var(--ui-text-tertiary)',
                    boxShadow: color ? `0 0 0 3px ${profileColorSoft(color, 18)}` : undefined
                  }}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
                      {row.key}
                    </span>
                    {active && (
                      <span className="shrink-0 rounded bg-(--chrome-action-hover) px-1.5 py-px text-[0.6rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
                        {ov.currentBadge}
                      </span>
                    )}
                  </div>
                  <div className="truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {ov.sessions(row.sessionCount)}
                    {' · '}
                    {row.lastActive ? relativeTime(row.lastActive * 1000) : ov.noActivity}
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-3 text-[length:var(--conversation-caption-font-size)]">
                  {row.workingCount > 0 && (
                    <span className="inline-flex items-center gap-1 text-(--dt-primary)">
                      <Loader2 aria-hidden className="size-3 animate-spin" />
                      {ov.working(row.workingCount)}
                    </span>
                  )}
                  {row.attentionCount > 0 && (
                    <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-300">
                      <MessageQuestion aria-hidden className="size-3.5" />
                      {ov.attention(row.attentionCount)}
                    </span>
                  )}
                  {!busy && <span className="text-(--ui-text-tertiary)">{ov.idle}</span>}
                </div>

                {!active && (
                  <Button
                    aria-label={t.profiles.switchToProfile(row.key)}
                    className="shrink-0"
                    onClick={() => switchToProfile(row.key)}
                    size="xs"
                    title={t.profiles.switchToProfile(row.key)}
                    type="button"
                    variant="text"
                  >
                    {ov.switch}
                  </Button>
                )}
              </li>
            )
          })}
        </ul>
      </section>

      <section>
        <div className="mb-1.5 flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-1.5 text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
            <Kanban aria-hidden className="size-3.5" />
            {k.title}
          </span>
          {!kanbanUnavailable && (
            <Button onClick={() => openKanban()} size="xs" type="button" variant="text">
              {ov.openBoard}
            </Button>
          )}
        </div>

        {kanbanUnavailable ? (
          <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {ov.kanbanUnavailable}
          </div>
        ) : !summary ? (
          <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {k.loading}
          </div>
        ) : (
          <>
            <div className="flex flex-wrap gap-x-4 gap-y-1.5">
              {summary.counts.map(entry => (
                <span
                  className={cn(
                    'inline-flex items-baseline gap-1.5 text-[length:var(--conversation-caption-font-size)]',
                    entry.count === 0 ? 'text-(--ui-text-tertiary)/60' : 'text-foreground'
                  )}
                  key={entry.name}
                >
                  <span className="font-semibold tabular-nums">{entry.count}</span>
                  <span className="text-(--ui-text-tertiary)">{k.columnLabels[entry.name] ?? entry.name}</span>
                </span>
              ))}
            </div>

            <div className="mt-3 flex flex-wrap gap-1.5">
              {summary.hotCards.length === 0 ? (
                <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {ov.noHotCards}
                </span>
              ) : (
                summary.hotCards.map(card => {
                  const tone = HOT_CHIP[card.status]

                  return (
                    <Badge asChild key={card.id} variant={tone?.variant}>
                      <button
                        aria-label={ov.openCard(card.title)}
                        className={cn(
                          'max-w-64 gap-1.5 rounded-full px-2.5 py-1 text-[length:var(--conversation-caption-font-size)] transition-colors',
                          tone ? tone.hover : 'bg-(--chrome-action-hover) text-foreground'
                        )}
                        onClick={() => openKanban(card.id)}
                        title={card.title}
                        type="button"
                      >
                        <span className="shrink-0 text-[0.6rem] font-medium uppercase tracking-[0.06em]">
                          {k.columnLabels[card.status] ?? card.status}
                        </span>
                        <span className="truncate">{card.title}</span>
                      </button>
                    </Badge>
                  )
                })
              )}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
