import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { $visibleLiveSessions } from '@/store/live-sessions'
import { $profileScope } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { SessionStatusDot } from '../session-status-dot'

import { SidebarRowBody, SidebarRowLabel, SidebarRowLead, SidebarRowShell } from './chrome'
import { filterSessionsByProfileScope } from './profile-scope'

// The rows refresh their relative ages on this cadence. The `session.active_list`
// poll re-runs every 1.5s but preserves the atom reference when nothing changed,
// so without its own clock a row would freeze at "5 min ago" during an idle
// minute (same trade the cron section makes with its peek clock).
const AGE_TICK_MS = 30_000

interface SidebarLiveSessionsSectionProps {
  label: string
  // The live row's only action is opening its session — the same door a stored
  // row uses. `session.resume` reattaches a live session with no DB row (the
  // gateway matches the stored key against its in-memory registry and attaches
  // this viewer alongside the creating client, never displacing it), so the
  // stored id on the row is a real handle, not a promise.
  onResumeSession: (sessionId: string, session?: SessionInfo) => void
}

/**
 * The sidebar's live group (#50799): sessions LIVE in the gateway process that
 * no stored-list slice can show yet — created over the gateway by another
 * client (TUI, CLI, a second window), invisible to the DB-backed sidebar until
 * the first prompt persists a row. Fed by the `session.active_list` poll
 * through `reconcileLiveSessions` (store/live-sessions); promotion out of this
 * group into Recents is automatic the moment the stored row lands, because the
 * reconciler dedupes against every slice.
 *
 * Deliberately NOT `SidebarSessionRow`: pin/archive/delete/unread all act on a
 * persisted row that does not exist yet, and rendering no-op menu items for
 * them would be a UI that lies. The row offers exactly one action — open.
 */
export function SidebarLiveSessionsSection({ label, onResumeSession }: SidebarLiveSessionsSectionProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const liveSessions = useStore($visibleLiveSessions)
  const profileScope = useStore($profileScope)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const visible = usePaneVisible()
  const [nowMs, setNowMs] = useState(() => Date.now())

  // The same scope rule the stored slices obey (index.tsx filters recents, cron
  // and messaging through `filterSessionsByProfileScope`): a session live on
  // this backend under a DIFFERENT profile must not surface while the sidebar
  // is scoped to one profile. `ALL_PROFILES` passes the array through
  // untouched, so the identity-stable common case stays memo-friendly.
  const rows = useMemo(() => filterSessionsByProfileScope(liveSessions, profileScope), [liveSessions, profileScope])

  // Rows are pure; one clock for the section, ticking only while the pane is
  // on screen — same shape as the cron section's countdown clock.
  useEffect(() => {
    if (!visible) {
      return
    }

    const id = window.setInterval(() => setNowMs(Date.now()), AGE_TICK_MS)

    return () => window.clearInterval(id)
  }, [visible])

  if (rows.length === 0) {
    return null
  }

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="flex shrink-0 items-center pb-1 pt-1.5">
        <SidebarPanelLabel>{label}</SidebarPanelLabel>
      </div>
      <SidebarGroupContent className="scrollbar-fade flex max-h-56 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75 compact:max-h-none compact:overflow-visible">
        {rows.map(session => (
          <LiveSessionSidebarRow
            isSelected={session.id === selectedStoredSessionId}
            key={session.id}
            nowMs={nowMs}
            onResume={() => onResumeSession(session.id, session)}
            session={session}
            untitledLabel={r.untitledPlaceholder}
          />
        ))}
      </SidebarGroupContent>
    </SidebarGroup>
  )
}

function LiveSessionSidebarRow({
  isSelected,
  nowMs,
  onResume,
  session,
  untitledLabel
}: {
  isSelected: boolean
  nowMs: number
  onResume: () => void
  session: SessionInfo
  untitledLabel: string
}) {
  const title = session.title?.trim() || untitledLabel
  const preview = session.preview?.trim()
  const lastActiveMs = (session.last_active || session.started_at || 0) * 1000

  return (
    <SidebarRowShell
      actions={
        lastActiveMs > 0 ? (
          <span className="text-[0.6875rem] text-(--ui-text-tertiary) tabular-nums">{relativeTime(lastActiveMs, nowMs)}</span>
        ) : null
      }
      className={cn('group/live relative hover:bg-(--chrome-action-hover)', isSelected && 'bg-(--ui-row-active-background)')}
    >
      {/* The shared status dot, keyed by the STORED id the poll bound into the
          live-state mirror (rehydrateLiveSessionStatuses publishes busy/needs
          input for exactly these runtimes). A running foreign session paints
          the same working/amber dot its stored sibling would — the live/unsaved
          state is carried by the section itself, so no new color is invented. */}
      <Tip label={title}>
        <SidebarRowBody onClick={onResume}>
          <SidebarRowLead>
            <SessionStatusDot session={session} storedSessionId={session.id} />
          </SidebarRowLead>
          <span className="flex min-w-0 flex-1 flex-col justify-center">
            <SidebarRowLabel className={cn(isSelected && 'text-foreground')}>{title}</SidebarRowLabel>
            {preview ? (
              <span className="min-w-0 truncate text-[0.6875rem] leading-[1.3] text-(--ui-text-quaternary)">
                {preview}
              </span>
            ) : null}
          </span>
        </SidebarRowBody>
      </Tip>
    </SidebarRowShell>
  )
}
