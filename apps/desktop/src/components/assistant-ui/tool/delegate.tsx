'use client'

import { useStore } from '@nanostores/react'
import { type FC, type ReactNode, useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { SCAFFOLD_LABEL_CLASS, SCAFFOLD_META_CLASS } from '@/components/chat/scaffold-row'
import { FadeText } from '@/components/ui/fade-text'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { AlertCircle } from '@/lib/icons'
import { displayModelName } from '@/lib/model-status-label'
import { useSessionSlice } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $subagentsBySession } from '@/store/subagents'
import { openSessionInNewWindow } from '@/store/windows'

import {
  type DelegateRow,
  delegateRowsFromCall,
  delegateRowTone,
  type DelegateRowTone,
  mergeDelegateRows
} from './delegate-model'
import { formatDurationSeconds, type ToolPart } from './fallback-model'
import { ToolRunTicker } from './run-ticker'

// Activity lines kept mounted behind the visible one. Enough for the reel to
// read as motion, few enough that a chatty child doesn't hold a hundred rows
// in the DOM per subagent.
const TICKER_DEPTH = 6

function statusGlyph(tone: DelegateRowTone, label: string): ReactNode {
  if (tone === 'live') {
    return (
      <GlyphSpinner ariaLabel={label} className="size-3.5 text-[0.95rem] text-(--ui-text-tertiary)" spinner="breathe" />
    )
  }

  if (tone === 'failed') {
    return <AlertCircle aria-label={label} className="size-3.5 text-destructive" />
  }

  if (tone === 'parked') {
    // Parked, not watched: the children outlived the turn that spawned them
    // and nothing in this transcript is streaming their progress. A spinner
    // here would claim a liveness we can't back up.
    return <span aria-hidden className="size-1.5 rounded-full bg-(--ui-text-tertiary)" />
  }

  // Outstanding, not done. A green check here would read the child's lifecycle
  // as task acceptance — the one thing the envelope never says. Amber is the
  // same claim the Spawn-tree panel makes about the same child.
  return <AlertCircle aria-label={label} className="size-3.5 text-amber-500/85" />
}

/**
 * One delegated child: who it is on the first line, what it is doing on the
 * second.
 *
 * The title carries the goal and the model running it — the two things that
 * identify a child you didn't dispatch yourself — with the elapsed time
 * trailing while it works. Underneath, a single ticking line of its relayed
 * activity, so a fan-out of five children costs ten lines of transcript
 * whatever they get up to.
 */
function DelegateRowView({ row }: { row: DelegateRow }) {
  const { t } = useI18n()
  const copy = t.assistant.tool
  const { sessionId } = row
  const tone = delegateRowTone(row)
  const live = tone === 'live'
  const elapsed = useElapsedSeconds(live, `delegate:${row.id}`)
  const activity = row.activity.slice(-TICKER_DEPTH)

  const statusLabel: Record<DelegateRowTone, string> = {
    failed: copy.statusError,
    live: copy.statusRunning,
    parked: copy.statusDone,
    partial: copy.statusPartial,
    unverified: copy.statusUnverified
  }

  const meta = [
    row.model ? displayModelName(row.model) : '',
    !live && row.durationSeconds ? formatDurationSeconds(row.durationSeconds) : ''
  ].filter(Boolean)

  // Only a child that reported its own session id has somewhere to go.
  const open = sessionId ? () => void openSessionInNewWindow(sessionId, { watch: true }) : undefined

  return (
    <div className="grid min-w-0 max-w-full gap-0.5" data-conversation-scaffold="">
      <div className="flex min-w-0 max-w-full items-center gap-1.5">
        <span className="grid size-3.5 shrink-0 place-items-center">{statusGlyph(tone, statusLabel[tone])}</span>
        <button
          className={cn(
            SCAFFOLD_LABEL_CLASS,
            'min-w-0 truncate text-left transition-colors',
            open ? 'hover:text-foreground focus-visible:text-foreground focus-visible:outline-none' : 'cursor-default'
          )}
          disabled={!open}
          onClick={open}
          type="button"
        >
          {row.goal}
        </button>
        {meta.length > 0 && <span className={SCAFFOLD_META_CLASS}>{meta.join(' · ')}</span>}
        {live && <ActivityTimerText className={cn(SCAFFOLD_META_CLASS, 'ml-auto')} seconds={elapsed} />}
      </div>
      {activity.length > 0 && (
        <div className="min-w-0 max-w-full pl-5">
          <ToolRunTicker>
            {activity.map((text, index) => (
              <FadeText
                className={cn(SCAFFOLD_LABEL_CLASS, 'text-(--conversation-scaffold-meta)', live && 'shimmer')}
                key={`${row.id}:${index}`}
              >
                {text}
              </FadeText>
            ))}
          </ToolRunTicker>
        </div>
      )}
    </div>
  )
}

/**
 * A `delegate_task` call, as the fan-out it is.
 *
 * The generic tool row can only say "Delegated 2 tasks" and hand over a blob
 * of JSON — the work itself happens in child sessions the transcript never
 * sees. This lists those children instead, joining what the call dispatched to
 * what the subagent store knows about them, so a delegation reads like the
 * several agents it actually is.
 *
 * A card, never folded into a run summary: the point of the block is the live
 * list, and a ticker cycling one line across five children would show four of
 * them nothing.
 */
export const DelegateTool: FC<Pick<ToolPart, 'args' | 'result' | 'toolCallId'>> = ({ args, result, toolCallId }) => {
  const sessionId = useStore(useSessionView().$runtimeId)
  const live = useSessionSlice($subagentsBySession, sessionId)

  const rows = useMemo(
    () => mergeDelegateRows(delegateRowsFromCall(args, result, toolCallId), live, toolCallId),
    [args, live, result, toolCallId]
  )

  if (rows.length === 0) {
    return null
  }

  return (
    <div className="grid min-w-0 gap-(--tool-row-gap)" data-delegate-card="" data-slot="tool-block">
      {rows.map(row => (
        <DelegateRowView key={row.id} row={row} />
      ))}
    </div>
  )
}
