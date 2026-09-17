import { useState } from 'react'

import { SubagentRow } from '@/app/agents'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useViewedInterval } from '@/hooks/use-viewed-interval'
import { useI18n } from '@/i18n'
import { AlertCircle, CheckCircle2 } from '@/lib/icons'
import { useSessionSlice } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $subagentsBySession, dismissSubagent, type SubagentProgress } from '@/store/subagents'

import { SubagentControls } from './subagent-controls'
import { SubagentTranscript } from './subagent-transcript'

interface SubagentSectionProps {
  sessionId: string
}

/** A composer-local roster: never borrow the global Agents panel's scope. */
export function SubagentSection({ sessionId }: SubagentSectionProps) {
  const { t } = useI18n()
  const items = useSessionSlice($subagentsBySession, sessionId)
  const live = items.filter(item => item.status === 'running' || item.status === 'queued')
  const [nowMs, setNowMs] = useState(Date.now)
  const [selected, setSelected] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const hasLive = live.length > 0
  const hasItems = items.length > 0

  useViewedInterval(() => setNowMs(Date.now()), 1000, hasLive)

  if (!hasItems) {
    return null
  }

  const row = (item: SubagentProgress) => {
    const running = item.status === 'running' || item.status === 'queued'
    const failed = item.status === 'failed' || item.status === 'interrupted'

    return (
      <StatusRow
        dismiss={
          running ? undefined : { label: t.statusStack.dismiss, onDismiss: () => dismissSubagent(sessionId, item.id) }
        }
        expanded={running && selected === item.id}
        key={item.id}
        leading={
          running ? (
            <GlyphSpinner
              ariaLabel={item.status === 'queued' ? t.agents.queued : t.agents.running}
              className="text-(--ui-purple)"
              spinner="braille"
            />
          ) : item.status === 'completed' ? (
            <CheckCircle2
              aria-label={t.agents.done}
              className="size-3.5 text-emerald-600/85 dark:text-emerald-400/85"
            />
          ) : (
            <AlertCircle aria-label={t.agents.failed} className="size-3.5 text-destructive" />
          )
        }
        onActivate={running ? () => setSelected(selected === item.id ? null : item.id) : undefined}
        trailing={
          running ? (
            <ActivityTimerText
              className="shrink-0 text-[0.65rem]"
              seconds={Math.max(0, Math.floor((nowMs - item.startedAt) / 1000))}
            />
          ) : undefined
        }
        trailingVisible
      >
        <span className="min-w-0 flex-1">
          <span className={cn('block truncate text-xs', failed ? 'text-destructive/90' : 'text-(--ui-text-primary)')}>
            {item.goal}
          </span>
          <span className="block truncate text-[0.68rem] text-(--ui-text-tertiary)">
            {item.stream.at(-1)?.text ||
              (item.status === 'queued'
                ? t.agents.queued
                : running
                  ? t.agents.waitingActivity
                  : item.status === 'completed'
                    ? t.agents.done
                    : t.agents.failed)}
          </span>
        </span>
      </StatusRow>
    )
  }

  const detail = live.find(item => item.id === selected)

  return (
    <div className="composer-no-drag min-w-0" data-slot="composer-subagents">
      <StatusSection
        collapsedIndicator={
          hasLive ? (
            <GlyphSpinner
              ariaLabel={live.some(item => item.status === 'running') ? t.agents.running : t.agents.queued}
              className="text-(--ui-purple)"
              spinner="braille"
            />
          ) : undefined
        }
        icon={<Codicon className="text-(--ui-purple)" name="agent" size="0.8rem" />}
        label={t.statusStack.subagents(items.length)}
      >
        <div className="max-h-[25vh] overflow-y-auto overscroll-y-auto">{items.map(row)}</div>
        {detail && (
          <div
            className="status-subagent-detail max-h-[25vh] overflow-y-auto overscroll-y-auto pr-3 py-2"
            data-slot="composer-subagent-detail"
          >
            <SubagentControls
              key={`${sessionId}:${detail.id}`}
              sessionId={sessionId}
              setText={text => setDrafts(previous => ({ ...previous, [detail.id]: text }))}
              subagentId={detail.id}
              text={drafts[detail.id] ?? ''}
            />
            <SubagentRow node={{ ...detail, children: [] }} nowMs={nowMs} />
            <SubagentTranscript key={`tail:${sessionId}:${detail.id}`} sessionId={sessionId} subagentId={detail.id} />
          </div>
        )}
      </StatusSection>
    </div>
  )
}
