import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { getSessionChildren } from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { cn } from '@/lib/utils'
import {
  buildHistoricalChildLanes,
  buildRunTimeline,
  type HistoricalChildInput,
  type ParentToolInput,
  type RunTimeline
} from '@/store/run-timeline'
import { $activeSessionId, $messages, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

import { Minimap } from './minimap'
import { TimelineCanvas } from './timeline-canvas'

type ObservatoryTab = 'canvas' | 'charts'

interface ObservatoryViewProps {
  onClose: () => void
}

const numOr = (v: unknown, fallback: number): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : fallback

/**
 * Extract the parent session's tool calls from chat message parts into the
 * timeline's minimal ParentToolInput shape. started_at rides in the tool-call
 * part's args (set in chat-messages.upsertToolPart); duration_s in result.
 * Exported for tests.
 */
export function parentToolInputs(messages: readonly ChatMessage[]): ParentToolInput[] {
  const inputs: ParentToolInput[] = []

  for (const message of messages) {
    for (const part of message.parts) {
      if (!part || typeof part !== 'object' || (part as { type?: string }).type !== 'tool-call') {
        continue
      }

      const p = part as {
        toolCallId?: string
        toolName?: string
        args?: Record<string, unknown>
        result?: Record<string, unknown>
      }

      const args = p.args ?? {}
      const result = p.result ?? {}
      const startedAt = typeof args.started_at === 'number' ? args.started_at : undefined
      const durationS = typeof result.duration_s === 'number' ? result.duration_s : undefined

      inputs.push({
        toolCallId: p.toolCallId || `${p.toolName || 'tool'}-${inputs.length}`,
        name: p.toolName || 'tool',
        startedAt,
        durationS
      })
    }
  }

  return inputs
}

function Tab({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      className={cn(
        'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
        active ? 'bg-(--ui-control-active-background) text-foreground' : 'text-muted-foreground hover:text-foreground'
      )}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}

/**
 * Pure charts summary from a timeline: total blocks, per-family counts,
 * outlier count. Exported for tests. (Charts tab is a stub in v1 — this is the
 * shared model the full pie/bar will use later.)
 */
export function timelineSummary(timeline: RunTimeline) {
  const byFamily: Record<string, number> = {}
  let blocks = 0
  let outliers = 0

  for (const lane of timeline.lanes) {
    for (const block of lane.blocks) {
      blocks += 1
      byFamily[block.family] = (byFamily[block.family] ?? 0) + 1

      if (block.isOutlier) {
        outliers += 1
      }
    }

    if (lane.bar?.isOutlier) {
      outliers += 1
    }
  }

  return { blocks, byFamily, outliers, lanes: timeline.lanes.length }
}

export function ObservatoryView({ onClose }: ObservatoryViewProps) {
  const [tab, setTab] = useState<ObservatoryTab>('canvas')
  const [seek, setSeek] = useState(0)

  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const messages = useStore($messages)
  const sessions = useStore($sessions)
  const subagentsBySession = useStore($subagentsBySession)

  const nowMs = Date.now()

  const sid = selectedStoredSessionId || activeSessionId || ''
  const session = sessions.find(s => s.id === sid)
  const liveSessionId = activeSessionId || sid
  const liveSubs: SubagentProgress[] = liveSessionId ? (subagentsBySession[liveSessionId] ?? []) : []
  const hasLiveChildren = liveSubs.length > 0

  // Historical child lanes come from persisted subagent sessions. Only fetched
  // when the LIVE store has no children for this session (finished run or fresh
  // app reload) — a running turn already has richer live data. Keyed by sid so
  // switching sessions refetches.
  const childrenQuery = useQuery({
    queryKey: ['observatory-children', sid],
    queryFn: () => getSessionChildren(sid, session?.profile),
    enabled: sid !== '' && !hasLiveChildren
  })

  const timeline = useMemo(() => {
    const createdAtMs = numOr(session?.started_at, nowMs / 1000) * 1000
    const tools = parentToolInputs(messages)
    // Finished-run clamp: once the session has ended, the axis + running-bar
    // fallback must stop at the real end, not stretch to wall-clock now.
    const endedAtMs = typeof session?.ended_at === 'number' ? session.ended_at * 1000 : null
    const effectiveNowMs = endedAtMs ?? nowMs

    const base = buildRunTimeline(liveSubs, tools, createdAtMs, sid || 'parent', session?.title || 'Current session', {
      nowMs: effectiveNowMs
    })

    // Exclusive-or by source (review blocker 3): use LIVE child lanes when the
    // live store is populated; otherwise splice in PERSISTED child lanes. Never
    // union — a live child plus its own persisted row would double-draw.
    if (hasLiveChildren) {
      return base
    }

    const persisted: HistoricalChildInput[] = (childrenQuery.data?.children ?? []).map(child => ({
      id: child.id,
      label: child.title || child.id,
      startedAtS: child.started_at,
      endedAtS: child.ended_at,
      toolCount: child.tool_call_count
    }))

    if (persisted.length === 0) {
      return base
    }

    const childLanes = buildHistoricalChildLanes(persisted, sid || 'parent', endedAtMs, { nowMs: effectiveNowMs })
    const lanes = [...base.lanes, ...childLanes]
    const startMs = lanes.reduce((min, l) => Math.min(min, l.startMs), base.startMs)
    const knownEnd = lanes.reduce((max, l) => Math.max(max, l.endMs ?? l.startMs), base.endMs)

    return { lanes, startMs, endMs: Math.max(knownEnd, effectiveNowMs) }
    // nowMs intentionally excluded: recomputing every ms is wasteful; store
    // changes (messages/subagents) + the children query already retrigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, messages, session, liveSubs, hasLiveChildren, childrenQuery.data])

  const hasData = timeline.lanes.some(l => l.blocks.length > 0 || l.bar)
  const summary = timelineSummary(timeline)
  const isHistorical = !hasLiveChildren && (childrenQuery.data?.children.length ?? 0) > 0

  return (
    <Panel onClose={onClose}>
      <PanelHeader
        actions={
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 rounded-lg border border-(--ui-stroke-tertiary) p-0.5">
              <Tab active={tab === 'canvas'} label="Canvas" onClick={() => setTab('canvas')} />
              <Tab active={tab === 'charts'} label="Charts" onClick={() => setTab('charts')} />
            </div>
            <Button className="gap-1.5" onClick={onClose} size="sm" variant="ghost">
              <Codicon name="close" size="0.875rem" />
              Close
            </Button>
          </div>
        }
        subtitle={`${summary.lanes} lanes · ${summary.blocks} tool blocks · ${summary.outliers} outliers${isHistorical ? ' · historical' : ''}`}
        title="Observatory"
      />

      {!hasData ? (
        <PanelEmpty
          description="Run some tools or delegate a subagent, then reopen to see the timeline."
          icon="dashboard"
          title="No run activity yet"
        />
      ) : tab === 'canvas' ? (
        <div className="flex min-h-0 flex-1 flex-col gap-2">
          <TimelineCanvas nowMs={nowMs} timeline={timeline} />
          <Minimap
            nowMs={nowMs}
            onSeek={setSeek}
            timeline={timeline}
            windowEnd={Math.min(1, seek + 0.25)}
            windowStart={seek}
          />
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto p-2 text-sm text-muted-foreground">
          <p className="mb-2">Aggregate charts land in a follow-up. Current run rollup:</p>
          <ul className="space-y-1">
            <li>Total tool blocks: {summary.blocks}</li>
            <li>Lanes: {summary.lanes}</li>
            <li>Outliers: {summary.outliers}</li>
            {Object.entries(summary.byFamily).map(([family, count]) => (
              <li key={family}>
                {family}: {count}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}
