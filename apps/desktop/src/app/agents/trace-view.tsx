import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import { $traceSelection } from '@/store/trace'

import { PanelEmpty } from '../overlays/panel'

import type { TraceLabels } from './fold-relay-events'
import { fmtDuration } from './format'
import { useTraceView } from './hooks/use-trace-view'
import { SpanInspector } from './span-inspector'
import { ROW_HEIGHT, TraceWaterfall } from './trace-waterfall'
import { TurnStrip } from './turn-strip'

/** The active session's execution waterfall, folded from the Relay events Hermes recorded. */
export function TraceView() {
  const { t } = useI18n()
  const a = t.agents

  const labels = useMemo<TraceLabels>(
    () => ({ llmCall: a.spanLlmCall, session: a.spanSession, subagent: a.spanSubagent, turn: a.spanTurn }),
    [a]
  )

  const {
    activeIndex,
    error,
    liveIndex,
    loading,
    recording,
    selectTurn,
    selection,
    sessionId,
    trace,
    truncated,
    turns
  } = useTraceView(labels)

  const hasTrace = !!trace && trace.spans.length > 0

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-2 flex h-7 shrink-0 items-center justify-between gap-3">
        <p className="truncate text-xs text-muted-foreground/80">
          {!sessionId
            ? a.traceNoSession
            : hasTrace
              ? a.traceSubtitle(trace.spans.length, fmtDuration(trace.duration))
              : ''}
          {hasTrace && truncated ? ` · ${a.traceTruncated}` : ''}
        </p>
        <TurnStrip
          activeIndex={activeIndex}
          allActive={selection === 'all'}
          liveIndex={liveIndex}
          onAll={() => $traceSelection.set('all')}
          onTurn={selectTurn}
          turns={turns}
        />
      </div>

      {hasTrace ? (
        <div className="flex min-h-0 flex-1 gap-3 overflow-hidden">
          <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
            <TraceWaterfall trace={trace} viewKey={`${sessionId ?? ''}:${selection}`} />
          </div>
          <div className="flex w-72 shrink-0 flex-col overflow-y-auto" style={{ paddingTop: ROW_HEIGHT }}>
            <SpanInspector trace={trace} />
          </div>
        </div>
      ) : error ? (
        <PanelEmpty description={error} icon="warning" />
      ) : loading ? (
        <PanelEmpty icon="loading~spin" title={a.traceLoading} />
      ) : !recording ? (
        <PanelEmpty description={a.traceOffDesc} icon="debug-pause" title={a.traceOffTitle} />
      ) : (
        <PanelEmpty description={a.traceEmptyDesc} icon="pulse" title={a.traceEmptyTitle} />
      )}
    </div>
  )
}
