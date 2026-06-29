import { $traceSelection } from '@/store/trace'

import { OverlayView } from '../overlays/overlay-view'

import { EmptyState } from './empty-state'
import { fmtDuration } from './format'
import { useTraceView } from './hooks/use-trace-view'
import { SpanInspector } from './span-inspector'
import { ROW_HEIGHT, TraceWaterfall } from './trace-waterfall'
import { TurnStrip } from './turn-strip'

interface AgentsViewProps {
  onClose: () => void
}

export function AgentsView({ onClose }: AgentsViewProps) {
  const { activeIndex, error, liveIndex, loading, selectTurn, selection, sessionId, trace } = useTraceView()
  const hasTrace = !!trace && trace.spans.length > 0

  return (
    <OverlayView
      closeLabel="Close"
      contentClassName="flex h-full flex-col px-4 py-4 sm:px-5"
      onClose={onClose}
      rootClassName="mx-auto flex h-full w-full max-w-6xl flex-col"
    >
      <header className="mb-2 flex shrink-0 items-center justify-between gap-3 pl-2">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">Trace</h2>
          <p className="truncate text-xs text-muted-foreground/80">
            {sessionId ? `Execution waterfall · ${sessionId.slice(0, 16)}` : 'No active session'}
            {hasTrace ? ` · ${trace.spans.length} spans · ${fmtDuration(trace.duration)}` : ''}
          </p>
        </div>
        <TurnStrip
          activeIndex={activeIndex}
          allActive={selection === 'all'}
          liveIndex={liveIndex}
          onAll={() => $traceSelection.set('all')}
          onTurn={selectTurn}
        />
      </header>

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
        <EmptyState icon="warning" text={error} />
      ) : (
        <EmptyState icon={loading ? 'loading~spin' : 'hubot'} text={loading ? 'Loading trace…' : 'No trace yet'} />
      )}
    </OverlayView>
  )
}
