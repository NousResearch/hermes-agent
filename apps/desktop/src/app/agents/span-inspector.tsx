import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { $hoveredSpanId, $selectedSpanId, type TraceDoc } from '@/store/trace'

import { fmtDuration } from './format'
import { ROW_HEIGHT } from './trace-waterfall'

const fmtInt = (n: number) => n.toLocaleString()

export function SpanInspector({ trace }: { trace: null | TraceDoc }) {
  const selectedId = useStore($selectedSpanId)
  const hoveredId = useStore($hoveredSpanId)
  // Hover previews; the clicked span stays pinned when nothing is hovered.
  const activeId = hoveredId ?? selectedId

  const span = useMemo(() => trace?.spans.find(s => s.id === activeId) ?? null, [trace, activeId])

  if (!span) {
    return (
      <div className="flex items-center text-[0.7rem] text-muted-foreground/55" style={{ height: ROW_HEIGHT }}>
        Select a span to inspect its details.
      </div>
    )
  }

  const attrs = span.attributes
  const num = (key: string) => (typeof attrs[key] === 'number' ? (attrs[key] as number) : undefined)

  const meta: [string, string][] = [['kind', span.kind], ['status', span.status]]

  // Where the span sits in the trace, then how long it ran.
  if (trace) {
    meta.push(['started', `+${fmtDuration(Math.max(0, span.start - trace.start))}`])
  }

  meta.push(['duration', fmtDuration(span.duration)])

  // Push an attribute row when present; numbers are thousands-formatted.
  const push = (label: string, key: string) => {
    const v = attrs[key]

    if (v !== undefined && v !== null && v !== '') {
      meta.push([label, typeof v === 'number' ? fmtInt(v) : String(v)])
    }
  }

  push('model', 'llm.model_name')
  push('tokens in', 'llm.token_count.prompt')
  push('tokens out', 'llm.token_count.completion')

  const treason = num('llm.token_count.reasoning')

  if (treason) {
    meta.push(['reasoning', fmtInt(treason)])
  }

  const tin = num('llm.token_count.prompt')
  const tout = num('llm.token_count.completion')

  if (tin !== undefined || tout !== undefined) {
    meta.push(['tokens total', fmtInt((tin ?? 0) + (tout ?? 0) + (treason ?? 0))])
  }

  push('finish', 'hermes.finish_reason')
  push('tool', 'tool.name')
  // Container (AGENT) spans carry session shape; subagents expose their id.
  push('source', 'session.source')
  push('messages', 'session.message_count')
  push('tool calls', 'session.tool_call_count')

  if (span.sessionId) {
    meta.push(['session', span.sessionId.slice(0, 12)])
  }

  const input = attrs['input.value']
  const output = attrs['output.value']

  return (
    <div className="flex flex-col gap-3 pb-3">
      <p
        className="flex items-center text-[0.82rem] font-medium break-words text-foreground/90"
        style={{ minHeight: ROW_HEIGHT }}
      >
        {span.name}
      </p>
      <dl className="grid grid-cols-[6rem_1fr] gap-x-3 gap-y-1 text-[0.7rem]">
        {meta.map(([k, v]) => (
          <div className="contents" key={k}>
            <dt className="truncate text-muted-foreground/55">{k}</dt>
            <dd className="min-w-0 break-words text-foreground/85">{v}</dd>
          </div>
        ))}
      </dl>
      {input ? <InspectorBlock label="input" value={String(input)} /> : null}
      {output ? <InspectorBlock label="output" value={String(output)} /> : null}
    </div>
  )
}

function InspectorBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-[0.6rem] font-medium tracking-wider text-muted-foreground/50 uppercase">{label}</span>
      <pre className="max-h-40 overflow-auto rounded bg-foreground/5 p-2 text-[0.66rem] break-words whitespace-pre-wrap text-foreground/80">
        {value}
      </pre>
    </div>
  )
}
