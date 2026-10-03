import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import { $hoveredSpanId, $selectedSpanId, type TraceDoc } from '@/store/trace'

import { fmtDuration } from './format'
import { ROW_HEIGHT } from './trace-waterfall'

const fmtInt = (n: number) => n.toLocaleString()

export function SpanInspector({ trace }: { trace: null | TraceDoc }) {
  const { t } = useI18n()
  const labels = t.agents.inspector
  const selectedId = useStore($selectedSpanId)
  const hoveredId = useStore($hoveredSpanId)
  // Hover previews; the clicked span stays pinned when nothing is hovered.
  const activeId = hoveredId ?? selectedId

  const span = useMemo(() => trace?.spans.find(s => s.id === activeId) ?? null, [trace, activeId])

  if (!span) {
    return (
      <div className="flex items-center text-[0.7rem] text-muted-foreground/55" style={{ height: ROW_HEIGHT }}>
        {t.agents.inspectHint}
      </div>
    )
  }

  const attrs = span.attributes
  const num = (key: string) => (typeof attrs[key] === 'number' ? (attrs[key] as number) : undefined)

  const meta: [string, string][] = [
    [labels.kind, span.kind],
    [labels.status, span.status]
  ]

  // Where the span sits in the trace, then how long it ran.
  if (trace) {
    meta.push([labels.started, `+${fmtDuration(Math.max(0, span.start - trace.start))}`])
  }

  meta.push([labels.duration, fmtDuration(span.duration)])

  const firstToken = num('llm.first_token_s')

  if (firstToken !== undefined) {
    meta.push([labels.firstToken, fmtDuration(firstToken)])
  }

  // Push an attribute row when present; numbers are thousands-formatted.
  const push = (label: string, key: string) => {
    const v = attrs[key]

    if (v !== undefined && v !== null && v !== '') {
      meta.push([label, typeof v === 'number' ? fmtInt(v) : String(v)])
    }
  }

  push(labels.model, 'llm.model_name')
  push(labels.tokensIn, 'llm.token_count.prompt')
  push(labels.tokensOut, 'llm.token_count.completion')

  const treason = num('llm.token_count.reasoning')

  if (treason) {
    meta.push([labels.reasoning, fmtInt(treason)])
  }

  const tin = num('llm.token_count.prompt')
  const tout = num('llm.token_count.completion')

  if (tin !== undefined || tout !== undefined) {
    meta.push([labels.tokensTotal, fmtInt((tin ?? 0) + (tout ?? 0) + (treason ?? 0))])
  }

  push(labels.finish, 'hermes.finish_reason')
  push(labels.tool, 'tool.name')
  push(labels.source, 'session.source')

  if (span.sessionId) {
    meta.push([labels.session, span.sessionId.slice(0, 12)])
  }

  const input = attrs['input.value']
  const thinking = attrs['llm.reasoning.value']
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
      {input ? <InspectorBlock label={labels.input} value={String(input)} /> : null}
      {thinking ? <InspectorBlock label={labels.thinking} value={String(thinking)} /> : null}
      {output ? <InspectorBlock label={labels.output} value={String(output)} /> : null}
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
