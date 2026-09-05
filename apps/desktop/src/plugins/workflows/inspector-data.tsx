/**
 * The step as run. Read-only throughout: everything here comes off the
 * runtime the event reducer built, so a scrubbed timeline shows the step as it
 * was at that point rather than as it ended.
 */

import { cn, SidePanelMeta, SidePanelMetaRow, SidePanelSection } from '@hermes/plugin-sdk'

import type { StepRuntime } from './protocol'
import type { StepKind } from './scenario'

const TODO_MARK: Record<string, string> = {
  cancelled: '[~]',
  completed: '[x]',
  in_progress: '[>]',
  pending: '[ ]'
}

function renderFieldValue(v: unknown): React.ReactNode {
  if (Array.isArray(v)) {
    return v.length ? v.join(', ') : '[]'
  }

  if (v !== null && typeof v === 'object') {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, val]) => `${k}: ${String(val)}`)
      .join(', ')
  }

  if (typeof v === 'string' && /^https?:\/\//i.test(v)) {
    return (
      <a className="node-link" href={v} rel="noreferrer" target="_blank">
        {v}
      </a>
    )
  }

  return String(v)
}

function Count({ n }: { n: number | string }) {
  return <span className="text-[0.62rem] tabular-nums text-(--ui-text-quaternary)">{n}</span>
}

export function DataTab({ kind, rt }: { kind: StepKind; rt: StepRuntime }) {
  // A gate takes children and returns a decision; every other kind takes input
  // and returns a summary. Same two rows either way.
  const isGate = kind === 'gate'

  return (
    <div className="flex flex-col gap-4 text-sm">
      <SidePanelMeta>
        <SidePanelMetaRow label="Status">
          {rt.status}
          {rt.verdict ? ` · ${rt.verdict}` : ''}
        </SidePanelMetaRow>
        {rt.durationMs != null && (
          <SidePanelMetaRow label="Duration">{(rt.durationMs / 1000).toFixed(1)}s</SidePanelMetaRow>
        )}
        {rt.tokens > 0 && (
          <SidePanelMetaRow label="Tokens">
            {rt.tokens >= 1000 ? `${(rt.tokens / 1000).toFixed(1)}k` : rt.tokens}
          </SidePanelMetaRow>
        )}
        {rt.maxIters > 0 && rt.iterations > 0 && (
          <SidePanelMetaRow label="Iterations">
            {rt.iterations}/{rt.maxIters}
          </SidePanelMetaRow>
        )}
        {rt.take > 1 && <SidePanelMetaRow label="Take">{rt.take}</SidePanelMetaRow>}
        <SidePanelMetaRow label={isGate ? 'Children' : 'Input'} wrap>
          {rt.input ?? '—'}
        </SidePanelMetaRow>
        <SidePanelMetaRow label={isGate ? 'Decision' : 'Summary'} wrap>
          {rt.summary ?? '—'}
        </SidePanelMetaRow>
      </SidePanelMeta>

      {rt.output && (
        <SidePanelSection action={<Count n={Object.keys(rt.output).length} />} label="Output">
          <ul className="flex flex-col gap-2">
            {Object.entries(rt.output).map(([k, v]) => (
              <li className="text-[0.75rem]" key={k}>
                <span className="font-medium text-(--ui-text-secondary)">{k}</span>
                <div className="whitespace-pre-wrap text-(--ui-text-tertiary)">{renderFieldValue(v)}</div>
              </li>
            ))}
          </ul>
        </SidePanelSection>
      )}

      {rt.todos.length > 0 && (
        <SidePanelSection
          action={<Count n={`${rt.todos.filter(t => t.status === 'completed').length}/${rt.todos.length}`} />}
          label="Plan · todo tool"
        >
          <ul className="flex flex-col gap-1">
            {rt.todos.map(t => (
              <li className="flex items-baseline gap-2 text-[0.6875rem]" key={t.id}>
                <span className="shrink-0 text-(--ui-text-quaternary)">{TODO_MARK[t.status]}</span>
                <span
                  className={cn(
                    'min-w-0 text-(--ui-text-secondary)',
                    (t.status === 'completed' || t.status === 'cancelled') && 'text-(--ui-text-tertiary) line-through'
                  )}
                >
                  {t.content}
                </span>
              </li>
            ))}
          </ul>
        </SidePanelSection>
      )}

      {rt.toolCalls.length > 0 && (
        <SidePanelSection action={<Count n={rt.toolCalls.length} />} label="Activity">
          <ul className="flex flex-col gap-1">
            {rt.toolCalls.map((c, i) => (
              <li className="flex items-baseline gap-2 text-[0.6875rem]" key={i}>
                <span className="shrink-0 text-(--ui-text-secondary)">{c.name}</span>
                {c.arg && (
                  <span className="min-w-0 truncate text-[0.625rem] text-(--ui-text-quaternary)" title={c.arg}>
                    {c.arg}
                  </span>
                )}
              </li>
            ))}
            {rt.currentTool && (rt.status === 'running' || rt.status === 'looping') && (
              <li className="flex items-baseline gap-2 text-[0.6875rem]">
                <span className="shrink-0 text-(--ui-text-secondary)">{rt.currentTool.name}</span>
                {rt.currentTool.arg && (
                  <span className="min-w-0 truncate text-[0.625rem] text-(--ui-text-quaternary)">
                    {rt.currentTool.arg}
                  </span>
                )}
              </li>
            )}
          </ul>
        </SidePanelSection>
      )}
    </div>
  )
}
