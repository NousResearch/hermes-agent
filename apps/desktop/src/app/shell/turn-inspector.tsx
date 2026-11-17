import type { RenderSnapshotChange, RenderSnapshotSegment, SessionRenderSnapshotResult } from '@hermes/shared'
import { compactNumber } from '@hermes/shared'
import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import type { ContextUsagePanelCopy } from '@/i18n/types_context_usage_panel'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

export type RequestGateway = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

type InspectorCopy = ContextUsagePanelCopy['inspector']

type SnapshotState =
  { status: 'error' } | { status: 'loading' } | { snapshot: SessionRenderSnapshotResult; status: 'ready' }

/** `session.render_snapshot` for the focused session, read when the inspector tab opens. The RPC is
 *  read-only over the stored prompt hash, so opening it never touches the prompt cache. */
export function useRenderSnapshot(requestGateway: RequestGateway, sessionId: null | string): SnapshotState {
  const [state, setState] = useState<SnapshotState>({ status: 'loading' })

  useEffect(() => {
    if (!sessionId) {
      setState({ status: 'error' })

      return
    }

    let cancelled = false
    setState({ status: 'loading' })
    void requestGateway<SessionRenderSnapshotResult>('session.render_snapshot', { session_id: sessionId })
      .then(snapshot => !cancelled && setState({ snapshot, status: 'ready' }))
      .catch(() => !cancelled && setState({ status: 'error' }))

    return () => {
      cancelled = true
    }
  }, [requestGateway, sessionId])

  return state
}

const CHANGE_LABELS: Record<
  RenderSnapshotChange['kind'],
  (copy: InspectorCopy, change: RenderSnapshotChange) => string
> = {
  model_changed: (copy, change) => copy.changes.model_changed(change.before || '—', change.after || '—'),
  prompt_changed: copy => copy.changes.prompt_changed,
  provider_changed: (copy, change) => copy.changes.provider_changed(change.before || '—', change.after || '—'),
  tool_added: (copy, change) => copy.changes.tool_added(change.name ?? ''),
  tool_removed: (copy, change) => copy.changes.tool_removed(change.name ?? '')
}

const CHANGE_TONE: Record<RenderSnapshotChange['kind'], string> = {
  model_changed: 'border-amber-500/40 text-amber-600 dark:text-amber-400',
  prompt_changed: 'border-sky-500/40 text-sky-600 dark:text-sky-400',
  provider_changed: 'border-amber-500/40 text-amber-600 dark:text-amber-400',
  tool_added: 'border-emerald-500/40 text-emerald-600 dark:text-emerald-400',
  tool_removed: 'border-rose-500/40 text-rose-600 dark:text-rose-400'
}

export function ChangeChips({ changes }: { changes: readonly RenderSnapshotChange[] }) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.inspector

  if (!changes.length) {
    return <span className="text-[0.625rem] text-muted-foreground">{copy.noChanges}</span>
  }

  return (
    <span className="flex flex-wrap gap-1" data-slot="inspector-change-chips">
      {changes.map(change => (
        <span
          className={cn('rounded-full border px-1.5 py-px text-[0.625rem] leading-4', CHANGE_TONE[change.kind])}
          data-kind={change.kind}
          key={`${change.kind}:${change.name ?? change.after ?? ''}`}
        >
          {CHANGE_LABELS[change.kind](copy, change)}
        </span>
      ))}
    </span>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <p className="flex items-baseline justify-between gap-2">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="truncate text-foreground">{value}</span>
    </p>
  )
}

function Section({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <section className="flex flex-col gap-1">
      <p className="text-[0.6875rem] font-medium text-foreground">{title}</p>
      {children}
    </section>
  )
}

function Lineage({ copy, segments }: { copy: InspectorCopy; segments: readonly RenderSnapshotSegment[] }) {
  return (
    <Section title={copy.lineage(segments.length)}>
      <ol className="flex flex-col gap-1.5" data-slot="inspector-lineage">
        {segments.map((segment, index) => (
          <li className="flex flex-col gap-0.5" key={segment.session_id}>
            <span className="flex items-baseline justify-between gap-2 text-[0.6875rem]">
              <span className="text-foreground">{copy.segment(index + 1)}</span>
              <span className="truncate font-mono text-[0.625rem] text-muted-foreground">{segment.session_id}</span>
            </span>
            {index > 0 && <ChangeChips changes={segment.changes ?? []} />}
          </li>
        ))}
      </ol>
    </Section>
  )
}

/** What the agent saw this session: the stored system prompt, the route, its tools (with their
 *  descriptions) and MCP servers, and the change chips between compression segments. */
export function TurnInspector({
  requestGateway,
  sessionId
}: {
  requestGateway: RequestGateway
  sessionId: null | string
}) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.inspector
  const state = useRenderSnapshot(requestGateway, sessionId)

  if (state.status === 'loading') {
    return (
      <p className="inline-flex items-center gap-1.5 text-[0.6875rem] text-muted-foreground">
        <Loader2 className="size-3 animate-spin" />
        {copy.loading}
      </p>
    )
  }

  if (state.status === 'error') {
    return <p className="text-[0.6875rem] text-muted-foreground">{copy.error}</p>
  }

  const { snapshot } = state
  const prompt = snapshot.system_prompt ?? ''

  return (
    <div className="flex max-h-[28rem] flex-col gap-3 overflow-y-auto pr-1" data-slot="turn-inspector">
      <div className="flex flex-col gap-0.5">
        <Field label={copy.model} value={snapshot.model || '—'} />
        <Field label={copy.provider} value={snapshot.provider || '—'} />
        <Field
          label={copy.mcpServers}
          value={snapshot.mcp_servers.length ? snapshot.mcp_servers.join(', ') : copy.none}
        />
      </div>

      {snapshot.segments.length > 1 && <Lineage copy={copy} segments={snapshot.segments} />}

      <Section title={copy.systemPrompt}>
        {prompt ? (
          <details className="group">
            <summary className="cursor-pointer text-[0.6875rem] text-muted-foreground">
              {copy.promptChars(compactNumber(prompt.length))}
            </summary>
            <pre className="mt-1 max-h-48 overflow-auto rounded-md bg-(--ui-bg-elevated) p-2 font-mono text-[0.625rem] leading-4 whitespace-pre-wrap text-foreground">
              {prompt}
            </pre>
          </details>
        ) : (
          <p className="text-[0.6875rem] text-muted-foreground">{copy.noPrompt}</p>
        )}
      </Section>

      <Section title={copy.tools(snapshot.tools.length)}>
        <p className="text-[0.625rem] text-muted-foreground">
          {snapshot.tools_source === 'live' ? copy.sourceLive : copy.sourceStored}
          {snapshot.toolsets.length ? ` · ${snapshot.toolsets.join(', ')}` : ''}
        </p>
        <ul className="flex flex-col gap-1" data-slot="inspector-tools">
          {snapshot.tools.map(tool => (
            <li className="flex flex-col" key={tool.name}>
              <span className="font-mono text-[0.6875rem] text-foreground">{tool.name}</span>
              {tool.description && (
                <span className="line-clamp-2 text-[0.625rem] text-muted-foreground">{tool.description}</span>
              )}
            </li>
          ))}
        </ul>
      </Section>
    </div>
  )
}
