import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { OverlayView } from '@/app/overlays/overlay-view'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { formatK } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import {
  $activeBucket,
  $activeTab,
  $contextData,
  $contextSource,
  closeContextInspector,
  type ContextInspectorBucket,
  type ContextInspectorTab
} from '@/store/context-inspector'
import type { ContextFull, ContextMessage, ContextSlice } from '@/types/hermes'

interface ContextInspectorViewProps {
  onClose: () => void
}

const BUCKETS: readonly ContextInspectorBucket[] = ['all', 'system', 'user', 'assistant', 'tools']
const TABS: readonly ContextInspectorTab[] = ['transcript', 'raw']

const ROLE_TONE: Record<ContextMessage['role'], string> = {
  assistant: 'border-emerald-500/70',
  system: 'border-purple-400/70',
  tool: 'border-amber-500/75',
  user: 'border-blue-500/70'
}

export function ContextInspectorView({ onClose }: ContextInspectorViewProps) {
  const { t } = useI18n()
  const copy = t.contextInspector
  const data = useStore($contextData)
  const source = useStore($contextSource)
  const activeBucket = useStore($activeBucket)
  const activeTab = useStore($activeTab)

  const close = () => {
    closeContextInspector()
    onClose()
  }

  return (
    <OverlayView closeLabel={copy.close} contentClassName="min-h-0" onClose={close}>
      <div className="flex min-h-0 flex-1 flex-col">
        <header className="flex shrink-0 items-start justify-between gap-3 border-b border-(--ui-stroke-secondary) px-4 pb-3 pt-[calc(var(--titlebar-height)/2+0.75rem)] sm:px-5">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 className="text-sm font-semibold text-foreground">{copy.title}</h2>
              <SourcePill data={data} />
            </div>
            <p className="mt-1 max-w-4xl text-xs text-muted-foreground/80">{copy.sourceHelp}</p>
          </div>

          <div className="flex shrink-0 items-center gap-2 pr-9 text-xs text-muted-foreground/85">
            {data ? <span className="rounded-full border border-border/55 px-2 py-1">{tokenSummary(data)}</span> : null}
            {data?.model ? <span className="rounded-full border border-border/55 px-2 py-1">{data.model}</span> : null}
          </div>
        </header>

        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-(--ui-stroke-secondary) px-4 py-2 sm:px-5">
          <div aria-label={copy.bucketGroupLabel} className="flex flex-wrap items-center gap-1.5" role="group">
            {BUCKETS.map(bucket => (
              <button
                className={cn(
                  'rounded-full border px-2.5 py-1 text-xs transition-colors',
                  activeBucket === bucket
                    ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-foreground'
                    : 'border-border/60 text-muted-foreground hover:bg-(--chrome-action-hover) hover:text-foreground'
                )}
                key={bucket}
                onClick={() => $activeBucket.set(bucket)}
                type="button"
              >
                {copy.buckets[bucket]}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 rounded-lg border border-border/55 bg-muted/15 p-0.5" role="tablist">
            {TABS.map(tab => (
              <button
                aria-selected={activeTab === tab}
                className={cn(
                  'rounded-md px-2.5 py-1 text-xs transition-colors',
                  activeTab === tab ? 'bg-(--chrome-action-hover) text-foreground' : 'text-muted-foreground hover:text-foreground'
                )}
                key={tab}
                onClick={() => $activeTab.set(tab)}
                role="tab"
                type="button"
              >
                {copy.tabs[tab]}
              </button>
            ))}
          </div>
        </div>

        <main className="min-h-0 flex-1 overflow-hidden px-4 py-3 sm:px-5">
          <ContextInspectorBody activeBucket={activeBucket} activeTab={activeTab} data={data} source={source} />
        </main>
      </div>
    </OverlayView>
  )
}

function ContextInspectorBody({
  activeBucket,
  activeTab,
  data,
  source
}: {
  activeBucket: ContextInspectorBucket
  activeTab: ContextInspectorTab
  data: ContextFull | null
  source: ReturnType<typeof $contextSource.get>
}) {
  const { t } = useI18n()
  const copy = t.contextInspector

  if (source.status === 'loading') {
    return <StateCard icon="loading~spin" text={copy.loading} />
  }

  if (source.status === 'error') {
    return <StateCard icon="error" text={copy.error(source.error)} tone="error" />
  }

  if (!data || !data.available || data.state === 'agent_not_built') {
    return <StateCard icon="info" text={copy.empty} />
  }

  if (activeTab === 'raw') {
    return <RawJsonExplorer data={data} />
  }

  return <TranscriptStream activeBucket={activeBucket} data={data} />
}

function StateCard({ icon, text, tone = 'muted' }: { icon: string; text: string; tone?: 'error' | 'muted' }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div
        className={cn(
          'flex max-w-md items-center gap-2 rounded-xl border border-border/60 bg-muted/20 px-4 py-3 text-sm',
          tone === 'error' ? 'text-destructive' : 'text-muted-foreground'
        )}
      >
        <Codicon aria-hidden name={icon} size="1rem" />
        <span>{text}</span>
      </div>
    </div>
  )
}

function SourcePill({ data }: { data: ContextFull | null }) {
  const { t } = useI18n()
  const copy = t.contextInspector
  const label = data?.source_label || copy.sourceFallback

  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-blue-400/35 bg-blue-500/10 px-2 py-0.5 text-[0.7rem] font-medium text-blue-700 dark:text-blue-200">
      {label}
      <Codicon aria-label={copy.sourceHelpLabel} name="info" size="0.78rem" />
    </span>
  )
}

function tokenSummary(data: ContextFull): string {
  return `${formatK(data.context_used)} / ${formatK(data.context_max)}`
}

function TranscriptStream({ activeBucket, data }: { activeBucket: ContextInspectorBucket; data: ContextFull }) {
  const { t } = useI18n()
  const copy = t.contextInspector
  const [systemOpen, setSystemOpen] = useState(true)

  const messages = useMemo(
    () => data.messages.filter(message => activeBucket === 'all' || bucketForMessage(message) === activeBucket),
    [activeBucket, data.messages]
  )

  const hasTruncatedSlice = useMemo(() => data.slices.some(slice => slice.truncated), [data.slices])

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 overflow-y-auto overscroll-contain pr-1" data-selectable-text="true">
      {hasTruncatedSlice ? (
        <p className="rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
          {copy.truncated}
        </p>
      ) : null}
      {messages.length ? (
        messages.map(message => {
          const isSystem = message.role === 'system'
          const collapsed = isSystem && !systemOpen

          return (
            <article
              className={cn('rounded-xl border border-border/55 border-l-4 bg-muted/15 p-3', ROLE_TONE[message.role])}
              key={`${message.index}:${message.role}`}
            >
              <div className="mb-2 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-xs font-semibold tracking-wider text-foreground/90 uppercase">
                    {messageLabel(message)}
                  </h3>
                  <p className="mt-1 flex flex-wrap gap-1.5 text-[0.68rem] text-muted-foreground/80">
                    <span>{copy.index(message.index)}</span>
                    <span>{copy.tokens(formatK(message.tokens))}</span>
                    {messageBadges(message).map(badge => (
                      <span className="rounded-full bg-muted/40 px-1.5 py-0.5" key={badge}>
                        {badge}
                      </span>
                    ))}
                  </p>
                </div>

                {isSystem ? (
                  <Button onClick={() => setSystemOpen(value => !value)} size="sm" type="button" variant="ghost">
                    {systemOpen ? copy.collapseSystem : copy.expandSystem}
                  </Button>
                ) : null}
              </div>

              {collapsed ? (
                <p className="text-xs text-muted-foreground/80">{copy.systemCollapsed}</p>
              ) : (
                <pre className="max-h-[34rem] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/45 bg-background/35 p-3 font-mono text-[0.72rem] leading-relaxed text-foreground/90">
                  {message.content_text}
                </pre>
              )}
            </article>
          )
        })
      ) : (
        <StateCard icon="search" text={copy.noMessages} />
      )}
    </div>
  )
}

function bucketForMessage(message: ContextMessage): ContextInspectorBucket {
  return message.role === 'tool' ? 'tools' : message.role
}

function messageLabel(message: ContextMessage): string {
  if (message.role === 'tool') {
    const rawName = message.raw.name
    const rawToolId = message.raw.tool_call_id
    const name = typeof rawName === 'string' && rawName ? rawName : 'tool'
    const toolId = typeof rawToolId === 'string' && rawToolId ? ` · tool_call_id: ${rawToolId}` : ''

    return `TOOL · ${name}${toolId}`
  }

  return message.role.toUpperCase()
}

function messageBadges(message: ContextMessage): string[] {
  const badges: string[] = []

  if (message.role === 'assistant') {
    if ('reasoning' in message.raw || 'reasoning_content' in message.raw) {
      badges.push('reasoning')
    }

    if ('tool_calls' in message.raw) {
      badges.push('tool_calls')
    }
  }

  // Tool name + tool_call_id are already shown in the message label; do not
  // repeat them as badges (would double-render the same text).
  return badges
}

function RawJsonExplorer({ data }: { data: ContextFull }) {
  const { t } = useI18n()
  const copy = t.contextInspector
  const [selected, setSelected] = useState<RawSelection>({ kind: 'payload' })
  const selectedPayload = selectedPayloadFor(data, selected)

  return (
    <div className="grid h-full min-h-0 grid-cols-[14rem_minmax(0,1fr)] overflow-hidden rounded-xl border border-border/55 max-[48rem]:grid-cols-1">
      <aside className="min-h-0 overflow-y-auto border-r border-border/55 bg-muted/15 p-3 font-mono text-[0.72rem] max-[48rem]:hidden">
        <RawTreeButton active={selected.kind === 'payload'} label={copy.fullPayload} onClick={() => setSelected({ kind: 'payload' })} />
        <p className="mt-3 text-muted-foreground">slices</p>
        {data.slices.map(slice => (
          <RawTreeButton
            active={selected.kind === 'slice' && selected.id === slice.id}
            key={slice.id}
            label={`${slice.id} · ${formatK(slice.tokens)}`}
            onClick={() => setSelected({ id: slice.id, kind: 'slice' })}
          />
        ))}
        <p className="mt-3 text-muted-foreground">messages [{data.messages.length}]</p>
        {data.messages.map(message => (
          <RawTreeButton
            active={selected.kind === 'message' && selected.index === message.index}
            key={message.index}
            label={`${message.index} ${message.role}`}
            onClick={() => setSelected({ index: message.index, kind: 'message' })}
          />
        ))}
      </aside>
      <section className="min-h-0 overflow-auto p-3" data-selectable-text="true">
        {selected.kind === 'slice' ? <SliceSummary slice={selectedPayload as ContextSlice} /> : null}
        <pre className="min-h-full whitespace-pre-wrap break-words rounded-lg bg-background/40 p-3 font-mono text-[0.72rem] leading-relaxed text-foreground/90">
          {JSON.stringify(selectedPayload, null, 2)}
        </pre>
      </section>
    </div>
  )
}

type RawSelection = { kind: 'payload' } | { id: ContextSlice['id']; kind: 'slice' } | { index: number; kind: 'message' }

function RawTreeButton({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      className={cn(
        'mt-1 block w-full truncate rounded px-2 py-1 text-left',
        active ? 'bg-(--chrome-action-hover) text-foreground' : 'text-muted-foreground hover:text-foreground'
      )}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}

function selectedPayloadFor(data: ContextFull, selection: RawSelection): ContextFull | ContextMessage | ContextSlice {
  if (selection.kind === 'slice') {
    return data.slices.find(slice => slice.id === selection.id) ?? data
  }

  if (selection.kind === 'message') {
    return data.messages.find(message => message.index === selection.index) ?? data
  }

  return data
}

function SliceSummary({ slice }: { slice: ContextSlice }) {
  const { t } = useI18n()
  const copy = t.contextInspector

  return slice.truncated ? (
    <p className="mb-2 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
      {copy.truncated}
    </p>
  ) : null
}
