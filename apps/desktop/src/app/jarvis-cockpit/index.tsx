import type { ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

interface CockpitAction {
  external_write?: boolean
  label?: string
  target?: string
  type?: string
}

interface CockpitStatusCard {
  actions?: CockpitAction[]
  details?: string[]
  id?: string
  kind?: string
  source?: string
  state?: string
  summary?: string
  title?: string
  updated_at?: string
}

interface CockpitTodoItem {
  area?: string
  artifact_refs?: string[]
  id?: string
  next_step?: string
  risk?: string
  source?: string
  status?: string
  title?: string
}

interface CockpitGate {
  id?: string
  reason?: string
  source?: string
  status?: string
  title?: string
  updated_at?: string
}

interface CockpitArtifact {
  id?: string
  kind?: string
  path?: string
  safe_to_open_in_cockpit?: boolean
  summary?: string
  title?: string
  updated_at?: string
}

interface CockpitSource {
  id?: string
  path?: string
  reason?: string
  state?: string
  updated_at?: string
}

interface JarvisCockpitResponse {
  artifacts?: CockpitArtifact[]
  dispatch_boundary?: {
    dispatch_embedded?: boolean
    dispatch_url?: string
    rule?: string
  }
  gates?: CockpitGate[]
  jarvis_todo?: CockpitTodoItem[]
  missing?: CockpitSource[]
  safety?: Record<string, boolean>
  sources?: CockpitSource[]
  status?: string
  status_cards?: CockpitStatusCard[]
  updated_at?: string
}

const sourceLabels: Record<string, string> = {
  'alfred-dispatch-operator-pack': 'Dispatch operator pointer',
  gates: 'Gates directory',
  'hermes-desktop-pinned-session-routing': 'Pinned chat routing',
  'jarvis-actions': 'Jarvis To Do',
  'jarvis-system-current-state': 'Jarvis status',
  'next-recommendation-guard': 'Goals / jobs guard',
  'personas-dashboards-source-map': 'Profiles / personas map'
}

const safetyLabels: Record<string, string> = {
  blikk_writes: 'Blikk writes',
  dispatch_embedded: 'Dispatch embedded',
  mail_mutation: 'Mail mutation',
  microsoft_writes: 'Microsoft writes',
  read_only: 'Read only',
  secrets_read: 'Secrets read'
}

const forbiddenSafetyKeys = new Set(['blikk_writes', 'dispatch_embedded', 'mail_mutation', 'microsoft_writes', 'secrets_read'])

const stateTone = (state?: string) => {
  if (state === 'ok') {
    return 'border-emerald-500/25 bg-emerald-500/8 text-emerald-200'
  }

  if (state === 'attention' || state === 'waiting') {
    return 'border-amber-500/30 bg-amber-500/10 text-amber-200'
  }

  if (state === 'missing' || state === 'error') {
    return 'border-zinc-500/25 bg-zinc-500/10 text-zinc-300'
  }

  return 'border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) text-(--ui-text-secondary)'
}

const niceDate = (value?: string) => {
  if (!value) {
    return null
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

const sourceLabel = (source?: CockpitSource | string) => {
  const id = typeof source === 'string' ? source : source?.id

  return (id && sourceLabels[id]) || id || 'Local source'
}

export function JarvisCockpitView() {
  const [data, setData] = useState<JarvisCockpitResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const loadCockpit = async () => {
    setRefreshing(true)
    setError(null)

    try {
      const response = await window.hermesDesktop.api<JarvisCockpitResponse>({ path: '/api/jarvis/cockpit' })
      setData(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load Jarvis Cockpit')
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    void loadCockpit()
  }, [])

  const sourcesById = useMemo(() => new Map((data?.sources || []).map(source => [source.id, source])), [data?.sources])
  const dispatchUrl = data?.dispatch_boundary?.dispatch_url
  const statusCards = data?.status_cards || []
  const todoItems = data?.jarvis_todo || []
  const gates = data?.gates || []
  const artifacts = data?.artifacts || []
  const sources = data?.sources || []
  const missing = data?.missing || []

  return (
    <div className="relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background) pt-(--titlebar-height) text-foreground">
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-5 pb-6 pt-4 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-(--ui-stroke-tertiary) pb-5 md:flex-row md:items-end md:justify-between">
          <div className="max-w-3xl">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) px-2.5 py-1 text-[0.72rem] font-medium uppercase tracking-[0.18em] text-(--ui-text-tertiary)">
              <Codicon className="size-3.5" name="shield" />
              Read-only local Jarvis docs
            </div>
            <h1 className="text-2xl font-semibold tracking-[-0.03em] md:text-3xl">Jarvis Cockpit</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-(--ui-text-secondary)">
              Desktop-first operator pane for Jarvis/Hermes system work. Dispatch, Microsoft, Blikk, mail,
              secrets and NUC/backup controls stay outside this read-only surface.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {data?.updated_at && <Pill label={`Updated ${niceDate(data.updated_at)}`} />}
            <button
              className="inline-flex h-8 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-60"
              disabled={refreshing}
              onClick={() => void loadCockpit()}
              type="button"
            >
              <Codicon className={cn('size-4', refreshing && 'animate-spin')} name="refresh" />
              {refreshing ? 'Refreshing' : 'Refresh'}
            </button>
          </div>
        </header>

        {error && (
          <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
            {error}
          </div>
        )}

        {!data && !error ? <CockpitSkeleton /> : null}

        {data ? (
          <main className="grid gap-5 pt-5">
            <section className="grid gap-3 lg:grid-cols-4">
              <MetricCard label="Status" tone={data.status === 'ok' ? 'ok' : 'muted'} value={data.status || 'unknown'} />
              <MetricCard detail="local action items" label="Jarvis To Do" value={String(todoItems.length)} />
              <MetricCard detail="local gates" label="Waiting gates" value={String(gates.filter(gate => gate.status === 'waiting').length)} />
              <MetricCard detail="reported, not fatal" label="Missing sources" tone={missing.length ? 'attention' : 'ok'} value={String(missing.length)} />
            </section>

            <section className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(21rem,0.8fr)]">
              <div className="grid gap-5">
                <Panel icon="shield" title="Jarvis status / safety">
                  <div className="grid gap-3 md:grid-cols-2">
                    {statusCards.map(card => (
                      <StatusCard card={card} key={card.id || card.title} />
                    ))}
                  </div>
                  <SafetyMatrix safety={data.safety || {}} />
                </Panel>

                <Panel icon="checklist" title="Jarvis To Do">
                  {todoItems.length ? (
                    <div className="grid gap-2">
                      {todoItems.map(item => (
                        <TodoRow item={item} key={item.id || item.title} />
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="No local Jarvis action items returned by the read-only API." />
                  )}
                </Panel>

                <Panel icon="lock" title="Gates">
                  {gates.length ? (
                    <div className="grid gap-2">
                      {gates.map(gate => (
                        <GateRow gate={gate} key={gate.id || gate.title} />
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="No gate files returned. Missing gate directories are shown in source health instead of crashing the pane." />
                  )}
                </Panel>
              </div>

              <aside className="grid content-start gap-5">
                <Panel icon="link-external" title="Dispatch boundary">
                  <p className="text-sm leading-6 text-(--ui-text-secondary)">{data.dispatch_boundary?.rule}</p>
                  <div className="mt-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 text-xs text-(--ui-text-tertiary)">
                    Dispatch embedded: <strong className="text-foreground">{String(Boolean(data.dispatch_boundary?.dispatch_embedded))}</strong>
                  </div>
                  {dispatchUrl && (
                    <button
                      className="mt-3 inline-flex h-8 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                      onClick={() => void window.hermesDesktop.openExternal(dispatchUrl)}
                      type="button"
                    >
                      <Codicon className="size-4" name="link-external" />
                      Open Dispatch Dashboard
                    </button>
                  )}
                </Panel>

                <PointerPanel
                  description="Profiles/personas are surfaced only as source-map pointers, not profile/config mutation controls."
                  icon="organization"
                  source={sourcesById.get('personas-dashboards-source-map')}
                  title="Profiles / personas pointer"
                />

                <PointerPanel
                  description="Goals, jobs and pinned-session routing stay pointers into local runbooks. No cron, kanban, profile or gateway controls are rendered here."
                  icon="target"
                  source={sourcesById.get('next-recommendation-guard')}
                  title="Goals / jobs pointer"
                />

                <PointerPanel
                  description="Source-map and skill hygiene are represented by allowlisted local docs and their health states."
                  icon="symbol-misc"
                  source={sourcesById.get('jarvis-system-current-state')}
                  title="Source-map / skill hygiene"
                />

                <Panel icon="file-media" title="Artifacts">
                  {artifacts.length ? (
                    <div className="grid gap-2">
                      {artifacts.map(artifact => (
                        <ArtifactRow artifact={artifact} key={artifact.id || artifact.path || artifact.title} />
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="No safe artifact pointers returned." />
                  )}
                </Panel>

                <Panel icon="database" title="Source health">
                  <div className="grid gap-2">
                    {sources.map(source => (
                      <SourceRow key={source.id || source.path} source={source} />
                    ))}
                  </div>
                </Panel>
              </aside>
            </section>
          </main>
        ) : null}
      </div>
    </div>
  )
}

function Panel({ children, icon, title }: { children: ReactNode; icon: string; title: string }) {
  return (
    <section className="rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-surface-elevated-background)/70 p-4 shadow-[0_18px_55px_rgba(0,0,0,0.18)]">
      <div className="mb-3 flex items-center gap-2">
        <span className="grid size-7 place-items-center rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) text-(--ui-text-secondary)">
          <Codicon className="size-4" name={icon} />
        </span>
        <h2 className="text-sm font-semibold tracking-[-0.01em]">{title}</h2>
      </div>
      {children}
    </section>
  )
}

function Pill({ label }: { label: string }) {
  return (
    <span className="inline-flex h-8 items-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) px-3 text-xs text-(--ui-text-tertiary)">
      {label}
    </span>
  )
}

function MetricCard({ detail, label, tone = 'muted', value }: { detail?: string; label: string; tone?: 'attention' | 'muted' | 'ok'; value: string }) {
  return (
    <div
      className={cn(
        'rounded-2xl border p-4',
        tone === 'ok'
          ? 'border-emerald-500/25 bg-emerald-500/8'
          : tone === 'attention'
            ? 'border-amber-500/30 bg-amber-500/10'
            : 'border-(--ui-stroke-tertiary) bg-(--ui-surface-elevated-background)/65'
      )}
    >
      <div className="text-xs uppercase tracking-[0.16em] text-(--ui-text-tertiary)">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-[-0.04em]">{value}</div>
      {detail && <div className="mt-1 text-xs text-(--ui-text-tertiary)">{detail}</div>}
    </div>
  )
}

function StatusCard({ card }: { card: CockpitStatusCard }) {
  const openUrlAction = card.actions?.find(action => action.type === 'open-url' && action.target && !action.external_write)

  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium">{card.title || card.id}</h3>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{card.summary}</p>
        </div>
        <StateBadge state={card.state} />
      </div>
      {card.details?.length ? (
        <ul className="mt-2 grid gap-1 text-xs text-(--ui-text-tertiary)">
          {card.details.map(detail => (
            <li className="flex gap-1.5" key={detail}>
              <span aria-hidden="true">•</span>
              <span>{detail}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {openUrlAction?.target && (
        <button
          className="mt-3 inline-flex h-7 items-center gap-1.5 rounded-md border border-(--ui-stroke-tertiary) px-2 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => void window.hermesDesktop.openExternal(openUrlAction.target!)}
          type="button"
        >
          <Codicon className="size-3.5" name="link-external" />
          {openUrlAction.label || 'Open link'}
        </button>
      )}
    </article>
  )
}

function SafetyMatrix({ safety }: { safety: Record<string, boolean> }) {
  return (
    <div className="mt-4 grid gap-2 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 sm:grid-cols-2 lg:grid-cols-3">
      {Object.entries(safety).map(([key, value]) => {
        const safe = key === 'read_only' ? value === true : forbiddenSafetyKeys.has(key) ? value === false : !value

        return (
          <div className="flex items-center justify-between gap-3 text-xs" key={key}>
            <span className="text-(--ui-text-secondary)">{safetyLabels[key] || key.replaceAll('_', ' ')}</span>
            <span className={cn('font-medium', safe ? 'text-emerald-300' : 'text-red-300')}>{String(value).toUpperCase()}</span>
          </div>
        )
      })}
    </div>
  )
}

function TodoRow({ item }: { item: CockpitTodoItem }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-medium">{item.title || item.id}</h3>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{item.next_step || item.source}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <StateBadge state={item.status} />
          {item.risk && <StateBadge state={item.risk} />}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-xs text-(--ui-text-tertiary)">
        {item.area && <span>{item.area}</span>}
        {item.source && <span>Source: {item.source}</span>}
      </div>
    </article>
  )
}

function GateRow({ gate }: { gate: CockpitGate }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-medium">{gate.title || gate.id || 'Gate'}</h3>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{gate.reason || gate.source || 'Local gate pointer'}</p>
        </div>
        <StateBadge state={gate.status} />
      </div>
      {gate.updated_at && <div className="mt-2 text-xs text-(--ui-text-tertiary)">Updated {niceDate(gate.updated_at)}</div>}
    </article>
  )
}

function PointerPanel({ description, icon, source, title }: { description: string; icon: string; source?: CockpitSource; title: string }) {
  return (
    <Panel icon={icon} title={title}>
      <p className="text-sm leading-6 text-(--ui-text-secondary)">{description}</p>
      {source ? <SourceRow source={source} /> : <div className="mt-3"><EmptyState text="Pointer source was not returned." /></div>}
    </Panel>
  )
}

function ArtifactRow({ artifact }: { artifact: CockpitArtifact }) {
  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="truncate font-medium">{artifact.title || artifact.id || 'Artifact'}</h3>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{artifact.summary || artifact.path}</p>
        </div>
        <StateBadge state={artifact.safe_to_open_in_cockpit ? 'ok' : 'pointer'} />
      </div>
      {artifact.path && <div className="mt-2 truncate font-mono text-[0.7rem] text-(--ui-text-tertiary)">{artifact.path}</div>}
    </article>
  )
}

function SourceRow({ source }: { source: CockpitSource }) {
  return (
    <div className="mt-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-2.5 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-(--ui-text-secondary)">{sourceLabel(source)}</span>
        <StateBadge state={source.state} />
      </div>
      {source.path && <div className="mt-1 truncate font-mono text-[0.68rem] text-(--ui-text-tertiary)">{source.path}</div>}
      {source.reason && <div className="mt-1 text-(--ui-text-tertiary)">{source.reason}</div>}
    </div>
  )
}

function StateBadge({ state }: { state?: string }) {
  return (
    <span className={cn('inline-flex h-5 shrink-0 items-center rounded-full border px-2 text-[0.68rem] font-medium', stateTone(state))}>
      {state || 'unknown'}
    </span>
  )
}

function EmptyState({ text }: { text: string }) {
  return <div className="rounded-xl border border-dashed border-(--ui-stroke-tertiary) p-4 text-sm text-(--ui-text-tertiary)">{text}</div>
}

function CockpitSkeleton() {
  return (
    <div className="grid gap-5 pt-5">
      <div className="grid gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div className="h-28 animate-pulse rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background)" key={i} />
        ))}
      </div>
      <div className="h-96 animate-pulse rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background)" />
    </div>
  )
}
