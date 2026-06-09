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
  decision?: string
  default_action?: string
  id?: string
  owner?: string
  recommendation?: string
  reason?: string
  risk?: string
  scope?: string
  source?: string
  status?: string
  title?: string
  updated_at?: string
}

interface CockpitImprovementSuggestion {
  benefit?: string
  classification?: 'Bygg nu' | 'Förbered' | 'Utforska' | string
  created_at?: string
  gate_required?: boolean
  id?: string
  next_step?: string
  owner?: string
  risk?: string
  source?: string
  status?: string
  title?: string
  updated_at?: string
  why?: string
}

interface CockpitImprovementHistory {
  action?: string
  actor?: string
  at?: string
  classification?: string
  id?: string
  reason?: string | null
  status_after?: string
  suggestion_id?: string
  title?: string
}

interface CockpitImprovements {
  history?: CockpitImprovementHistory[]
  rules?: {
    cron_history_rule?: string
    external_writes_allowed?: boolean
    labels?: string[]
    local_only?: boolean
    tabs?: string[]
  }
  suggestions?: CockpitImprovementSuggestion[]
  updated_at?: string
  version?: number
}

interface CockpitImprovementActionResponse {
  action?: string
  improvements?: CockpitImprovements
  safety?: Record<string, boolean>
  status?: string
  suggestion_id?: string
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
  graphify?: CockpitGraphifyLane
  improvements?: CockpitImprovements
  jarvis_todo?: CockpitTodoItem[]
  missing?: CockpitSource[]
  safety?: Record<string, boolean>
  sources?: CockpitSource[]
  status?: string
  status_cards?: CockpitStatusCard[]
  updated_at?: string
}

interface JarvisCockpitLocalReportResponse {
  report_path?: string
  safety?: Record<string, boolean>
  status?: string
  updated_at?: string
}

interface CockpitGraphifyLane {
  command?: string
  communities?: number
  edges?: number
  graph_path?: string
  html_path?: string
  latest_note_path?: string
  latest_note_title?: string
  latest_note_updated_at?: string
  nodes?: number
  notes_dir?: string
  report_path?: string
  safety?: Record<string, boolean>
  scope?: string
  state?: string
  status?: string
  token_reduction?: string
  updated_at?: string
}

interface CockpitGraphifyQueryResponse {
  graph_path?: string
  mode?: string
  output?: string
  safety?: Record<string, boolean>
  status?: string
  stderr?: string
}

interface CockpitGraphifyNoteResponse {
  action?: string
  note_path?: string
  note_title?: string
  notes_dir?: string
  safety?: Record<string, boolean>
  status?: string
  updated_at?: string
}

interface CockpitGraphifyInsight {
  id: string
  mode?: string
  output: string
  question?: string
  title: string
}

const sourceLabels: Record<string, string> = {
  'alfred-dispatch-operator-pack': 'Dispatch operator pointer',
  gates: 'Gates directory',
  'hermes-desktop-pinned-session-routing': 'Pinned chat routing',
  'jarvis-actions': 'Jarvis To Do',
  'jarvis-cockpit-graphify': 'Graphify project graph',
  'jarvis-cockpit-improvements': 'Cockpit förbättringar',
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
const improvementLabels = ['Bygg nu', 'Förbered', 'Utforska']
const improvementActionLabels: Record<string, string> = {
  dismiss: 'Avfärdad',
  park: 'Parkerad',
  restore: 'Tillbakalagd',
  run: 'Körd'
}

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

const healthTone = (statusCards: CockpitStatusCard[], sources: CockpitSource[], gates: CockpitGate[], safety: Record<string, boolean>) => {
  const unsafe = Object.entries(safety).some(([key, value]) => key === 'read_only' ? value !== true : forbiddenSafetyKeys.has(key) && value !== false)
  const errors = statusCards.filter(card => card.state === 'error').length + sources.filter(source => source.state === 'error').length
  const warnings = statusCards.filter(card => ['attention', 'waiting', 'missing'].includes(card.state || '')).length + sources.filter(source => ['missing', 'waiting'].includes(source.state || '')).length
  const waitingGates = gates.filter(gate => gate.status === 'waiting').length

  if (unsafe || errors) {
    return { label: 'Röd', state: 'error', summary: 'Cockpit ser en trasig eller osäker lokal signal.', waitingGates, warnings, errors }
  }
  if (warnings || waitingGates) {
    return { label: 'Gul', state: 'attention', summary: 'Cockpit fungerar, men något behöver koll eller beslut.', waitingGates, warnings, errors }
  }
  return { label: 'Grön', state: 'ok', summary: 'Jarvis Cockpit ser frisk ut utifrån lokala, säkra källor.', waitingGates, warnings, errors }
}

const sourceLabel = (source?: CockpitSource | string) => {
  const id = typeof source === 'string' ? source : source?.id

  return (id && sourceLabels[id]) || id || 'Local source'
}

export function JarvisCockpitView() {
  const [data, setData] = useState<JarvisCockpitResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [reportReceipt, setReportReceipt] = useState<JarvisCockpitLocalReportResponse | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [reporting, setReporting] = useState(false)
  const [improvementActionError, setImprovementActionError] = useState<string | null>(null)
  const [improvementActionReceipt, setImprovementActionReceipt] = useState<string | null>(null)
  const [improvementActionLoading, setImprovementActionLoading] = useState<string | null>(null)
  const [improvementTab, setImprovementTab] = useState<'active' | 'history' | 'parked'>('active')

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

  const generateLocalReport = async () => {
    setReporting(true)
    setReportError(null)
    setReportReceipt(null)

    try {
      const response = await window.hermesDesktop.api<JarvisCockpitLocalReportResponse>({
        path: '/api/jarvis/cockpit/local-report',
        method: 'POST'
      })
      setReportReceipt(response)
    } catch (err) {
      setReportError(err instanceof Error ? err.message : 'Could not generate local Cockpit report')
    } finally {
      setReporting(false)
    }
  }

  const applyImprovementAction = async (suggestion: CockpitImprovementSuggestion, action: 'dismiss' | 'park' | 'restore' | 'run') => {
    if (!suggestion.id) {
      return
    }
    setImprovementActionError(null)
    setImprovementActionReceipt(null)
    setImprovementActionLoading(`${suggestion.id}:${action}`)

    try {
      const response = await window.hermesDesktop.api<CockpitImprovementActionResponse>({
        path: '/api/jarvis/cockpit/improvements/action',
        method: 'POST',
        body: { suggestion_id: suggestion.id, action, actor: 'Tobias' }
      })
      setData(previous => (previous ? { ...previous, improvements: response.improvements || previous.improvements } : previous))
      setImprovementActionReceipt(`${improvementActionLabels[action] || action}: ${suggestion.title || suggestion.id}`)
    } catch (err) {
      setImprovementActionError(err instanceof Error ? err.message : 'Could not update improvement suggestion')
    } finally {
      setImprovementActionLoading(null)
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
  const graphify = data?.graphify
  const sources = data?.sources || []
  const missing = data?.missing || []
  const jarvisHealth = healthTone(statusCards, sources, gates, data?.safety || {})
  const improvementSuggestions = data?.improvements?.suggestions || []
  const improvementHistory = data?.improvements?.history || []
  const activeImprovements = improvementSuggestions.filter(item => (item.status || 'active') === 'active' || item.status === 'running')
  const parkedImprovements = improvementSuggestions.filter(item => item.status === 'parked')

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
              disabled={reporting || !data}
              onClick={() => void generateLocalReport()}
              type="button"
            >
              <Codicon className={cn('size-4', reporting && 'animate-pulse')} name="file-media" />
              {reporting ? 'Generating report' : 'Generate local report'}
            </button>
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

        {reportError && (
          <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
            {reportError}
          </div>
        )}

        {reportReceipt && (
          <div className="mt-4 rounded-xl border border-emerald-500/25 bg-emerald-500/10 p-4 text-sm text-emerald-100">
            Local report created: <span className="text-emerald-200">{reportReceipt.report_path || 'local Cockpit report'}</span>
            {reportReceipt.safety?.external_writes === false ? <span className="ml-2 text-emerald-300">External writes: NO</span> : null}
          </div>
        )}

        {improvementActionError && (
          <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
            {improvementActionError}
          </div>
        )}

        {improvementActionReceipt && (
          <div className="mt-4 rounded-xl border border-emerald-500/25 bg-emerald-500/10 p-4 text-sm text-emerald-100">
            Lokal förbättringshistorik uppdaterad: <span className="text-emerald-200">{improvementActionReceipt}</span>
          </div>
        )}

        {!data && !error ? <CockpitSkeleton /> : null}

        {data ? (
          <main className="grid gap-5 pt-5">
            <section className="grid gap-3 lg:grid-cols-5">
              <MetricCard label="Status" tone={data.status === 'ok' ? 'ok' : 'muted'} value={data.status || 'unknown'} />
              <MetricCard detail="lokala åtgärder" label="Jarvis To Do" value={String(todoItems.length)} />
              <MetricCard detail="lokala gates" label="Beslut" value={String(gates.filter(gate => gate.status === 'waiting').length)} />
              <MetricCard detail="aktuella förbättringar" label="Förbättringar" value={String(activeImprovements.length)} />
              <MetricCard detail="rapporterat, inte fatal" label="Saknade källor" tone={missing.length ? 'attention' : 'ok'} value={String(missing.length)} />
            </section>

            <section className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(21rem,0.8fr)]">
              <div className="grid gap-5">
                <Panel icon="pulse" title="Hermes / Jarvis hälsa">
                  <JarvisHealthSummary health={jarvisHealth} missing={missing.length} statusCards={statusCards} />
                </Panel>

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

                <Panel icon="lock" title="Gates / Beslut">
                  {gates.length ? (
                    <div className="grid gap-2">
                      {gates.map(gate => (
                        <GateRow gate={gate} key={gate.id || gate.title} />
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="Inga lokala gates väntar. Saknade gate-mappar visas under källhälsa utan att stoppa Cockpit." />
                  )}
                </Panel>

                <Panel icon="graph" title="Graphify project graph">
                  <GraphifyLane graphify={graphify} />
                </Panel>

                <Panel icon="sparkle" title="Förbättringar">
                  <ImprovementBoard
                    activeItems={activeImprovements}
                    history={improvementHistory}
                    loadingKey={improvementActionLoading}
                    onAction={(suggestion, action) => void applyImprovementAction(suggestion, action)}
                    parkedItems={parkedImprovements}
                    selectedTab={improvementTab}
                    setSelectedTab={setImprovementTab}
                  />
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

function JarvisHealthSummary({
  health,
  missing,
  statusCards
}: {
  health: ReturnType<typeof healthTone>
  missing: number
  statusCards: CockpitStatusCard[]
}) {
  return (
    <div className="grid gap-3 md:grid-cols-[12rem_minmax(0,1fr)]">
      <div className={cn('rounded-xl border p-4', stateTone(health.state))}>
        <div className="text-xs uppercase tracking-[0.16em] opacity-80">Trafikljus</div>
        <div className="mt-2 text-3xl font-semibold tracking-[-0.05em]">{health.label}</div>
        <div className="mt-1 text-xs opacity-80">{health.summary}</div>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <InfoBlock label="Beslut väntar" value={`${health.waitingGates} lokala gate(s)`} />
        <InfoBlock label="Källor saknas" value={`${missing} rapporterade, inte fatal`} />
        <InfoBlock label="Varningar" value={`${health.warnings} gul signal(er)`} />
        <InfoBlock label="Fel" value={`${health.errors} röd signal(er)`} />
      </div>
      {statusCards.length ? (
        <div className="md:col-span-2 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 text-xs text-(--ui-text-secondary)">
          Hälsan beräknas från lokala statuskort, source health, safety receipt och gates. Ingen Graph, Blikk, mail, secrets, cron- eller gateway-mutation körs här.
        </div>
      ) : null}
    </div>
  )
}

function GraphifyLane({ graphify }: { graphify?: CockpitGraphifyLane }) {
  const [query, setQuery] = useState('')
  const [source, setSource] = useState('')
  const [target, setTarget] = useState('')
  const [node, setNode] = useState('')
  const [busyMode, setBusyMode] = useState<string | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryResult, setQueryResult] = useState<CockpitGraphifyQueryResponse | null>(null)
  const [copied, setCopied] = useState(false)
  const [insights, setInsights] = useState<CockpitGraphifyInsight[]>([])
  const [noteError, setNoteError] = useState<string | null>(null)
  const [noteReceipt, setNoteReceipt] = useState<string | null>(null)
  const [latestNote, setLatestNote] = useState<{ path?: string; title?: string; updated_at?: string } | null>(null)
  const [savingNoteId, setSavingNoteId] = useState<string | null>(null)

  useEffect(() => {
    setLatestNote(graphify?.latest_note_path ? {
      path: graphify.latest_note_path,
      title: graphify.latest_note_title || 'Latest Graphify note',
      updated_at: graphify.latest_note_updated_at
    } : null)
  }, [graphify?.latest_note_path, graphify?.latest_note_title, graphify?.latest_note_updated_at])

  if (!graphify || graphify.state === 'missing') {
    return <EmptyState text="No local Graphify manifest yet. Kör structural-only graph för att fylla projektkartan." />
  }

  const unsafe = graphify.safety
    ? graphify.safety.external_writes || graphify.safety.semantic_llm || graphify.safety.hooks || graphify.safety.mcp || graphify.safety.watch || graphify.safety.secrets_found
    : true

  const insightTitle = (mode: string, body: Record<string, unknown>) => {
    if (mode === 'affected') {
      return `Affected: ${String(body.node || body.question || 'node')}`
    }
    if (mode === 'path') {
      return `Path: ${String(body.source || 'source')} → ${String(body.target || 'target')}`
    }
    return `Query: ${String(body.question || 'Graphify insight')}`
  }

  const addInsight = (mode: string, body: Record<string, unknown>, response: CockpitGraphifyQueryResponse) => {
    if (!response.output) {
      return
    }
    const insight: CockpitGraphifyInsight = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      mode: response.mode || mode,
      output: response.output,
      question: String(body.question || body.node || `${body.source || ''} ${body.target || ''}`).trim(),
      title: insightTitle(response.mode || mode, body)
    }
    setInsights(previous => [insight, ...previous].slice(0, 3))
  }

  const saveInsightNote = async (insight: CockpitGraphifyInsight) => {
    setSavingNoteId(insight.id)
    setNoteError(null)
    setNoteReceipt(null)
    try {
      const response = await window.hermesDesktop.api<CockpitGraphifyNoteResponse>({
        path: '/api/jarvis/cockpit/graphify/note',
        method: 'POST',
        body: {
          title: insight.title,
          mode: insight.mode,
          question: insight.question,
          output: insight.output
        }
      })
      setNoteReceipt(response.note_path || 'local Graphify note')
      if (response.note_path) {
        setLatestNote({
          path: response.note_path,
          title: response.note_title || insight.title,
          updated_at: response.updated_at
        })
      }
    } catch (err) {
      setNoteError(err instanceof Error ? err.message : 'Could not save local Graphify note')
    } finally {
      setSavingNoteId(null)
    }
  }

  const runHelper = async (mode: 'affected' | 'path' | 'query', overrideBody?: Record<string, unknown>) => {
    setBusyMode(mode)
    setQueryError(null)
    setNoteError(null)
    setQueryResult(null)
    setCopied(false)
    try {
      const body = overrideBody || (mode === 'query'
        ? { mode, question: query, budget: 1200 }
        : mode === 'path'
          ? { mode, source, target }
          : { mode, node, depth: 2 })
      const response = await window.hermesDesktop.api<CockpitGraphifyQueryResponse>({
        path: '/api/jarvis/cockpit/graphify/query',
        method: 'POST',
        body
      })
      setQueryResult(response)
      addInsight(mode, body, response)
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : 'Graphify helper failed')
    } finally {
      setBusyMode(null)
    }
  }

  const runPreset = (preset: { mode: 'affected' | 'path' | 'query'; label: string; node?: string; question?: string; source?: string; target?: string }) => {
    if (preset.mode === 'query' && preset.question) {
      setQuery(preset.question)
      void runHelper('query', { mode: 'query', question: preset.question, budget: 1200 })
    } else if (preset.mode === 'path' && preset.source && preset.target) {
      setSource(preset.source)
      setTarget(preset.target)
      void runHelper('path', { mode: 'path', source: preset.source, target: preset.target })
    } else if (preset.mode === 'affected' && preset.node) {
      setNode(preset.node)
      void runHelper('affected', { mode: 'affected', node: preset.node, depth: 2 })
    }
  }

  const copyResult = async () => {
    if (!queryResult?.output || !navigator.clipboard?.writeText) {
      return
    }
    await navigator.clipboard.writeText(queryResult.output)
    setCopied(true)
  }

  const presets = [
    { label: 'What affects Cockpit health?', mode: 'affected' as const, node: 'healthTone' },
    { label: 'API to UI path', mode: 'path' as const, source: 'get_jarvis_cockpit', target: 'JarvisCockpitView' },
    { label: 'Graphify lane map', mode: 'query' as const, question: 'GraphifyLane JarvisCockpitView' }
  ]

  return (
    <div className="grid gap-3">
      <div className="grid gap-2 sm:grid-cols-4">
        <InfoBlock label="Nodes" value={String(graphify.nodes ?? '—')} />
        <InfoBlock label="Edges" value={String(graphify.edges ?? '—')} />
        <InfoBlock label="Communities" value={String(graphify.communities ?? '—')} />
        <InfoBlock label="Token reduction" value={graphify.token_reduction || '—'} />
      </div>
      <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 text-sm leading-6 text-(--ui-text-secondary)">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <span className="font-medium text-foreground">{graphify.nodes ?? 0} nodes / {graphify.edges ?? 0} edges</span>
            <span className="ml-2 text-(--ui-text-tertiary)">{graphify.scope || 'Local project graph'}</span>
          </div>
          <StateBadge state={unsafe ? 'attention' : 'ok'} />
        </div>
        <div className="mt-2 text-xs text-(--ui-text-tertiary)">
          Policy: local structural graph. semantic LLM, hooks, MCP and watch remain gated.
        </div>
        {graphify.command && <code className="mt-2 block overflow-hidden text-ellipsis rounded-md bg-black/20 px-2 py-1 text-xs text-(--ui-text-tertiary)">{graphify.command}</code>}
      </div>
      <div className="grid gap-3 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-medium">Local graph helpers</h3>
            <p className="mt-1 text-xs text-(--ui-text-tertiary)">Runs query/path/affected against this approved local graph only. Query log disabled.</p>
          </div>
          <StateBadge state={unsafe ? 'attention' : 'ok'} />
        </div>
        <div className="flex flex-wrap gap-2">
          {presets.map(preset => (
            <button
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-emerald-500/25 bg-emerald-500/8 px-3 text-xs font-medium text-emerald-100 hover:bg-emerald-500/15 disabled:opacity-60"
              disabled={unsafe || busyMode !== null}
              key={preset.label}
              onClick={() => runPreset(preset)}
              type="button"
            >
              <Codicon className="size-3.5" name={preset.mode === 'path' ? 'type-hierarchy-sub' : preset.mode === 'affected' ? 'target' : 'search'} />
              {preset.label}
            </button>
          ))}
        </div>
        <div className="grid gap-2 lg:grid-cols-3">
          <label className="grid gap-1 text-xs text-(--ui-text-secondary)">
            Graphify query
            <input
              className="h-9 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm text-foreground outline-none focus:border-emerald-500/50"
              onChange={event => setQuery(event.target.value)}
              placeholder="JarvisCockpitView"
              value={query}
            />
            <button className="inline-flex h-8 items-center justify-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-xs font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) disabled:opacity-60" disabled={unsafe || busyMode !== null || !query.trim()} onClick={() => void runHelper('query')} type="button">
              {busyMode === 'query' ? 'Querying' : 'Query'}
            </button>
          </label>
          <div className="grid gap-1 text-xs text-(--ui-text-secondary)">
            Path
            <input aria-label="Graphify path source" className="h-9 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm text-foreground outline-none focus:border-emerald-500/50" onChange={event => setSource(event.target.value)} placeholder="source node" value={source} />
            <input aria-label="Graphify path target" className="h-9 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm text-foreground outline-none focus:border-emerald-500/50" onChange={event => setTarget(event.target.value)} placeholder="target node" value={target} />
            <button className="inline-flex h-8 items-center justify-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-xs font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) disabled:opacity-60" disabled={unsafe || busyMode !== null || !source.trim() || !target.trim()} onClick={() => void runHelper('path')} type="button">
              {busyMode === 'path' ? 'Finding path' : 'Path'}
            </button>
          </div>
          <label className="grid gap-1 text-xs text-(--ui-text-secondary)">
            Affected node
            <input
              className="h-9 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm text-foreground outline-none focus:border-emerald-500/50"
              onChange={event => setNode(event.target.value)}
              placeholder="healthTone"
              value={node}
            />
            <button className="inline-flex h-8 items-center justify-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-xs font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) disabled:opacity-60" disabled={unsafe || busyMode !== null || !node.trim()} onClick={() => void runHelper('affected')} type="button">
              {busyMode === 'affected' ? 'Checking' : 'Affected'}
            </button>
          </label>
        </div>
        {queryError && <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-xs text-red-200">{queryError}</div>}
        {queryResult?.output && (
          <div className="grid gap-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1.5">
                <StateBadge state={queryResult.status || 'ok'} />
                {queryResult.mode ? <StateBadge state={queryResult.mode} /> : null}
                {queryResult.safety?.query_log === false ? <StateBadge state="query log off" /> : null}
              </div>
              <button className="inline-flex h-7 items-center gap-1.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-2 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)" onClick={() => void copyResult()} type="button">
                <Codicon className="size-3.5" name="copy" />
                {copied ? 'Copied' : 'Copy result'}
              </button>
            </div>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-lg border border-(--ui-stroke-tertiary) bg-black/25 p-3 text-xs leading-5 text-(--ui-text-secondary)">{queryResult.output}</pre>
          </div>
        )}
        {noteError && <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-xs text-red-200">{noteError}</div>}
        {noteReceipt && <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/10 p-2 text-xs text-emerald-100">Saved local Graphify note: <span className="text-emerald-200">{noteReceipt}</span></div>}
        <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)/35 p-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary)">Local notes</div>
              {latestNote?.path ? (
                <div className="mt-2 grid gap-1">
                  <div className="text-sm font-medium text-foreground">{latestNote.title || 'Latest Graphify note'}</div>
                  <div className="text-xs text-(--ui-text-tertiary)">{niceDate(latestNote.updated_at) || 'Saved locally'} · {latestNote.path}</div>
                </div>
              ) : (
                <p className="mt-2 text-xs leading-5 text-(--ui-text-tertiary)">No saved Graphify notes yet. Save an insight to create a local Cockpit note.</p>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {latestNote?.path ? (
                <button className="inline-flex h-8 items-center gap-1.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)" onClick={() => void window.hermesDesktop.openExternal(latestNote.path!)} type="button">
                  <Codicon className="size-3.5" name="note" />
                  Open latest note
                </button>
              ) : null}
              {graphify.notes_dir ? (
                <button className="inline-flex h-8 items-center gap-1.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)" onClick={() => void window.hermesDesktop.openExternal(graphify.notes_dir!)} type="button">
                  <Codicon className="size-3.5" name="folder-opened" />
                  Open notes folder
                </button>
              ) : null}
            </div>
          </div>
        </div>
        {insights.length ? (
          <div className="grid gap-2">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary)">Latest insights</div>
            {insights.map(insight => (
              <article className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)/45 p-2" key={insight.id}>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <h4 className="text-xs font-medium text-foreground">{insight.title}</h4>
                      <StateBadge state={insight.mode || 'query'} />
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs leading-5 text-(--ui-text-tertiary)">{insight.output}</p>
                  </div>
                  <button
                    className="inline-flex h-7 items-center gap-1.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-2 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) disabled:opacity-60"
                    disabled={savingNoteId !== null}
                    onClick={() => void saveInsightNote(insight)}
                    type="button"
                  >
                    <Codicon className="size-3.5" name="save" />
                    {savingNoteId === insight.id ? 'Saving note' : 'Save as local Cockpit note'}
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : null}
      </div>
      <div className="flex flex-wrap gap-2">
        {graphify.html_path && (
          <button className="inline-flex h-8 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground" onClick={() => void window.hermesDesktop.openExternal(graphify.html_path!)} type="button">
            <Codicon className="size-4" name="graph" />
            Open graph
          </button>
        )}
        {graphify.report_path && (
          <button className="inline-flex h-8 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 text-sm font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground" onClick={() => void window.hermesDesktop.openExternal(graphify.report_path!)} type="button">
            <Codicon className="size-4" name="file-media" />
            Open report
          </button>
        )}
      </div>
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
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-sm font-medium">{gate.title || gate.id || 'Gate'}</h3>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{gate.decision || gate.reason || gate.source || 'Lokalt beslut som behöver Tobias ställningstagande.'}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <StateBadge state={gate.status || 'waiting'} />
          {gate.risk ? <StateBadge state={gate.risk} /> : null}
        </div>
      </div>
      <div className="mt-3 grid gap-2 text-xs text-(--ui-text-secondary) md:grid-cols-3">
        {gate.recommendation && <InfoBlock label="Rekommendation" value={gate.recommendation} />}
        {gate.scope && <InfoBlock label="Scope" value={gate.scope} />}
        {gate.default_action && <InfoBlock label="Default" value={gate.default_action} />}
      </div>
      <div className="mt-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)/45 p-2 text-xs text-(--ui-text-tertiary)">
        Gate v1 är ett beslutsunderlag: ett ja betyder bara det scope som står här. Externa writes, secrets, Microsoft/Blikk/mail/NUC/cron kräver fortfarande separat exakt gate.
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-xs text-(--ui-text-tertiary)">
        {gate.owner && <span>Owner: {gate.owner}</span>}
        {gate.source && <span>Source: {gate.source}</span>}
        {gate.updated_at && <span>Updated {niceDate(gate.updated_at)}</span>}
      </div>
    </article>
  )
}

function ImprovementBoard({
  activeItems,
  history,
  loadingKey,
  onAction,
  parkedItems,
  selectedTab,
  setSelectedTab
}: {
  activeItems: CockpitImprovementSuggestion[]
  history: CockpitImprovementHistory[]
  loadingKey: string | null
  onAction: (suggestion: CockpitImprovementSuggestion, action: 'dismiss' | 'park' | 'restore' | 'run') => void
  parkedItems: CockpitImprovementSuggestion[]
  selectedTab: 'active' | 'history' | 'parked'
  setSelectedTab: (tab: 'active' | 'history' | 'parked') => void
}) {
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-2">
        <TabButton active={selectedTab === 'active'} label={`Aktuella (${activeItems.length})`} onClick={() => setSelectedTab('active')} />
        <TabButton active={selectedTab === 'parked'} label={`Parkerad (${parkedItems.length})`} onClick={() => setSelectedTab('parked')} />
        <TabButton active={selectedTab === 'history'} label={`Historik (${history.length})`} onClick={() => setSelectedTab('history')} />
      </div>

      {selectedTab === 'active' ? (
        <div className="grid gap-4">
          {improvementLabels.map(label => {
            const items = activeItems.filter(item => item.classification === label)
            return (
              <div className="grid gap-2" key={label}>
                <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary)">{label}</h3>
                {items.length ? (
                  items.map(item => (
                    <ImprovementCard
                      item={item}
                      key={item.id || item.title}
                      loadingKey={loadingKey}
                      onAction={onAction}
                      variant="active"
                    />
                  ))
                ) : (
                  <EmptyState text={`Inga aktuella förslag under ${label}.`} />
                )}
              </div>
            )
          })}
        </div>
      ) : null}

      {selectedTab === 'parked' ? (
        parkedItems.length ? (
          <div className="grid gap-2">
            {parkedItems.map(item => (
              <ImprovementCard item={item} key={item.id || item.title} loadingKey={loadingKey} onAction={onAction} variant="parked" />
            ))}
          </div>
        ) : (
          <EmptyState text="Inga parkerade förslag ännu." />
        )
      ) : null}

      {selectedTab === 'history' ? <ImprovementHistoryList history={history} /> : null}
    </div>
  )
}

function TabButton({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      className={cn(
        'inline-flex h-8 items-center rounded-md border px-3 text-xs font-medium',
        active
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground'
      )}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}

function ImprovementCard({
  item,
  loadingKey,
  onAction,
  variant
}: {
  item: CockpitImprovementSuggestion
  loadingKey: string | null
  onAction: (suggestion: CockpitImprovementSuggestion, action: 'dismiss' | 'park' | 'restore' | 'run') => void
  variant: 'active' | 'parked'
}) {
  const actionBusy = (action: string) => loadingKey === `${item.id}:${action}`
  const disabled = Boolean(loadingKey)

  return (
    <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-medium">{item.title || item.id}</h3>
            <StateBadge state={item.classification} />
            {item.gate_required ? <StateBadge state="gate" /> : null}
          </div>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{item.why}</p>
        </div>
        <StateBadge state={item.status || 'active'} />
      </div>
      <div className="mt-3 grid gap-2 text-xs text-(--ui-text-secondary) md:grid-cols-3">
        {item.benefit && <InfoBlock label="Nytta" value={item.benefit} />}
        {item.risk && <InfoBlock label="Risk" value={item.risk} />}
        {item.next_step && <InfoBlock label="Nästa steg" value={item.next_step} />}
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <SmallActionButton disabled={disabled} label={actionBusy('run') ? 'Kör…' : 'Kör'} onClick={() => onAction(item, 'run')} />
        {variant === 'parked' ? (
          <SmallActionButton disabled={disabled} label={actionBusy('restore') ? 'Tar tillbaka…' : 'Ta tillbaka'} onClick={() => onAction(item, 'restore')} />
        ) : (
          <SmallActionButton disabled={disabled} label={actionBusy('park') ? 'Parkerar…' : 'Parkera'} onClick={() => onAction(item, 'park')} />
        )}
        <SmallActionButton disabled={disabled} label={actionBusy('dismiss') ? 'Avfärdar…' : 'Avfärda'} onClick={() => onAction(item, 'dismiss')} />
      </div>
    </article>
  )
}

function InfoBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)/45 p-2">
      <div className="text-[0.68rem] uppercase tracking-[0.14em] text-(--ui-text-tertiary)">{label}</div>
      <div className="mt-1 leading-5">{value}</div>
    </div>
  )
}

function SmallActionButton({ disabled, label, onClick }: { disabled?: boolean; label: string; onClick: () => void }) {
  return (
    <button
      className="inline-flex h-7 items-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-2.5 text-xs font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-55"
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}

function ImprovementHistoryList({ history }: { history: CockpitImprovementHistory[] }) {
  if (!history.length) {
    return <EmptyState text="Ingen historik ännu. När Tobias kör, parkerar eller avfärdar ett förslag sparas det här så cronjobbet kan undvika dubbletter." />
  }

  return (
    <div className="grid gap-2">
      {history.slice().reverse().slice(0, 20).map(entry => (
        <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-control-muted-background) p-3 text-sm" key={entry.id || `${entry.suggestion_id}-${entry.at}`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="font-medium">{entry.title || entry.suggestion_id || 'Förslag'}</div>
            <StateBadge state={improvementActionLabels[entry.action || ''] || entry.action || 'history'} />
          </div>
          <div className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">
            {entry.actor || 'Tobias'} · {entry.classification || 'oklassad'} · {entry.at ? niceDate(entry.at) : 'okänd tid'}
          </div>
          {entry.reason ? <div className="mt-1 text-xs text-(--ui-text-tertiary)">Orsak: {entry.reason}</div> : null}
        </article>
      ))}
    </div>
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
