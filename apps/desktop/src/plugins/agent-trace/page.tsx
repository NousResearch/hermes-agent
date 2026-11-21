/**
 * The full AgentTrace page (`/agent-trace`) — header controls, a live stats
 * strip, the execution tree and the inspection drawer. Snapshot hydration
 * (delegation tree, selected-session history + child list) runs through React
 * Query on slow intervals and merges into the store through the same pure
 * helpers the event reducer uses; live frames remain the primary source.
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  compactNumber,
  EmptyState,
  host,
  ScrollArea,
  SearchField,
  SegmentedControl,
  Tip,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect } from 'react'

import { InspectionDrawer } from './drawer'
import { useAgentTrace } from './i18n'
import {
  $selectedStep,
  $trace,
  $traceFilter,
  $tracePaused,
  $traceScope,
  activeSessions,
  activeSubagentCount,
  fetchDelegationStatus,
  fetchSessionHistory,
  fetchSubagentList,
  hydrateHistory,
  mergeDelegationStatus,
  mergeSubagentList,
  runningToolCount,
  sessionById,
  traceKeys
} from './trace-store'
import { TraceTree } from './tree'
import { ContextGauge, fmtDurationS } from './ui'

function StatCell({ label, tip, children }: { label: string; tip: string; children: ReactNode }) {
  return (
    <Tip label={tip}>
      <div className="flex min-w-0 flex-col gap-0.5 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-2.5 py-1.5">
        <span className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">{label}</span>
        <span className="truncate text-sm font-medium tabular-nums text-foreground">{children}</span>
      </div>
    </Tip>
  )
}

export function TracePage() {
  const k = useAgentTrace()
  const state = useValue($trace)
  const scope = useValue($traceScope)
  const filter = useValue($traceFilter)
  const paused = useValue($tracePaused)
  const focusedId = useValue(host.state.focusedSessionId)
  const profile = useValue(host.state.profile)
  const selected = useValue($selectedStep)
  const queryClient = useQueryClient()

  // Profile-wide running tree — the authoritative subagent snapshot.
  const delegation = useQuery({
    queryKey: traceKeys.delegation(profile),
    queryFn: () => fetchDelegationStatus(profile),
    refetchInterval: 10_000,
    retry: false
  })

  useEffect(() => {
    if (delegation.data) {
      $trace.set(mergeDelegationStatus($trace.get(), delegation.data, Date.now() / 1000))
    }
  }, [delegation.data])

  // Focused-session hydration: persisted tool rows + live child snapshots.
  const history = useQuery({
    queryKey: traceKeys.history(profile, focusedId ?? ''),
    queryFn: () => fetchSessionHistory(profile, focusedId as string),
    enabled: Boolean(focusedId),
    refetchInterval: 60_000,
    retry: false
  })

  useEffect(() => {
    if (history.data && focusedId) {
      $trace.set(hydrateHistory($trace.get(), focusedId, history.data, Date.now() / 1000))
    }
  }, [history.data, focusedId])

  const children = useQuery({
    queryKey: traceKeys.subagents(profile, focusedId ?? ''),
    queryFn: () => fetchSubagentList(profile, focusedId as string),
    enabled: Boolean(focusedId),
    refetchInterval: 10_000,
    retry: false
  })

  useEffect(() => {
    if (children.data && focusedId) {
      $trace.set(mergeSubagentList($trace.get(), focusedId, children.data, Date.now() / 1000))
    }
  }, [children.data, focusedId])

  const sessions = scope === 'focused' ? state.sessions.filter(s => s.sessionId === focusedId) : activeSessions(state)
  const statSession = sessionById(state, focusedId)
  const usage = statSession?.usage
  const tokens = (usage?.input ?? 0) + (usage?.output ?? 0)
  const agents = activeSubagentCount(state)
  const tools = runningToolCount(state)

  return (
    <div className="relative flex h-full min-w-0 flex-col">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-(--ui-stroke-tertiary) px-4 py-3">
        <div className="min-w-0">
          <h1 className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <Codicon className="text-primary" name="pulse" size="0.9rem" />
            {k.title}
            <Badge size="xs" variant={paused ? 'warn' : 'success'}>
              {!paused && <StatusDotPulse />}
              {paused ? k.paused : k.live}
            </Badge>
          </h1>
          <p className="mt-0.5 text-[0.6875rem] text-(--ui-text-quaternary)">{k.subtitle}</p>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <SegmentedControl
            onChange={id => $traceScope.set(id)}
            options={[
              { id: 'focused', label: k.scopeFocused },
              { id: 'all', label: k.scopeAll }
            ]}
            sizing="content"
            value={scope}
          />
          <SearchField
            containerClassName="w-48"
            onChange={value => $traceFilter.set(value)}
            placeholder={k.filterPlaceholder}
            value={filter}
            variant="box"
          />
          <Tip label={paused ? k.resume : k.pause}>
            <Button
              aria-label={paused ? k.resume : k.pause}
              onClick={() => $tracePaused.set(!paused)}
              size="icon-sm"
              variant={paused ? 'default' : 'ghost'}
            >
              <Codicon name={paused ? 'debug-start' : 'debug-pause'} size="0.85rem" />
            </Button>
          </Tip>
          <Tip label={k.refresh}>
            <Button
              aria-label={k.refresh}
              onClick={() => void queryClient.invalidateQueries({ queryKey: ['agent-trace'] })}
              size="icon-sm"
              variant="ghost"
            >
              <Codicon name="refresh" size="0.85rem" />
            </Button>
          </Tip>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-2 px-4 py-2.5 sm:grid-cols-3 lg:grid-cols-5">
        <StatCell label={k.statAgents} tip={k.statAgentsTip}>
          {agents}
        </StatCell>
        <StatCell label={k.statTools} tip={k.statToolsTip}>
          {tools}
        </StatCell>
        <StatCell label={k.statContext} tip={k.statContextTip}>
          <ContextGauge max={usage?.context_max} percent={usage?.context_percent} used={usage?.context_used} />
        </StatCell>
        <StatCell label={k.statTokens} tip={k.statTokensTip}>
          {compactNumber(tokens)}
        </StatCell>
        <StatCell label={k.statLatency} tip={k.statLatencyTip}>
          {fmtDurationS(usage?.avg_latency_s)}
        </StatCell>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className={cn('px-4 pb-4', sessions.length === 0 && 'grid h-full place-items-center')}>
          {sessions.length === 0 ? (
            <EmptyState description={k.emptyDescription} title={k.emptyTitle} />
          ) : (
            <TraceTree
              filter={filter}
              onSelect={sel => $selectedStep.set(sel)}
              selected={selected}
              sessions={sessions}
              showProfile={scope === 'all'}
            />
          )}
        </div>
      </ScrollArea>

      <InspectionDrawer />
    </div>
  )
}

/** Tiny live pulse inside the Live badge (decorative). */
function StatusDotPulse() {
  return <span aria-hidden className="inline-block size-1 animate-pulse rounded-full bg-current" />
}
