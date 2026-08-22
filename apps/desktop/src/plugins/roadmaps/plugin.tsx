/**
 * Roadmaps — a bundled plugin port of the disk prototype. A first-class
 * `/roadmaps` orchestration page + sidebar nav row, a pure consumer of the
 * plugin's own `/api/plugins/roadmaps` REST router (PR4) through `ctx.rest`.
 * No new backend, no core edits, no `host.request`.
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Settings ▸ Plugins and registers nothing until the user flips the switch.
 *
 * Layout adapts to the REAL container width (ResizeObserver; thresholds in
 * config): wide (3 columns Thread | view | Inspector), mid (2 columns with a
 * view/Inspector switch), compact (1 column + collapsible Inspector).
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  CopyButton,
  EmptyState,
  ErrorState,
  type HermesPlugin,
  host,
  type RouteContribution,
  ROUTES_AREA,
  ScrollArea,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  Skeleton,
  StatusDot,
  useValue
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { $projectId, $roadmapId, useRoadmapsList, useRoadmapSnapshot } from './api'
import { ID, INSPECTOR_TABS, LAYOUT, TABS } from './config'
import { CopilotBar } from './copilot'
import { activeVersion, errorCopy } from './data'
import { DecisionsView } from './decisions'
import { FilesView } from './files'
import { Inspector } from './inspector'
import { MapView } from './map'
import { MilestonesView } from './milestones'
import { PlanView } from './plan'
import { ScopeBar } from './scope'
import { ThreadView } from './thread'
import type { RoadmapVersion, Scope, SnapshotResponse } from './types'

/** Underline tabs — active = accent underline, no boxes. */
function ViewTabs({ active, onChange }: { active: string; onChange: (id: string) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-4 px-0.5">
      {TABS.map((t) => (
        <button
          className={cn(
            'inline-flex items-center gap-1 border-b-2 px-0.5 pb-1.5 pt-0.5 text-xs transition-colors',
            active === t.id
              ? 'border-(--ui-accent) font-medium text-foreground'
              : 'border-transparent text-(--ui-text-tertiary) hover:text-foreground'
          )}
          key={t.id}
          onClick={() => onChange(t.id)}
          title={t.label}
          type="button"
        >
          <Codicon name={t.codicon} size="0.7rem" />
          <span>{t.label}</span>
        </button>
      ))}
    </div>
  )
}

/** One scrollable grid column; `divider` draws a hairline between columns. */
function GridColumn({ header, divider, children }: { header?: React.ReactNode; divider?: boolean; children: React.ReactNode }) {
  return (
    <div className={cn('flex min-h-0 min-w-0 flex-col gap-1.5', divider && 'border-l border-(--ui-stroke-tertiary) pl-2.5')}>
      {header ?? null}
      <ScrollArea className="min-h-0 flex-1 px-0.5">{children}</ScrollArea>
    </div>
  )
}

/** Mid-mode two-segment switch: active view ↔ Inspector, selection kept. */
function MidPaneSwitch({ activeTab, pane, onPane }: { activeTab: string; pane: string; onPane: (v: string) => void }) {
  const tabMeta = TABS.find((t) => t.id === activeTab)

  const seg = (key: string, codicon: string, label: string) => (
    <button
      className={cn(
        'inline-flex items-center gap-1 rounded-[2px] px-1.5 py-0.5 text-[0.625rem] transition-colors',
        pane === key ? 'bg-(--ui-bg-elevated) font-medium text-foreground' : 'text-(--ui-text-tertiary) hover:text-foreground'
      )}
      key={key}
      onClick={() => onPane(key)}
      title={label}
      type="button"
    >
      <Codicon name={codicon} size="0.65rem" />
      <span>{label}</span>
    </button>
  )

  return (
    <div className="inline-flex items-center gap-0.5 self-start rounded-[3px] bg-(--ui-bg-quaternary) p-0.5">
      {seg('view', tabMeta?.codicon ?? 'milestone', tabMeta?.label ?? activeTab)}
      {seg('inspector', 'info', 'Inspector')}
    </div>
  )
}

/** The content grid — three modes driven by the measured container width. */
function RoadmapsGrid({
  mode,
  activeTab,
  canInspect,
  snapshot,
  version,
  selectedNodeId,
  onSelect,
  scope,
  onMutated,
  compact,
  actor,
  setActor,
  inspectorOpen
}: {
  mode: string
  activeTab: string
  canInspect: boolean
  snapshot: SnapshotResponse | null | undefined
  version: RoadmapVersion | null
  selectedNodeId: string
  onSelect: (id: string) => void
  scope: Scope | null
  onMutated: () => void
  compact: boolean
  actor: string
  setActor: (v: string) => void
  inspectorOpen: boolean
}) {
  const [midPane, setMidPane] = useState('inspector')
  const pane = canInspect ? midPane : 'view'

  const thread = <ThreadView compact={compact} dense onSelect={onSelect} selectedId={selectedNodeId} version={version} />
  const view = <ActiveView activeTab={activeTab} actor={actor} compact={compact} onMutated={onMutated} onSelect={onSelect} scope={scope} selectedId={selectedNodeId} snapshot={snapshot} version={version} />

  const inspector = (
    <Inspector
      actor={actor}
      compact={compact}
      nodeId={selectedNodeId}
      onMutated={onMutated}
      onSelect={onSelect}
      scope={scope}
      setActor={setActor}
      snapshot={snapshot}
      version={version}
    />
  )

  if (mode === 'wide') {
    return (
      <div className="grid min-h-0 flex-1 gap-2.5" style={{ gridTemplateColumns: `minmax(0, 1.1fr) minmax(0, 1fr) ${LAYOUT.inspectorWidth}px` }}>
        <GridColumn>{thread}</GridColumn>
        <GridColumn divider>{view}</GridColumn>
        <GridColumn divider>{inspector}</GridColumn>
      </div>
    )
  }

  if (mode === 'mid') {
    return (
      <div className="grid min-h-0 flex-1 grid-cols-2 gap-2.5">
        <GridColumn>{thread}</GridColumn>
        <GridColumn divider header={canInspect ? <MidPaneSwitch activeTab={activeTab} onPane={setMidPane} pane={pane} /> : null}>
          {pane === 'inspector' ? inspector : view}
        </GridColumn>
      </div>
    )
  }

  // compact — single column; the Inspector is toggled from the tab row.
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <ScrollArea className="min-h-0 flex-1 px-0.5">{view}</ScrollArea>
      {compact && canInspect && inspectorOpen ? (
        <div className="flex min-h-0 flex-1 flex-col border-t border-(--ui-stroke-tertiary) pt-1.5">
          <ScrollArea className="min-h-0 flex-1 px-0.5">{inspector}</ScrollArea>
        </div>
      ) : null}
    </div>
  )
}

/** Tab content dispatch. */
function ActiveView({
  activeTab,
  snapshot,
  version,
  selectedId,
  onSelect,
  compact,
  scope,
  actor,
  onMutated
}: {
  activeTab: string
  snapshot: SnapshotResponse | null | undefined
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
  compact: boolean
  scope: Scope | null
  actor: string
  onMutated: () => void
}) {
  if (activeTab === 'thread') {
    return <ThreadView compact={compact} dense onSelect={onSelect} selectedId={selectedId} version={version} />
  }

  if (activeTab === 'map') {
    return <MapView onSelect={onSelect} selectedId={selectedId} version={version} />
  }

  if (activeTab === 'plan') {
    return <PlanView actor={actor} onMutated={onMutated} scope={scope} snapshot={snapshot} />
  }

  if (activeTab === 'milestones') {
    return <MilestonesView compact={compact} onSelect={onSelect} selectedId={selectedId} version={version} />
  }

  if (activeTab === 'decisions') {
    return <DecisionsView />
  }

  return <FilesView />
}

/** Layout mode measured on the REAL container via ResizeObserver. */
function useLayoutMode(initialWidth: number) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [containerWidth, setContainerWidth] = useState(initialWidth)
  useEffect(() => {
    const el = containerRef.current
    const RO = globalThis.ResizeObserver

    if (!el || typeof RO !== 'function') {return}

    const ro = new RO((entries) => {
      for (const entry of entries) {setContainerWidth(entry.contentRect.width)}
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [])
  const width = containerWidth > 0 ? containerWidth : initialWidth
  const mode = width >= LAYOUT.wide ? 'wide' : width >= LAYOUT.compact ? 'mid' : 'compact'

  return { containerRef, mode, compact: mode === 'compact' }
}

/** Node selection with hygiene: reset on scope identity change, drop vanished nodes. */
function useNodeSelection(scopeIdentity: unknown[], version: RoadmapVersion | null) {
  const [selectedNodeId, setSelectedNodeId] = useState('')

  useEffect(() => {
    setSelectedNodeId('')
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scope identity only
  }, scopeIdentity)

  useEffect(() => {
    if (selectedNodeId !== '' && version && !version.nodes.some((n) => n.node_id === selectedNodeId)) {
      setSelectedNodeId('')
    }
  }, [version, selectedNodeId])

  const onSelect = useCallback((nodeId: string) => {
    setSelectedNodeId((cur) => (cur === nodeId ? '' : nodeId))
  }, [])

  return { selectedNodeId, setSelectedNodeId, onSelect }
}

export function RoadmapsPage() {
  const profile = useValue(host.state.profile)
  const viewport = useValue(host.state.viewport)
  const projectId = useValue($projectId)
  const roadmapId = useValue($roadmapId)

  const [activeTab, setActiveTab] = useState('thread')
  const [actor, setActor] = useState('user')
  const [inspectorOpen, setInspectorOpen] = useState(false)

  const { containerRef, mode, compact } = useLayoutMode(viewport?.width ?? 0)

  // Identity guard: profile is displayed read-only; a missing profile is an
  // explicit "not initialized" state — NEVER a silent fallback to 'default'.
  const profileReady = typeof profile === 'string' && profile.trim() !== ''
  const projectReady = projectId.trim() !== ''

  const listQuery = useRoadmapsList(profile, projectId, profileReady && projectReady)

  const roadmapOptions = useMemo(() => {
    const roadmaps = listQuery.data?.roadmaps ?? []

    return roadmaps
      .filter((r) => r.lifecycle_state !== 'archived')
      .sort((a, b) => String(a.title ?? a.roadmap_id).localeCompare(String(b.title ?? b.roadmap_id)))
  }, [listQuery.data])

  // Selection hygiene: drop a roadmap that vanished from the list.
  useEffect(() => {
    if (roadmapId !== '' && projectReady && !roadmapOptions.some((r) => r.roadmap_id === roadmapId)) {
      $roadmapId.set('')
    }
  }, [roadmapOptions, roadmapId, projectReady])

  const onProjectChange = useCallback(
    (v: string) => {
      $projectId.set(v)

      if (v.trim() !== projectId.trim()) {
        $roadmapId.set('')
      }
    },
    [projectId]
  )

  const scopeReady = profileReady && projectReady && roadmapId !== ''
  const snapshotQuery = useRoadmapSnapshot(profile, projectId, roadmapId, scopeReady)
  const snapshot = snapshotQuery.data
  const found = snapshot?.found === true
  const version = useMemo(() => activeVersion(snapshot), [snapshot])

  const { selectedNodeId, setSelectedNodeId, onSelect } = useNodeSelection([profile, projectId, roadmapId], version)

  const reloadSnapshot = useCallback(() => {
    void snapshotQuery.refetch()
  }, [snapshotQuery])

  const scope: Scope | null = scopeReady ? { profile, projectId, roadmapId } : null

  const canInspect = selectedNodeId !== '' && INSPECTOR_TABS.has(activeTab)

  if (!profileReady) {
    return (
      <EmptyState
        description={'No active profile identity is available. Roadmaps refuses to guess a profile (no silent fallback to "default").'}
        title="Profile not initialized"
      />
    )
  }

  const listError = listQuery.isError ? errorCopy(listQuery.error) : null
  const snapshotError = snapshotQuery.isError ? errorCopy(snapshotQuery.error) : null
  const panel = (content: React.ReactNode) => <ScrollArea className="min-h-0 flex-1 px-0.5">{content}</ScrollArea>
  const needsVersion = activeTab === 'thread' || activeTab === 'map' || activeTab === 'milestones'

  let content: React.ReactNode

  if (!scopeReady) {
    content = panel(
      <EmptyState
        description="The Thread, Map, Plan, Milestones, Decisions, and Files views appear once a project and a roadmap are chosen."
        title="Select a project and a roadmap…"
      />
    )
  } else if (snapshotQuery.isLoading) {
    content = panel(<Skeleton className="h-24 w-full" />)
  } else if (snapshotQuery.isError) {
    content = panel(
      <ErrorState
        description={`${snapshotError?.hint ?? ''}${snapshotError?.code != null ? ` (code ${snapshotError.code})` : ''}`}
        title="Snapshot unavailable"
      >
        <Button onClick={() => void snapshotQuery.refetch()} size="xs" type="button" variant="secondary">
          {snapshotError?.code === 5064 ? 'Reload' : 'Retry'}
        </Button>
      </ErrorState>
    )
  } else if (!found) {
    content = panel(
      <EmptyState
        description={`No roadmap found for ${projectId} / ${roadmapId} in profile ${profile}.`}
        title="No roadmap for this scope"
      />
    )
  } else if (needsVersion && !version) {
    content = panel(<EmptyState description="This roadmap has no active version to display." title="No active version" />)
  } else {
    content = (
      <RoadmapsGrid
        activeTab={activeTab}
        actor={actor}
        canInspect={canInspect}
        compact={compact}
        inspectorOpen={inspectorOpen}
        mode={mode}
        onMutated={reloadSnapshot}
        onSelect={onSelect}
        scope={scope}
        selectedNodeId={selectedNodeId}
        setActor={setActor}
        snapshot={snapshot}
        version={version}
      />
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 p-3" ref={containerRef}>
      <ScopeBar
        actor={actor}
        compact={compact}
        onProjectChange={onProjectChange}
        profile={profile}
        projectId={projectId}
        roadmapId={roadmapId}
        roadmapOptions={roadmapOptions}
        setRoadmapId={(v) => {
          $roadmapId.set(v)
          setSelectedNodeId('')
        }}
        setSelectedNodeId={setSelectedNodeId}
      />

      {listError ? (
        <ErrorState
          description={`${listError.hint}${listError.code != null ? ` (code ${listError.code})` : ''}`}
          title="Roadmap list unavailable"
        >
          <Button onClick={() => void listQuery.refetch()} size="xs" type="button" variant="secondary">
            Retry
          </Button>
        </ErrorState>
      ) : projectReady && roadmapOptions.length === 0 && !listQuery.isLoading ? (
        <EmptyState
          description={`Project "${projectId}" has no roadmaps in profile ${profile}. Create one with the + button.`}
          title="No roadmaps for this scope"
        />
      ) : null}

      {found && snapshot?.roadmap ? (
        <div className="flex flex-wrap items-center gap-2 border-b border-(--ui-stroke-tertiary) px-0.5 pb-2">
          <div className="min-w-0 flex-1">
            <div className="truncate text-[0.8125rem] font-medium">{snapshot.roadmap.title || roadmapId}</div>
            {!compact && snapshot.roadmap.purpose ? (
              <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">{snapshot.roadmap.purpose}</div>
            ) : null}
          </div>
          <Badge size="xs" variant="outline">
            <StatusDot tone="good" />
            {snapshot.roadmap.lifecycle_state}
          </Badge>
          <span className="font-mono text-[0.625rem] text-(--ui-text-tertiary)">{`v${snapshot.roadmap.active_version}`}</span>
          <CopyButton
            appearance="icon"
            buttonSize="icon-xs"
            buttonVariant="ghost"
            label="Copy scope"
            text={`${profile} / ${projectId} / ${roadmapId}`}
            title="Copy scope (profile / project / roadmap)"
          />
        </div>
      ) : null}

      {found && snapshot?.roadmap ? <CopilotBar dense={compact} onSelect={onSelect} selectedId={selectedNodeId} version={version} /> : null}

      {scopeReady ? (
        <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-(--ui-stroke-tertiary)">
          <ViewTabs active={activeTab} onChange={setActiveTab} />
          {compact && canInspect ? (
            <Button
              className="gap-1"
              onClick={() => setInspectorOpen((v) => !v)}
              size="xs"
              type="button"
              variant={inspectorOpen ? 'secondary' : 'ghost'}
            >
              <Codicon name="info" size="0.7rem" />
              Details
            </Button>
          ) : null}
        </div>
      ) : null}

      {content}
    </div>
  )
}

const plugin: HermesPlugin = {
  id: ID,
  name: 'Roadmaps',
  description:
    'Project roadmaps — orchestration thread, canonical relation map, versioned plan history, milestones, and a data-driven copilot with versioned manual steering.',
  defaultEnabled: false,
  register(ctx) {
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/roadmaps' } satisfies RouteContribution,
        render: () => <RoadmapsPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 51,
        data: { codicon: 'milestone', label: 'Roadmaps', path: '/roadmaps' } satisfies SidebarNavContribution
      }
    ])
  }
}

export default plugin
