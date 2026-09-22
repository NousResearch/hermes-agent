import { Button, cn, Codicon, useQuery } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { fetchFleetStatus, FLEET_STATUS_KEY } from './api'
import type { FleetRunnerStatus, FleetStatusResponse, FleetTaskStatus } from './types'

interface FleetNode {
  activeLoad: number
  models: string[]
  nodeId: string
  online: boolean
  platform: string
  profiles: number
  projects: string[]
}

// The two production supervisors use stable node IDs. Ignore retired smoke
// runners in the shared registry so the pool summary reflects actual machines.
const PRODUCTION_NODE_IDS = new Set(['mac', 'windows'])

function groupNodes(runners: FleetRunnerStatus[]): FleetNode[] {
  const nodes = new Map<string, FleetNode>()

  for (const runner of runners) {
    if (!PRODUCTION_NODE_IDS.has(runner.node_id)) { continue }

    const existing = nodes.get(runner.node_id) ?? {
      activeLoad: 0,
      models: [],
      nodeId: runner.node_id,
      online: false,
      platform: runner.capability.platform ?? 'unknown',
      profiles: 0,
      projects: []
    }

    existing.activeLoad += runner.active_load
    existing.online ||= runner.online
    existing.profiles += 1
    existing.models.push(...(runner.capability.models ?? []))
    existing.projects.push(...(runner.capability.projects ?? []))
    nodes.set(runner.node_id, existing)
  }

  return [...nodes.values()]
    .map(node => ({
      ...node,
      models: [...new Set(node.models)].sort(),
      projects: [...new Set(node.projects)].sort()
    }))
    .sort((a, b) => Number(b.online) - Number(a.online) || a.nodeId.localeCompare(b.nodeId))
}

function taskLabel(task: FleetTaskStatus): string {
  const owner = task.node_id ? `${task.node_id}${task.runner_profile ? ` · ${task.runner_profile}` : ''}` : 'waiting'

  return `${task.title} · ${owner}`
}

function fleetCopy(status: FleetStatusResponse | undefined): string {
  if (!status?.reachable) {
    return status?.error ? 'Coordinator unavailable' : 'Checking coordinator…'
  }

  const nodes = groupNodes(status.runners)
  const online = nodes.filter(node => node.online).length

  return `${online}/${nodes.length} computers online`
}

export function FleetStatusPanel() {
  const { data, isFetching, refetch } = useQuery({
    queryKey: FLEET_STATUS_KEY,
    queryFn: fetchFleetStatus,
    refetchInterval: 5_000,
    retry: false,
    staleTime: 2_000
  })

  const nodes = useMemo(() => groupNodes(data?.runners ?? []), [data?.runners])
  const activeTasks = useMemo(
    () => (data?.tasks ?? []).filter(task => task.status === 'pending' || task.status === 'running'),
    [data?.tasks]
  )

  if (data && !data.enabled) {
    return null
  }

  return (
    <section
      aria-label="Federated runner pool"
      className="mx-4 mb-2 flex shrink-0 flex-col gap-2 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-quinary) px-3 py-2"
    >
      <div className="flex items-center gap-2 text-[0.6875rem]">
        <Codicon name="server-environment" size="0.8rem" />
        <span className="font-semibold text-foreground">Federated runner pool</span>
        <span
          className={cn(
            'rounded-full px-1.5 py-px text-[0.625rem] tabular-nums',
            data?.reachable ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
          )}
        >
          {fleetCopy(data)}
        </span>
        <Button
          aria-label="Refresh runner pool"
          className="ml-auto"
          disabled={isFetching}
          onClick={() => void refetch()}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.75rem" />
        </Button>
      </div>

      {!data?.reachable ? (
        <p className="text-[0.6875rem] text-(--ui-text-tertiary)">
          {data?.error ?? 'Connecting to the shared coordinator…'}
        </p>
      ) : (
        <>
          <div className="grid gap-1.5 sm:grid-cols-2">
            {nodes.map(node => (
              <div
                className="flex min-w-0 items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-2 py-1.5"
                key={node.nodeId}
              >
                <span
                  className={cn(
                    'size-2 shrink-0 rounded-full',
                    node.online ? 'bg-emerald-400' : 'bg-(--ui-text-quaternary)'
                  )}
                />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[0.6875rem] font-medium text-foreground">{node.nodeId}</div>
                  <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                    {node.online ? node.platform : 'offline'} · {node.profiles} profiles · {node.projects.length}{' '}
                    projects
                    {node.activeLoad > 0 ? ` · ${node.activeLoad} active` : ''}
                  </div>
                </div>
                <span className="max-w-48 truncate text-right text-[0.625rem] text-(--ui-text-quaternary)">
                  {node.models.join(', ') || 'model discovery pending'}
                </span>
              </div>
            ))}
          </div>

          {activeTasks.length > 0 && (
            <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[0.625rem] text-(--ui-text-tertiary)">
              <span className="font-semibold text-(--ui-text-secondary)">Shared queue</span>
              {activeTasks.slice(0, 4).map(task => (
                <span className="max-w-72 truncate" key={task.task_id} title={taskLabel(task)}>
                  {task.status === 'running' ? '●' : '○'} {taskLabel(task)}
                </span>
              ))}
              {activeTasks.length > 4 && <span>+{activeTasks.length - 4} more</span>}
            </div>
          )}
        </>
      )}
    </section>
  )
}
