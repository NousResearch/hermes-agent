/**
 * Roadmaps plugin — Thread view ("what to do now").
 *
 * Actionable nodes (ready / in_progress / blocked, blocked first) with
 * dependency, blocker, and dependant info, plus the implicit critical path.
 */

import { cn, Codicon, EmptyState, StatusDot } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { NODE_TONE } from './config'
import { criticalChain, nodeBlockers, nodeDependants, nodeDepsInfo, nodeLabel, plural, threadNodes } from './data'
import type { RoadmapNode, RoadmapVersion } from './types'
import { NodeStateTag, ProgressBar, SectionTitle } from './ui'

function CriticalChainStrip({
  chain,
  version,
  selectedId,
  onSelect
}: {
  chain: string[]
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
}) {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))
  const ordered = [...chain].reverse() // leaf dependency → deepest dependant

  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5 rounded-[3px] bg-(--ui-bg-quaternary) px-2 py-1 text-[0.625rem]">
      <span className="mr-1 font-medium uppercase tracking-wide text-(--ui-text-tertiary)">Critical path</span>
      {ordered.map((id, i) => (
        <span className="flex items-center gap-1" key={id}>
          {i > 0 ? <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="chevron-right" size="0.6rem" /> : null}
          <button
            className={cn(
              'min-w-0 max-w-44 truncate hover:underline',
              id === selectedId ? 'font-medium text-primary' : 'text-(--ui-text-secondary) hover:text-foreground'
            )}
            onClick={() => onSelect(id)}
            title={nodeLabel(byId.get(id))}
            type="button"
          >
            {nodeLabel(byId.get(id))}
          </button>
        </span>
      ))}
    </div>
  )
}

export function NodeRow({
  node,
  version,
  selected,
  onSelect,
  compact,
  dense
}: {
  node: RoadmapNode
  version: RoadmapVersion | null
  selected: boolean
  onSelect: (id: string) => void
  compact: boolean
  dense: boolean
}) {
  const deps = useMemo(() => nodeDepsInfo(node, version), [node, version])
  const dependants = useMemo(() => nodeDependants(node, version), [node, version])
  const blockers = useMemo(() => nodeBlockers(node, version), [node, version])
  const pendingCount = deps.total - deps.satisfied

  // Dense column row: dot, kind · title, state tag and progress on ONE line.
  if (dense) {
    return (
      <button
        className={cn(
          'group flex w-full items-center gap-2 px-2 py-1 text-left transition-colors',
          selected ? 'bg-primary/[0.06]' : 'hover:bg-(--chrome-action-hover)'
        )}
        onClick={() => onSelect(node.node_id)}
        type="button"
      >
        <StatusDot tone={NODE_TONE[node.state] ?? 'muted'} />
        <span className="min-w-0 flex-1 truncate text-xs font-medium">
          <span className={cn('text-[0.625rem]', selected ? 'text-primary' : 'text-(--ui-text-tertiary)')}>
            {`${node.kind} · `}
          </span>
          {nodeLabel(node)}
        </span>
        <NodeStateTag state={node.state} />
        <ProgressBar value={node.progress} />
      </button>
    )
  }

  return (
    <button
      className={cn(
        'group flex w-full flex-col gap-1 px-2 py-1.5 text-left transition-colors',
        selected ? 'bg-primary/[0.06]' : 'hover:bg-(--chrome-action-hover)'
      )}
      onClick={() => onSelect(node.node_id)}
      type="button"
    >
      <div className="flex items-center gap-2">
        <StatusDot tone={NODE_TONE[node.state] ?? 'muted'} />
        <span className="min-w-0 flex-1 truncate text-xs font-medium">
          <span className={cn('text-[0.625rem]', selected ? 'text-primary' : 'text-(--ui-text-tertiary)')}>
            {`${node.kind} · `}
          </span>
          {nodeLabel(node)}
        </span>
        {!compact ? <NodeStateTag state={node.state} /> : null}
      </div>
      {!compact ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-3.5">
          {deps.total > 0 ? (
            pendingCount === 0 ? (
              <span className="inline-flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
                <Codicon name="check" size="0.65rem" />
                {`${deps.satisfied}/${deps.total} deps satisfied`}
              </span>
            ) : (
              <span
                className="inline-flex items-center gap-1 text-[0.625rem] text-amber-500/90 dark:text-amber-300/90"
                title={deps.deps
                  .filter((d) => !d.satisfied)
                  .map((d) => nodeLabel(d.target) || d.targetId)
                  .join(', ')}
              >
                <Codicon name="hourglass" size="0.65rem" />
                {plural(pendingCount, 'pending dep')}
              </span>
            )
          ) : null}
          {blockers.length > 0 ? (
            <span className="inline-flex items-center gap-1 text-[0.625rem] text-destructive">
              <Codicon name="debug-disconnect" size="0.65rem" />
              {plural(blockers.length, 'blocker')}
            </span>
          ) : null}
          {dependants.length > 0 ? (
            <span className="inline-flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
              <Codicon name="arrow-down" size="0.65rem" />
              {plural(dependants.length, 'dependant')}
            </span>
          ) : null}
          <span className="ml-auto flex shrink-0 items-center gap-2">
            <ProgressBar value={node.progress} />
            {node.owner_agent ? (
              <span className="inline-flex min-w-0 max-w-32 items-center gap-1 truncate text-[0.625rem] text-(--ui-text-tertiary)">
                <Codicon name="person" size="0.65rem" />
                <span className="truncate">{node.owner_agent}</span>
              </span>
            ) : null}
          </span>
        </div>
      ) : null}
      {node.state === 'blocked' && node.block_reason ? (
        <div className="flex items-start gap-1 pl-3.5 text-[0.625rem] text-destructive">
          <Codicon className="mt-px shrink-0" name="debug-disconnect" size="0.7rem" />
          <span className="whitespace-pre-wrap break-words">{node.block_reason}</span>
        </div>
      ) : null}
    </button>
  )
}

export function ThreadView({
  version,
  selectedId,
  onSelect,
  compact,
  dense
}: {
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
  compact: boolean
  dense: boolean
}) {
  const nodes = threadNodes(version)
  const chain = useMemo(() => criticalChain(version), [version])

  if (nodes.length === 0) {
    return (
      <EmptyState
        description="No ready, in_progress, or blocked nodes in the active version of this roadmap."
        title="Nothing in flight"
      />
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      {chain.length > 1 ? (
        <CriticalChainStrip chain={chain} onSelect={onSelect} selectedId={selectedId} version={version} />
      ) : null}
      <SectionTitle right={<span className="tabular-nums text-(--ui-text-quaternary)">{plural(nodes.length, 'node')}</span>}>
        Thread
      </SectionTitle>
      <div className="flex flex-col divide-y divide-(--ui-stroke-tertiary)">
        {nodes.map((n) => (
          <NodeRow
            compact={compact}
            dense={dense}
            key={n.node_id}
            node={n}
            onSelect={onSelect}
            selected={n.node_id === selectedId}
            version={version}
          />
        ))}
      </div>
    </div>
  )
}
