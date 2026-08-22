/**
 * Roadmaps plugin — Milestones view.
 *
 * Milestone / objective nodes of the active version, grouped by their parent
 * node when one exists (flat otherwise), with progress and owner.
 */

import { cn, Codicon, EmptyState, StatusDot } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { NODE_TONE } from './config'
import { groupMilestones, milestoneNodes, nodeLabel, plural } from './data'
import type { RoadmapNode, RoadmapVersion } from './types'
import { NodeStateTag, ProgressBar, SectionTitle } from './ui'

function MilestoneRow({
  node,
  selected,
  onSelect,
  compact
}: {
  node: RoadmapNode
  selected: boolean
  onSelect: (id: string) => void
  compact: boolean
}) {
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
        <span className="min-w-0 flex-1 truncate text-xs font-medium">{nodeLabel(node)}</span>
        {!compact ? <span className="font-mono text-[0.6rem] uppercase text-(--ui-text-quaternary)">{node.kind}</span> : null}
        {!compact ? <NodeStateTag state={node.state} /> : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-3.5">
        <ProgressBar value={node.progress} />
        {node.owner_agent ? (
          <span className="inline-flex min-w-0 items-center gap-1 truncate text-[0.625rem] text-(--ui-text-tertiary)">
            <Codicon name="person" size="0.65rem" />
            <span className="truncate">{node.owner_agent}</span>
          </span>
        ) : null}
      </div>
    </button>
  )
}

export function MilestonesView({
  version,
  selectedId,
  onSelect,
  compact
}: {
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
  compact: boolean
}) {
  const nodes = useMemo(() => milestoneNodes(version), [version])
  const groups = useMemo(() => groupMilestones(version), [version])

  if (nodes.length === 0) {
    return <EmptyState description="The active version contains no milestones or objectives." title="No milestones" />
  }

  return (
    <div className="flex flex-col gap-2">
      <SectionTitle right={<span className="tabular-nums text-(--ui-text-quaternary)">{plural(nodes.length, 'item')}</span>}>
        Milestones &amp; objectives
      </SectionTitle>
      {groups.map((g, gi) => (
        <div className="flex flex-col" key={`group-${gi}`}>
          {g.label ? (
            <div className="flex items-center gap-1 px-1 py-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
              <Codicon name="milestone" size="0.65rem" />
              <span className="truncate">{g.label}</span>
              <span className="tabular-nums text-(--ui-text-quaternary)">{g.nodes.length}</span>
            </div>
          ) : null}
          <div className="flex flex-col divide-y divide-(--ui-stroke-tertiary)">
            {g.nodes.map((n) => (
              <MilestoneRow compact={compact} key={n.node_id} node={n} onSelect={onSelect} selected={n.node_id === selectedId} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
