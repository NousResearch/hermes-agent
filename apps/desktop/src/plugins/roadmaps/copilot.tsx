/**
 * Roadmaps plugin — orchestration copilot.
 *
 * A data-driven strip computed ONLY from the active version's real nodes and
 * relations (no LLM, no static blocks, no fixtures): Next action plus
 * Now / In flight / Waiting / Blocked chip groups.
 */

import { cn, Codicon, StatusDot } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { NEXT_ACTION_LABEL, NODE_TONE } from './config'
import { copilotSections, nextAction, nodeLabel, plural } from './data'
import type { RoadmapNode, RoadmapVersion } from './types'

function CopilotChip({
  node,
  selected,
  onSelect
}: {
  node: RoadmapNode
  selected: boolean
  onSelect: (id: string) => void
}) {
  return (
    <button
      className={cn(
        'inline-flex min-w-0 max-w-full items-center gap-1 truncate rounded-[3px] px-1.5 py-0.5 text-[0.6875rem] transition-colors',
        selected ? 'bg-primary/10 text-primary' : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )}
      onClick={() => onSelect(node.node_id)}
      title={`${node.kind} · ${node.state}`}
      type="button"
    >
      <StatusDot tone={NODE_TONE[node.state] ?? 'muted'} />
      <span className="truncate">{nodeLabel(node)}</span>
    </button>
  )
}

/** Dense chips: at most 3 per group (2 when dense), then a quiet "+N" marker. */
function CopilotChips({
  nodes,
  selectedId,
  onSelect,
  dense
}: {
  nodes: RoadmapNode[]
  selectedId: string
  onSelect: (id: string) => void
  dense: boolean
}) {
  const shown = nodes.slice(0, dense ? 2 : 3)
  const extra = nodes.length - shown.length

  return (
    <span className="flex min-w-0 flex-wrap items-center gap-1">
      {shown.map((n) => (
        <CopilotChip key={n.node_id} node={n} onSelect={onSelect} selected={n.node_id === selectedId} />
      ))}
      {extra > 0 ? <span className="text-[0.625rem] text-(--ui-text-quaternary)">{`+${extra}`}</span> : null}
    </span>
  )
}

function NextActionRow({
  action,
  selected,
  onSelect
}: {
  action: ReturnType<typeof nextAction>
  selected: boolean
  onSelect: (id: string) => void
}) {
  if (!action) {return null}
  const { node, kind, pending } = action
  let detail: string

  if (kind === 'unblock') {detail = 'Blocked — dependencies satisfied'}
  else if (kind === 'claim') {detail = 'Ready — dependencies satisfied'}
  else if (kind === 'advance')
    {detail = node.state === 'in_progress' ? `In flight${node.owner_agent ? ` · ${node.owner_agent}` : ''}` : 'Ready — dependencies satisfied'}
  else if (kind === 'assign') {detail = 'In progress without an owner'}
  else {detail = `Waiting on ${plural(pending, 'pending dependency')}`}

  return (
    <div className="flex items-center gap-1.5 text-[0.6875rem]">
      <span className="inline-flex shrink-0 items-center gap-1 font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
        <Codicon name="target" size="0.7rem" />
        Next action
      </span>
      <button
        className={cn(
          'inline-flex min-w-0 max-w-full items-center gap-1 rounded-[3px] px-1.5 py-0.5 transition-colors',
          selected ? 'bg-primary/10 text-primary' : 'text-foreground hover:bg-(--chrome-action-hover)'
        )}
        onClick={() => onSelect(node.node_id)}
        type="button"
      >
        <StatusDot tone={NODE_TONE[node.state] ?? 'muted'} />
        <span className="font-medium">{NEXT_ACTION_LABEL[kind] ?? kind}</span>
        <span className="truncate">{nodeLabel(node)}</span>
        <span className="text-(--ui-text-tertiary)">{`· ${detail}`}</span>
      </button>
    </div>
  )
}

export function CopilotBar({
  version,
  selectedId,
  onSelect,
  dense
}: {
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
  dense: boolean
}) {
  const sections = useMemo(() => copilotSections(version), [version])
  const action = useMemo(() => nextAction(version), [version])

  const groups = (sections
    ? [
        { key: 'now', label: 'Now', codicon: 'play', nodes: sections.now },
        { key: 'inflight', label: 'In flight', codicon: 'list-ordered', nodes: sections.inflight },
        { key: 'waiting', label: 'Waiting', codicon: 'hourglass', nodes: sections.waiting },
        { key: 'blocked', label: 'Blocked', codicon: 'debug-disconnect', nodes: sections.blocked }
      ]
    : []
  ).filter((g) => g.nodes.length > 0)

  if (!action && groups.length === 0) {
    return (
      <div className="px-0.5 text-xs text-(--ui-text-tertiary)">
        Nothing actionable — every node is resolved or not ready yet.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <NextActionRow action={action} onSelect={onSelect} selected={action ? action.node.node_id === selectedId : false} />
      {groups.length > 0 ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {groups.map((g, i) => (
            <div className={cn('flex min-w-0 items-center gap-1.5', i > 0 && 'border-l border-(--ui-stroke-tertiary) pl-4')} key={g.key}>
              <span className="flex shrink-0 items-center gap-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
                <Codicon name={g.codicon} size="0.7rem" />
                {g.label}
              </span>
              <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{g.nodes.length}</span>
              <CopilotChips dense={dense} nodes={g.nodes} onSelect={onSelect} selectedId={selectedId} />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
