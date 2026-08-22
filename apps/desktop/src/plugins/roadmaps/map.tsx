/**
 * Roadmaps plugin — Map view.
 *
 * Canonical relations (depends_on, blocks) of the active version — active by
 * default, with a toggle to include inactive ones. Each row is a real
 * relation from the snapshot; nothing decorative.
 */

import { cn, Codicon, EmptyState } from '@hermes/plugin-sdk'
import { useMemo, useState } from 'react'

import { RELATION_ICON, RELATION_LABEL } from './config'
import { type MappedRelation, mapRelations, nodeLabel } from './data'
import type { RoadmapVersion } from './types'
import { SectionTitle } from './ui'

export function RelationRow({
  rel,
  selectedNodeId,
  onSelect
}: {
  rel: MappedRelation
  selectedNodeId: string
  onSelect: (id: string) => void
}) {
  return (
    <div
      className={cn(
        'grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2 px-2 py-1.5 text-xs transition-colors',
        (rel.from_node_id === selectedNodeId || rel.to_node_id === selectedNodeId) && 'bg-primary/[0.04]'
      )}
    >
      <button
        className={cn('min-w-0 truncate text-left hover:underline', rel.from_node_id === selectedNodeId ? 'text-primary' : 'text-foreground')}
        onClick={() => onSelect(rel.from_node_id)}
        type="button"
      >
        {nodeLabel(rel.from)}
      </button>
      <span className="flex shrink-0 items-center gap-1 text-[0.625rem] uppercase tracking-wide text-(--ui-text-tertiary)">
        <Codicon name={RELATION_ICON[rel.kind] ?? 'arrow-right'} size="0.65rem" />
        {RELATION_LABEL[rel.kind] ?? rel.kind}
      </span>
      <button
        className={cn('min-w-0 truncate text-right hover:underline', rel.to_node_id === selectedNodeId ? 'text-primary' : 'text-foreground')}
        onClick={() => onSelect(rel.to_node_id)}
        type="button"
      >
        {nodeLabel(rel.to)}
      </button>
    </div>
  )
}

export function MapView({
  version,
  selectedId,
  onSelect
}: {
  version: RoadmapVersion | null
  selectedId: string
  onSelect: (id: string) => void
}) {
  const [showInactive, setShowInactive] = useState(false)
  const rels = useMemo(() => mapRelations(version, { includeInactive: showInactive }), [version, showInactive])

  return (
    <div className="flex flex-col gap-1.5">
      <SectionTitle
        right={
          <button
            className="rounded-[3px] px-1 text-[0.625rem] normal-case tracking-normal text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={() => setShowInactive((v) => !v)}
            type="button"
          >
            {showInactive ? 'active only' : 'include inactive'}
          </button>
        }
      >
        {`Relations (${rels.length})`}
      </SectionTitle>
      {rels.length === 0 ? (
        <EmptyState
          description="Each row is a canonical relation (depends on, blocks) of the active version."
          title={showInactive ? 'No relations' : 'No active relations'}
        />
      ) : (
        <div className="flex flex-col divide-y divide-(--ui-stroke-tertiary)">
          {rels.map((r) => (
            <RelationRow key={r.relation_id} onSelect={onSelect} rel={r} selectedNodeId={selectedId} />
          ))}
        </div>
      )}
    </div>
  )
}
