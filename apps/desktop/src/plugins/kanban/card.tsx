/** One kanban card: title, summary, footer chrome, the machine-activity arc,
 *  drag source, and its right-click actions. Shared by the flat board and the
 *  swimlane grid. */

import {
  cn,
  Codicon,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
  formatModifierToken,
  Tip
} from '@hermes/plugin-sdk'
import { type CSSProperties, type ReactNode, useState } from 'react'

import { columnMeta, type KanbanTask } from './types'
import {
  ago,
  type ArcState,
  arcState,
  Avatar,
  columnLabel,
  isLockedTarget,
  PriorityGlyph,
  RunClock,
  shortId,
  useDefaultAssignee,
  useKanban,
  useOrchestration
} from './ui'

function Meta({ children, icon }: { children: ReactNode; icon: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <Codicon name={icon} size="0.7rem" />
      {children}
    </span>
  )
}

function CardFooter({ arc, task }: { arc: ArcState | null; task: KanbanTask }) {
  const k = useKanban()
  const created = ago(task.created_at)
  const links = task.link_counts ? task.link_counts.parents + task.link_counts.children : 0
  const fallback = useDefaultAssignee()
  const orchestrator = useOrchestration()?.resolved_orchestrator_profile ?? ''
  // Ready + no assignee: with a configured default assignee the dispatcher
  // auto-assigns on its next tick (#27145) — say THAT, not "won't run". Only
  // a board with no fallback has the genuine silent failure.
  const unassignedReady = task.status === 'ready' && !task.assignee

  // The agent on the hook for a queued card: the explicit assignee, else the
  // auto-default (ready), else the specifier that rewrites triage cards.
  const attached = task.assignee || (task.status === 'ready' ? fallback : task.status === 'triage' ? orchestrator : '')

  const meta = columnMeta(task.status)

  return (
    <div className="flex items-center gap-2 whitespace-nowrap text-[0.625rem] text-(--ui-text-tertiary)">
      {arc === 'queued' && attached ? (
        // WHO is coming for the card. The arc only animates once the agent is
        // actually working; while queued, the named chip carries "attached".
        <Tip
          label={
            task.status === 'review'
              ? k.reviewChecking
              : task.assignee
                ? k.attachedTip(attached)
                : task.status === 'triage'
                  ? k.orchestratorTip(attached)
                  : k.autoAssignTip(attached)
          }
        >
          <span className="inline-flex min-w-0 cursor-help items-center gap-1 font-medium" style={{ color: meta.tone }}>
            <Avatar name={attached} size="1.125rem" />
            <span className="truncate">
              {!task.assignee && '→ '}
              {attached}
            </span>
          </span>
        </Tip>
      ) : task.assignee ? (
        <Avatar name={task.assignee} size="1.125rem" />
      ) : null}
      {arc === 'running' && (
        <Tip label={k.arcRunning}>
          <span className="shrink-0 cursor-help">
            <RunClock task={task} />
          </span>
        </Tip>
      )}
      {arc === 'stale' && (
        <Tip label={k.arcStale}>
          <span className="shrink-0 cursor-help font-medium text-amber-500">{k.noHeartbeat}</span>
        </Tip>
      )}
      {task.status === 'blocked' && task.block_kind && (
        // #124391: say WHY the card is blocked — the kind arrives on every
        // card payload; without it every blocked card reads identically.
        <Tip label={k.blockKindTip(task.block_kind)}>
          <span className="inline-flex shrink-0 cursor-help items-center gap-1 text-destructive">
            <Codicon name="debug-breakpoint-data-unverified" size="0.7rem" />
            {task.block_kind}
          </span>
        </Tip>
      )}
      {unassignedReady && !fallback && (
        <Tip label={k.wontRunTip}>
          <span className="inline-flex shrink-0 cursor-help items-center gap-1 text-amber-500">
            <Codicon name="debug-disconnect" size="0.7rem" />
            {k.wontRun}
          </span>
        </Tip>
      )}
      <div className="ml-auto flex min-w-0 shrink items-center gap-2">
        {typeof task.priority === 'number' && task.priority > 0 && <PriorityGlyph priority={task.priority} />}
        {task.progress && task.progress.total > 0 && (
          <Meta icon="checklist">
            {task.progress.done}/{task.progress.total}
          </Meta>
        )}
        {Boolean(task.comment_count) && <Meta icon="comment">{task.comment_count}</Meta>}
        {links > 0 && <Meta icon="references">{links}</Meta>}
        {task.warnings && task.warnings.count > 0 && (
          <span className="inline-flex items-center gap-0.5 text-destructive">
            <Codicon name="warning" size="0.7rem" />
            {task.warnings.count}
          </span>
        )}
        {created && !task.assignee && !unassignedReady ? (
          <span className="text-(--ui-text-quaternary)">{created}</span>
        ) : null}
        <span className="min-w-0 truncate font-mono text-(--ui-text-quaternary)">{shortId(task.id)}</span>
      </div>
    </div>
  )
}

export function Card({
  columns,
  onDelete,
  onMove,
  onOpen,
  onToggleSelect,
  selected,
  task
}: {
  columns: string[]
  onDelete: (id: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggleSelect: (id: string) => void
  selected: boolean
  task: KanbanTask
}) {
  const k = useKanban()
  const [dragging, setDragging] = useState(false)
  const meta = columnMeta(task.status)
  const summary = task.latest_summary || task.body
  const fallback = useDefaultAssignee()
  const arc = arcState(task, fallback)

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div
          className={cn(
            'group relative flex cursor-grab flex-col gap-2 rounded-md border border-(--ui-stroke-tertiary) border-l-2 bg-(--ui-bg-elevated) p-2.5',
            // Hover matches the provider-picker rows: a quiet primary fill;
            // selected = the theme's focus color (same as a focused input).
            'transition-colors hover:bg-primary/[0.06] active:cursor-grabbing',
            selected && 'border-(--dt-composer-ring) bg-[color-mix(in_srgb,var(--dt-composer-ring)_7%,transparent)]',
            dragging && 'opacity-40'
          )}
          draggable
          onClick={event => (event.metaKey || event.ctrlKey ? onToggleSelect(task.id) : onOpen(task.id))}
          onDragEnd={() => setDragging(false)}
          onDragStart={event => {
            event.dataTransfer.setData('text/plain', task.id)
            event.dataTransfer.effectAllowed = 'move'
            // Snapshot the drag image before dimming the source, so the ghost
            // stays a solid card (dimming first would bake 40% into it).
            event.dataTransfer.setDragImage(event.currentTarget, event.nativeEvent.offsetX, event.nativeEvent.offsetY)
            setDragging(true)
          }}
          style={{ '--kanban-tone': meta.tone, borderLeftColor: meta.tone } as CSSProperties}
        >
          {/* Machine-activity arc: animates ONLY while an agent is actually on
              the card (claimed + working; amber when the heartbeat is gone).
              Queued attachment is the footer's named-agent chip — a moving
              border on an idle card would lie. Hidden during drag/selection
              so those states stay legible. */}
          {(arc === 'running' || arc === 'stale') && !dragging && !selected && (
            <span aria-hidden className={cn('kanban-arc', arc === 'stale' && 'kanban-arc--stale')} />
          )}
          <span className="line-clamp-2 text-[0.8125rem] font-medium leading-snug text-foreground">
            {task.title || task.id}
          </span>
          {summary && (
            <span className="line-clamp-2 text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{summary}</span>
          )}
          <CardFooter arc={arc} task={task} />
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => onOpen(task.id)}>
          <Codicon name="link-external" size="0.85rem" />
          {k.open}
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => onToggleSelect(task.id)}>
          <Codicon name={selected ? 'close' : 'check-all'} size="0.85rem" />
          {selected ? k.deselect : k.select(formatModifierToken('mod'))}
        </ContextMenuItem>
        <ContextMenuSeparator />
        {columns
          .filter(name => name !== task.status && !isLockedTarget(name))
          .map(name => (
            <ContextMenuItem key={name} onSelect={() => onMove(task.id, name)}>
              <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
              {k.moveTo(columnLabel(k, name))}
            </ContextMenuItem>
          ))}
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => onDelete(task.id)} variant="destructive">
          <Codicon name="trash" size="0.85rem" />
          {k.delete}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
