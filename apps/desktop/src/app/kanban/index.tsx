import {
  DndContext,
  type DragEndEvent,
  DragOverlay,
  type DragStartEvent,
  PointerSensor,
  pointerWithin,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { type Translations, useI18n } from '@/i18n'
import { kanbanDispatchNudge } from '@/kanban-api'
import { cn } from '@/lib/utils'
import {
  $kanbanActiveBoard,
  $kanbanBoard,
  $kanbanBoardError,
  $kanbanBoards,
  $kanbanSelectedTaskId,
  isAllowedDrop,
  KANBAN_COLUMNS,
  moveKanbanTask,
  refreshKanbanBoard,
  refreshKanbanBoards,
  setKanbanActiveBoard,
  setKanbanSelectedTaskId
} from '@/store/kanban'
import { notify, notifyError } from '@/store/notifications'
import type { KanbanCard as KanbanCardData, KanbanColumn as KanbanColumnData } from '@/types/kanban'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

import { KanbanCardDrawer } from './card-drawer'
import { NewCardDialog } from './new-card-dialog'

// Live updates via polling (WS /events plumbing is out of scope for v1): the
// board payload's latest_event_id is the change cursor, so an unchanged poll
// keeps the previous atom identity and re-renders nothing. Paused while the
// window is hidden, same as the cron/messaging pollers.
const BOARD_POLL_INTERVAL_MS = 4_000

const PRIORITY_TONE: Record<string, string> = {
  high: 'bg-destructive/10 text-destructive',
  low: 'bg-foreground/10 text-muted-foreground'
}

const SEVERITY_TONE: Record<string, string> = {
  critical: 'bg-destructive/15 text-destructive',
  error: 'bg-destructive/10 text-destructive',
  warning: 'bg-amber-500/10 text-amber-600 dark:text-amber-300'
}

interface KanbanViewProps {
  onClose: () => void
}

export function KanbanView({ onClose }: KanbanViewProps) {
  const { t } = useI18n()

  return (
    <Panel closeLabel={t.kanban.close} onClose={onClose}>
      {/* The board must never crash the shell — everything inside the panel
          chrome mounts behind its own boundary so the close button survives. */}
      <ErrorBoundary label="kanban">
        <KanbanBoardContent />
      </ErrorBoundary>
    </Panel>
  )
}

function KanbanBoardContent() {
  const { t } = useI18n()
  const k = t.kanban
  const boards = useStore($kanbanBoards)
  const activeBoard = useStore($kanbanActiveBoard)
  const board = useStore($kanbanBoard)
  const boardError = useStore($kanbanBoardError)
  const selectedTaskId = useStore($kanbanSelectedTaskId)

  const [nudging, setNudging] = useState(false)
  const [newCardOpen, setNewCardOpen] = useState(false)
  const [dragCard, setDragCard] = useState<KanbanCardData | null>(null)

  const refresh = useCallback(() => {
    void refreshKanbanBoards()
    void refreshKanbanBoard()
  }, [])

  useRefreshHotkey(refresh)

  useEffect(() => {
    refresh()
  }, [refresh, activeBoard])

  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === 'visible') {
        void refreshKanbanBoard()
      }
    }

    const intervalId = window.setInterval(tick, BOARD_POLL_INTERVAL_MS)

    document.addEventListener('visibilitychange', tick)

    return () => {
      window.clearInterval(intervalId)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [])

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }))

  const columns = useMemo<KanbanColumnData[]>(() => {
    const byName = new Map((board?.columns ?? []).map(column => [column.name, column]))

    return KANBAN_COLUMNS.map(name => byName.get(name) ?? { name, tasks: [] })
  }, [board])

  const totalCount = useMemo(() => columns.reduce((sum, column) => sum + column.tasks.length, 0), [columns])

  const boardValue = activeBoard ?? boards.find(entry => entry.is_current)?.slug ?? 'default'

  const handleDragStart = useCallback(({ active }: DragStartEvent) => {
    setDragCard((active.data.current?.card as KanbanCardData | undefined) ?? null)
  }, [])

  const handleDragEnd = useCallback(
    ({ active, over }: DragEndEvent) => {
      setDragCard(null)

      const card = active.data.current?.card as KanbanCardData | undefined
      const target = over ? String(over.id) : null

      if (!card || !target || target === card.status || !isAllowedDrop(card.status, target)) {
        return
      }

      void moveKanbanTask(card.id, target).then(result => {
        if (!result.ok) {
          notify({ kind: 'error', title: k.moveFailed, message: result.error ?? '' })
        } else if (result.ruleMoved && result.status) {
          notify({ kind: 'info', message: k.movedByRules(k.columnLabels[result.status] ?? result.status) })
        }
      })
    },
    [k]
  )

  async function handleNudge() {
    setNudging(true)

    try {
      const result = await kanbanDispatchNudge(activeBoard)

      notify({
        kind: 'success',
        message: k.nudgeSent(result.spawned?.length ?? 0, result.promoted ?? 0)
      })
      void refreshKanbanBoard()
    } catch (error) {
      notifyError(error, k.nudgeFailed)
    } finally {
      setNudging(false)
    }
  }

  if (!board && !boardError) {
    return <PageLoader label={k.loading} />
  }

  return (
    <>
      <PanelHeader
        actions={
          <>
            <Select onValueChange={value => setKanbanActiveBoard(value)} value={boardValue}>
              <SelectTrigger aria-label={k.board} className="h-7 w-40 rounded-md text-xs">
                <SelectValue placeholder={k.board} />
              </SelectTrigger>
              <SelectContent>
                {(boards.length > 0 ? boards : [{ slug: boardValue, name: boardValue }]).map(entry => (
                  <SelectItem key={entry.slug} value={entry.slug}>
                    {entry.name || entry.slug}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              className="gap-1.5 text-muted-foreground hover:text-foreground"
              disabled={nudging}
              onClick={() => void handleNudge()}
              size="sm"
              variant="ghost"
            >
              <Codicon name={nudging ? 'loading' : 'zap'} size="0.875rem" spinning={nudging} />
              {k.nudgeDispatcher}
            </Button>
            <Button onClick={() => setNewCardOpen(true)} size="sm">
              {k.newCard}
            </Button>
          </>
        }
        subtitle={boardError ? boardError : k.count(totalCount)}
        title={k.title}
      />

      <div className="flex min-h-0 flex-1 gap-4 overflow-hidden">
        {boardError && !board ? (
          <PanelEmpty description={boardError} icon="warning" title={k.failedLoad} />
        ) : (
          <DndContext
            collisionDetection={pointerWithin}
            onDragCancel={() => setDragCard(null)}
            onDragEnd={handleDragEnd}
            onDragStart={handleDragStart}
            sensors={sensors}
          >
            <div className="flex min-h-0 flex-1 gap-2.5 overflow-x-auto overscroll-contain pb-2">
              {columns.map(column => (
                <KanbanColumn
                  column={column}
                  dragCard={dragCard}
                  k={k}
                  key={column.name}
                  onSelect={setKanbanSelectedTaskId}
                  selectedTaskId={selectedTaskId}
                />
              ))}
            </div>

            <DragOverlay dropAnimation={null}>
              {dragCard ? <KanbanCardTile card={dragCard} dragging k={k} /> : null}
            </DragOverlay>
          </DndContext>
        )}

        {selectedTaskId ? (
          <KanbanCardDrawer
            key={selectedTaskId}
            onClose={() => setKanbanSelectedTaskId(null)}
            taskId={selectedTaskId}
          />
        ) : null}
      </div>

      <NewCardDialog onClose={() => setNewCardOpen(false)} open={newCardOpen} />
    </>
  )
}

function KanbanColumn({
  column,
  dragCard,
  k,
  onSelect,
  selectedTaskId
}: {
  column: KanbanColumnData
  dragCard: KanbanCardData | null
  k: Translations['kanban']
  onSelect: (id: string) => void
  selectedTaskId: null | string
}) {
  const { isOver, setNodeRef } = useDroppable({ id: column.name })

  const dropAllowed = dragCard ? isAllowedDrop(dragCard.status, column.name) : false
  const dropBlocked = dragCard !== null && !dropAllowed && column.name !== dragCard.status

  return (
    <div
      className={cn(
        'flex w-52 shrink-0 flex-col rounded-lg bg-foreground/[0.03] transition-opacity',
        dropBlocked && 'opacity-40',
        isOver && dropAllowed && 'bg-primary/8 ring-1 ring-primary/30'
      )}
      ref={setNodeRef}
    >
      <div className="flex shrink-0 items-center justify-between px-2.5 pb-1 pt-2">
        <span className="text-[0.6rem] font-medium uppercase tracking-wider text-muted-foreground/60">
          {k.columnLabels[column.name] ?? column.name}
        </span>
        <span className="text-[0.62rem] tabular-nums text-muted-foreground/45">{column.tasks.length}</span>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto overscroll-contain p-1.5 pt-0.5">
        {column.tasks.length === 0 ? (
          <div className="py-3 text-center text-[0.65rem] text-muted-foreground/40">{k.noTasks}</div>
        ) : (
          column.tasks.map(card => (
            <DraggableCard
              active={card.id === selectedTaskId}
              card={card}
              k={k}
              key={card.id}
              onSelect={() => onSelect(card.id)}
            />
          ))
        )}
      </div>
    </div>
  )
}

function DraggableCard({
  active,
  card,
  k,
  onSelect
}: {
  active: boolean
  card: KanbanCardData
  k: Translations['kanban']
  onSelect: () => void
}) {
  const { attributes, isDragging, listeners, setNodeRef } = useDraggable({
    id: card.id,
    data: { card }
  })

  return (
    <div
      className={cn('touch-none', isDragging && 'opacity-30')}
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      onClick={onSelect}
      onKeyDown={event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect()
        }
      }}
      role="button"
      tabIndex={0}
    >
      <KanbanCardTile active={active} card={card} k={k} />
    </div>
  )
}

function priorityTone(priority: number): string {
  if (priority > 0) {
    return PRIORITY_TONE.high
  }

  return PRIORITY_TONE.low
}

export function KanbanCardTile({
  active,
  card,
  dragging,
  k
}: {
  active?: boolean
  card: KanbanCardData
  dragging?: boolean
  k: Translations['kanban']
}) {
  const warnings = card.warnings
  const commentCount = card.comment_count ?? 0
  const progress = card.progress

  return (
    <div
      className={cn(
        'cursor-pointer select-none rounded-md border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) p-2 text-left shadow-xs transition-colors hover:border-(--ui-stroke-tertiary)',
        active && 'border-primary/50 ring-1 ring-primary/30',
        dragging && 'shadow-md'
      )}
      data-kanban-card={card.id}
    >
      <div className="mb-1 line-clamp-3 text-[0.74rem] font-medium leading-snug text-foreground/90">
        {card.title || card.id}
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span
          className={cn(
            'inline-flex max-w-24 items-center gap-0.5 truncate rounded-full bg-foreground/8 px-1.5 py-px text-[0.6rem] text-muted-foreground'
          )}
          title={card.assignee ?? k.unassigned}
        >
          <Codicon className="shrink-0" name="account" size="0.6rem" />
          <span className="truncate">{card.assignee ?? k.unassigned}</span>
        </span>
        {card.priority !== 0 && (
          <span
            className={cn(
              'inline-flex items-center rounded-full px-1.5 py-px text-[0.6rem] tabular-nums',
              priorityTone(card.priority)
            )}
            title={k.priority}
          >
            P{card.priority}
          </span>
        )}
        {progress && progress.total > 0 && (
          <span
            className="inline-flex items-center gap-0.5 rounded-full bg-foreground/8 px-1.5 py-px text-[0.6rem] tabular-nums text-muted-foreground"
            title={k.progressTitle(progress.done, progress.total)}
          >
            <Codicon name="type-hierarchy-sub" size="0.6rem" />
            {progress.done}/{progress.total}
          </span>
        )}
        {warnings && warnings.count > 0 && (
          <span
            className={cn(
              'inline-flex items-center gap-0.5 rounded-full px-1.5 py-px text-[0.6rem] tabular-nums',
              SEVERITY_TONE[warnings.highest_severity ?? 'warning'] ?? SEVERITY_TONE.warning
            )}
            title={k.diagnostics}
          >
            <Codicon name="warning" size="0.6rem" />
            {warnings.count}
          </span>
        )}
        {commentCount > 0 && (
          <span
            className="inline-flex items-center gap-0.5 rounded-full bg-foreground/8 px-1.5 py-px text-[0.6rem] tabular-nums text-muted-foreground"
            title={k.comments}
          >
            <Codicon name="comment" size="0.6rem" />
            {commentCount}
          </span>
        )}
      </div>
    </div>
  )
}
