import {
  DndContext,
  type DragEndEvent,
  DragOverlay,
  type DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'
import { createKanbanTask, getKanbanBoard, updateKanbanTask } from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { KanbanBoardResponse, KanbanStatus, KanbanTask } from '@/types/hermes'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { PAGE_INSET_X } from '../layout-constants'
import { PageSearchShell } from '../page-search-shell'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { KanbanCardContent } from './card'
import { KanbanColumn } from './column'
import { isKanbanColumn, KANBAN_COLUMNS } from './constants'
import { KanbanDetail } from './detail'
import { filterTasksByStatus, moveTaskStatus } from './helpers'

const KANBAN_POLL_INTERVAL_MS = 3000

interface KanbanViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function KanbanView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: KanbanViewProps) {
  const [board, setBoard] = useState<KanbanBoardResponse | null>(null)
  const [query, setQuery] = useState('')
  const [assignee, setAssignee] = useState<string | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [activeTask, setActiveTask] = useState<KanbanTask | null>(null)
  const [newTitle, setNewTitle] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  const lastEventId = useRef(-1)
  const busy = useRef(false)

  const load = useCallback(async (force: boolean) => {
    try {
      const next = await getKanbanBoard()

      if (force || next.latest_event_id !== lastEventId.current) {
        lastEventId.current = next.latest_event_id
        setBoard(next)
      }
    } catch (err) {
      notifyError(err, 'Failed to load the Kanban board')
    }
  }, [])

  const refresh = useCallback(async () => {
    setRefreshing(true)

    try {
      await load(true)
    } finally {
      setRefreshing(false)
    }
  }, [load])

  useRefreshHotkey(() => void refresh())

  useEffect(() => {
    void load(true)
  }, [load])

  useEffect(() => {
    const timer = setInterval(() => {
      if (busy.current || activeTask) {
        return
      }

      void load(false)
    }, KANBAN_POLL_INTERVAL_MS)

    return () => clearInterval(timer)
  }, [load, activeTask])

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }))

  const grouped = useMemo(() => {
    if (!board) {
      return null
    }

    return filterTasksByStatus({ board, query, assignee })
  }, [board, query, assignee])

  const selectedTask = useMemo(() => {
    if (!board || !selectedTaskId) {
      return null
    }

    return board.columns.flatMap(column => column.tasks).find(task => task.id === selectedTaskId) ?? null
  }, [board, selectedTaskId])

  function handleDragStart(event: DragStartEvent) {
    const task = board?.columns.flatMap(column => column.tasks).find(item => item.id === event.active.id)

    setActiveTask(task ?? null)
  }

  async function handleDragEnd(event: DragEndEvent) {
    setActiveTask(null)

    const overId = event.over?.id

    if (!board || typeof overId !== 'string' || !isKanbanColumn(overId)) {
      return
    }

    const taskId = String(event.active.id)
    const toStatus: KanbanStatus = overId
    const current = board.columns.flatMap(column => column.tasks).find(task => task.id === taskId)

    if (!current || current.status === toStatus) {
      return
    }

    busy.current = true
    setBoard(moveTaskStatus({ board, taskId, toStatus }))

    try {
      await updateKanbanTask(taskId, { status: toStatus })
      await load(true)
    } catch (err) {
      notifyError(err, 'Failed to move task')
      await load(true)
    } finally {
      busy.current = false
    }
  }

  async function handleCreate() {
    const title = newTitle.trim()

    if (!title) {
      return
    }

    busy.current = true

    try {
      await createKanbanTask({ title, triage: true })
      setNewTitle('')
      notify({ kind: 'success', title: 'Task created', message: 'Added to triage' })
      await load(true)
    } catch (err) {
      notifyError(err, 'Failed to create task')
    } finally {
      busy.current = false
    }
  }

  if (!board || !grouped) {
    return (
      <PageSearchShell {...props} onSearchChange={setQuery} searchHidden searchPlaceholder="" searchValue="">
        <PageLoader />
      </PageSearchShell>
    )
  }

  const now = board.now

  return (
    <PageSearchShell
      {...props}
      filters={
        board.assignees.length > 0 ? (
          <>
            <TextTab active={assignee === null} onClick={() => setAssignee(null)}>
              Everyone
            </TextTab>
            {board.assignees.map(name => {
              const count = board.columns.reduce(
                (total, column) => total + column.tasks.filter(task => task.assignee === name).length,
                0
              )

              return (
                <TextTab
                  active={assignee === name}
                  key={name}
                  onClick={() => setAssignee(assignee === name ? null : name)}
                >
                  {name} <TextTabMeta>{count}</TextTabMeta>
                </TextTab>
              )
            })}
          </>
        ) : undefined
      }
      onSearchChange={setQuery}
      searchPlaceholder="Search tasks"
      searchTrailingAction={
        <Button
          aria-label={refreshing ? 'Refreshing' : 'Refresh'}
          className="text-(--ui-text-tertiary) hover:bg-transparent hover:text-foreground"
          disabled={refreshing}
          onClick={() => void refresh()}
          size="icon-xs"
          title={refreshing ? 'Refreshing' : 'Refresh'}
          type="button"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.875rem" spinning={refreshing} />
        </Button>
      }
      searchValue={query}
    >
      <div className="flex h-full min-h-0">
        <div className={cn('flex min-w-0 flex-1 flex-col gap-2 pb-3', PAGE_INSET_X)}>
          <form
            className="flex shrink-0 items-center gap-2 pt-1"
            onSubmit={event => {
              event.preventDefault()
              void handleCreate()
            }}
          >
            <input
              className="min-w-0 flex-1 rounded-[6px] border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) px-2.5 py-1.5 text-xs text-(--ui-text-primary) placeholder:text-(--ui-text-tertiary) focus-visible:border-(--ui-accent) focus-visible:outline-none"
              onChange={event => setNewTitle(event.target.value)}
              placeholder="Add a task to triage…"
              value={newTitle}
            />
            <Button disabled={!newTitle.trim()} size="sm" type="submit" variant="secondary">
              <Codicon name="add" size="0.875rem" /> Add
            </Button>
          </form>

          <DndContext
            onDragEnd={event => void handleDragEnd(event)}
            onDragStart={handleDragStart}
            sensors={sensors}
          >
            <div className="flex min-h-0 flex-1 gap-3 overflow-x-auto pb-1">
              {KANBAN_COLUMNS.map(status => (
                <KanbanColumn
                  key={status}
                  now={now}
                  onOpenTask={setSelectedTaskId}
                  status={status}
                  tasks={grouped[status] ?? []}
                />
              ))}
            </div>

            <DragOverlay>
              {activeTask ? (
                <div className="w-64">
                  <KanbanCardContent dragging now={now} task={activeTask} />
                </div>
              ) : null}
            </DragOverlay>
          </DndContext>
        </div>

        {selectedTask ? (
          <KanbanDetail now={now} onClose={() => setSelectedTaskId(null)} task={selectedTask} />
        ) : null}
      </div>
    </PageSearchShell>
  )
}
