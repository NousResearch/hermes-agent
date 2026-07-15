import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { useI18n } from '@/i18n'
import { useNavigate } from 'react-router-dom'
import { sessionRoute } from '@/app/routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface KanbanBoard {
  id: string
  title: string
  description: string
  createdAt: number
}

interface KanbanTask {
  id: string
  boardId: string
  title: string
  description: string
  status: KanbanStatus
  priority: KanbanPriority
  assignee: string
  createdBy: string
  createdAt: number
  updatedAt: number
  archived: boolean
  order: number
  labels?: string[]
  sessionId?: string
  source?: KanbanTaskSource
  profileId?: string
  profileLabel?: string
  messageId?: string
  assigneeType: KanbanAssigneeType
  assigneeId?: string
  assigneeLabel?: string
  agentId?: string
  syncMode?: string
}

interface KanbanComment {
  id: string
  taskId: string
  author: string
  body: string
  createdAt: number
}

type KanbanStatus = 'todo' | 'ready' | 'running' | 'review' | 'done' | 'blocked'
type KanbanPriority = 'low' | 'medium' | 'high'
type KanbanAssigneeType = 'user' | 'agent' | 'unassigned'
type KanbanTaskSource = 'manual' | 'chat' | 'agent' | 'cron'

interface KanbanStatusColumn {
  id: KanbanStatus
  label: string
  color: string
  icon: string
}

const STATUS_COLUMNS: KanbanStatusColumn[] = [
  { id: 'todo', label: 'Todo', color: 'bg-sky-500', icon: 'circle-outline' },
  { id: 'ready', label: 'Ready', color: 'bg-blue-500', icon: 'circle-filled' },
  { id: 'running', label: 'Running', color: 'bg-amber-500', icon: 'debug-start' },
  { id: 'review', label: 'Review', color: 'bg-purple-500', icon: 'eye' },
  { id: 'done', label: 'Done', color: 'bg-emerald-500', icon: 'check' },
  { id: 'blocked', label: 'Blocked', color: 'bg-red-500', icon: 'error' }
]

const PRIORITY_CONFIG = {
  high: { label: 'High', color: 'text-red-500', icon: 'arrow-up' },
  medium: { label: 'Medium', color: 'text-amber-500', icon: 'dash' },
  low: { label: 'Low', color: 'text-sky-500', icon: 'arrow-down' }
} as const

// ---------------------------------------------------------------------------
// i18n helpers
// ---------------------------------------------------------------------------

function columnLabel(t: ReturnType<typeof useI18n>['t'], id: KanbanStatus): string {
  switch (id) {
    case 'todo': return t.desktop.kanban.columnTodo
    case 'ready': return t.desktop.kanban.columnReady
    case 'running': return t.desktop.kanban.columnRunning
    case 'review': return t.desktop.kanban.columnReview
    case 'done': return t.desktop.kanban.columnDone
    case 'blocked': return t.desktop.kanban.columnBlocked
  }
}

function priorityLabel(t: ReturnType<typeof useI18n>['t'], p: string): string {
  switch (p) {
    case 'high': return t.desktop.kanban.priorityHigh
    case 'medium': return t.desktop.kanban.priorityMedium
    case 'low': return t.desktop.kanban.priorityLow
    default: return p
  }
}

// ---------------------------------------------------------------------------
// Sortable task card
// ---------------------------------------------------------------------------

function SortableTaskCard({
  task,
  onEdit,
  onDelete,
  onSelect
}: {
  task: KanbanTask
  onEdit: (task: KanbanTask) => void
  onDelete: (task: KanbanTask) => void
  onSelect: (task: KanbanTask) => void
}) {
  const { t } = useI18n()
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: task.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition
  }

  const priority = PRIORITY_CONFIG[task.priority as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.medium

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        'group cursor-pointer rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3 text-xs shadow-sm transition-shadow duration-100 hover:border-(--ui-stroke-tertiary)',
        isDragging && 'z-50 shadow-lg opacity-90'
      )}
      {...attributes}
      {...listeners}
      onClick={() => onSelect(task)}
    >
      <div className="mb-1.5 flex items-start justify-between gap-2">
        <span className="flex-1 text-sm font-medium leading-snug text-(--ui-text-primary)">
          {task.title}
        </span>
        <div className="flex shrink-0 gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
          <button
            className="flex h-5 w-5 items-center justify-center rounded text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)"
            onClick={e => {
              e.stopPropagation()
              onEdit(task)
            }}
            title={t.desktop.kanban.editTaskTooltip}
          >
            <Codicon name="edit" size="0.75rem" />
          </button>
          <button
            className="flex h-5 w-5 items-center justify-center rounded text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-red-500"
            onClick={e => {
              e.stopPropagation()
              onDelete(task)
            }}
            title={t.desktop.kanban.deleteTaskTooltip}
          >
            <Codicon name="trash" size="0.75rem" />
          </button>
        </div>
      </div>
      {task.description && (
        <p className="mb-2 line-clamp-2 leading-relaxed text-(--ui-text-tertiary)">
          {task.description}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <Codicon name={priority.icon} size="0.75rem" className={priority.color} />
        <Badge variant="outline" className="text-[0.6rem] uppercase tracking-wider">
          {priorityLabel(t, task.priority)}
        </Badge>
        {task.assignee && (
          <span className="ml-auto flex items-center gap-1 text-[0.6rem] text-(--ui-text-tertiary)">
            <Codicon name="person" size="0.625rem" />
            {task.assignee}
          </span>
        )}
        {task.status === 'blocked' && (
          <Codicon name="error" size="0.75rem" className="text-red-500" />
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Kanban column
// ---------------------------------------------------------------------------

function KanbanColumn({
  column,
  tasks,
  onEditTask,
  onDeleteTask,
  onSelectTask
}: {
  column: KanbanStatusColumn
  tasks: KanbanTask[]
  onEditTask: (task: KanbanTask) => void
  onDeleteTask: (task: KanbanTask) => void
  onSelectTask: (task: KanbanTask) => void
}) {
  const { t } = useI18n()
  const taskIds = useMemo(() => tasks.map(t => t.id), [tasks])
  const { setNodeRef: columnDroppableRef, isOver } = useDroppable({
    id: `column:${column.id}`,
    data: { type: 'column', status: column.id }
  })

  return (
    <div className="flex h-full w-72 shrink-0 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary)/40">
      {/* Column header */}
      <div className="mb-3 flex shrink-0 items-center gap-2 px-3 pt-3">
        <span className={cn('h-2 w-2 shrink-0 rounded-full', column.color)} />
        <span className="text-xs font-semibold uppercase tracking-wider text-(--ui-text-secondary)">
          {columnLabel(t, column.id)}
        </span>
        <span className="ml-auto text-[0.65rem] text-(--ui-text-tertiary)">
          {tasks.length}
        </span>
      </div>

      {/* Task list */}
      <SortableContext items={taskIds} strategy={verticalListSortingStrategy}>
        <div
          ref={columnDroppableRef}
          className={cn(
            'flex flex-1 flex-col gap-2 overflow-y-auto px-2 pb-2 transition-colors',
            isOver && 'bg-(--ui-bg-tertiary)'
          )}
        >
          {tasks.map(task => (
            <SortableTaskCard key={task.id} task={task} onEdit={onEditTask} onDelete={onDeleteTask} onSelect={onSelectTask} />
          ))}
          {tasks.length === 0 && (
            <div className="flex flex-1 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary)">
              <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{t.desktop.kanban.dropTasksHere}</span>
            </div>
          )}
        </div>
      </SortableContext>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Create / Edit task dialog
// ---------------------------------------------------------------------------

function TaskDialog({
  open,
  onOpenChange,
  task,
  onSave
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  task: KanbanTask | null
  onSave: (data: Partial<KanbanTask> & { title: string }) => void
}) {
  const { t } = useI18n()
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState<KanbanPriority>('medium')
  const [assignee, setAssignee] = useState('')
  const [status, setStatus] = useState<KanbanStatus>('todo')

  useEffect(() => {
    if (open) {
      setTitle(task?.title ?? '')
      setDescription(task?.description ?? '')
      setPriority(task?.priority ?? 'medium')
      setAssignee(task?.assignee ?? '')
      setStatus(task?.status ?? 'todo')
    }
  }, [open, task])

  const handleSave = useCallback(() => {
    if (!title.trim()) return
    onSave({ title: title.trim(), description, priority, assignee, status })
    onOpenChange(false)
  }, [title, description, priority, assignee, status, onSave, onOpenChange])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{task ? t.desktop.kanban.editTask : t.desktop.kanban.newTask}</DialogTitle>
          <DialogDescription>
            {task ? 'Update the task details below.' : 'Create a new kanban task.'}
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-2">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-(--ui-text-secondary)">{t.desktop.kanban.title}</label>
            <Input value={title} onChange={e => setTitle(e.target.value)} placeholder={t.desktop.kanban.taskTitlePlaceholder} autoFocus />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-(--ui-text-secondary)">{t.desktop.kanban.description}</label>
            <Textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder={t.desktop.kanban.descriptionPlaceholder}
              rows={3}
            />
          </div>
          <div className="flex gap-3">
            <div className="flex flex-1 flex-col gap-1">
              <label className="text-xs font-medium text-(--ui-text-secondary)">{t.desktop.kanban.status}</label>
              <select
                className={cn(
                  'h-8 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) px-2 text-xs text-(--ui-text-primary) outline-none',
                  'focus:border-(--ui-stroke-tertiary) focus:ring-[0.125rem] focus:ring-ring/30'
                )}
                value={status}
                onChange={e => setStatus(e.target.value as KanbanStatus)}
              >
                {STATUS_COLUMNS.map(col => (
                  <option key={col.id} value={col.id}>
                    {columnLabel(t, col.id)}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-1 flex-col gap-1">
              <label className="text-xs font-medium text-(--ui-text-secondary)">{t.desktop.kanban.priority}</label>
              <select
                className={cn(
                  'h-8 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) px-2 text-xs text-(--ui-text-primary) outline-none',
                  'focus:border-(--ui-stroke-tertiary) focus:ring-[0.125rem] focus:ring-ring/30'
                )}
                value={priority}
                onChange={e => setPriority(e.target.value as KanbanPriority)}
              >
                <option value="high">{t.desktop.kanban.priorityHigh}</option>
                <option value="medium">{t.desktop.kanban.priorityMedium}</option>
                <option value="low">{t.desktop.kanban.priorityLow}</option>
              </select>
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-(--ui-text-secondary)">{t.desktop.kanban.assignee}</label>
            <Input
              value={assignee}
              onChange={e => setAssignee(e.target.value)}
              placeholder={t.desktop.kanban.unassigned}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t.desktop.kanban.cancel}
          </Button>
          <Button variant="default" onClick={handleSave} disabled={!title.trim()}>
            {task ? t.desktop.kanban.saveChanges : t.desktop.kanban.createTask}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Task detail / comments panel
// ---------------------------------------------------------------------------

function TaskDetailPanel({
  task,
  comments,
  onClose,
  onAddComment,
  onDeleteComment,
  onArchive,
  onStatusChange
}: {
  task: KanbanTask
  comments: KanbanComment[]
  onClose: () => void
  onAddComment: (body: string) => void
  onDeleteComment: (id: string) => void
  onArchive: () => void
  onStatusChange: (status: KanbanStatus) => void
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [commentText, setCommentText] = useState('')

  const priority = PRIORITY_CONFIG[task.priority as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.medium
  const statusCol = STATUS_COLUMNS.find(c => c.id === task.status) ?? STATUS_COLUMNS[0]

  const handleSubmitComment = useCallback(() => {
    if (!commentText.trim()) return
    onAddComment(commentText.trim())
    setCommentText('')
  }, [commentText, onAddComment])

  return (
    <div className="flex h-full flex-col border-l border-(--ui-stroke-secondary) bg-(--ui-bg-primary)">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-(--ui-stroke-secondary) px-4 py-3">
        <span className="text-xs font-semibold text-(--ui-text-secondary)">{t.desktop.kanban.taskDetails}</span>
        <button
          className="flex h-6 w-6 items-center justify-center rounded text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)"
          onClick={onClose}
        >
          <Codicon name="close" size="0.875rem" />
        </button>
      </div>

      {/* Task info */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <h3 className="mb-2 text-sm font-semibold leading-snug text-(--ui-text-primary)">
          {task.title}
        </h3>
        {task.description && (
          <p className="mb-4 whitespace-pre-wrap text-xs leading-relaxed text-(--ui-text-tertiary)">
            {task.description}
          </p>
        )}

        {/* Meta badges */}
        <div className="mb-4 flex flex-wrap gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <span className={cn('h-1.5 w-1.5 rounded-full', statusCol.color)} />
            {columnLabel(t, statusCol.id)}
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <Codicon name={priority.icon} size="0.75rem" className={priority.color} />
            {priorityLabel(t, task.priority)}
          </Badge>
          {task.assignee && (
            <Badge variant="outline" className="flex items-center gap-1">
              <Codicon name="person" size="0.75rem" />
              {task.assignee}
            </Badge>
          )}
        </div>

        {/* Source metadata */}
        {(task.source || task.profileId || task.sessionId) && (
          <div className="mb-4 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-2.5">
            <span className="mb-1.5 block text-[0.6rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
              Metadata
            </span>
            <div className="flex flex-col gap-1 text-[0.6875rem] text-(--ui-text-secondary)">
              {task.source && (
                <div className="flex items-center gap-1.5">
                  <Codicon name="repo" size="0.75rem" className="shrink-0 text-(--ui-text-tertiary)" />
                  <span>Source: {task.source.charAt(0).toUpperCase() + task.source.slice(1)}</span>
                </div>
              )}
              {task.assigneeLabel && (
                <div className="flex items-center gap-1.5">
                  <Codicon name="person" size="0.75rem" className="shrink-0 text-(--ui-text-tertiary)" />
                  <span>Assignee: {task.assigneeLabel}</span>
                </div>
              )}
              {task.sessionId && (
                <button
                  className="flex items-center gap-1.5 text-(--ui-text-secondary) hover:text-(--ui-text-primary)"
                  onClick={(e) => { e.stopPropagation(); navigate(sessionRoute(task.sessionId!)) }}
                >
                  <Codicon name="link-external" size="0.75rem" className="shrink-0 text-(--ui-text-tertiary)" />
                  Open conversation
                </button>
              )}
            </div>
          </div>
        )}

        {/* Status quick-change */}
        <div className="mb-4">
          <label className="mb-1 block text-[0.65rem] font-medium uppercase tracking-wider text-(--ui-text-tertiary)">
            {t.desktop.kanban.moveTo}
          </label>
          <div className="flex flex-wrap gap-1">
            {STATUS_COLUMNS.map(col => (
              <button
                key={col.id}
                className={cn(
                  'rounded px-2 py-1 text-[0.65rem] font-medium transition-colors',
                  task.status === col.id
                    ? 'bg-(--ui-control-active-background) text-(--ui-text-primary)'
                    : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)'
                )}
                onClick={() => onStatusChange(col.id)}
              >
                {columnLabel(t, col.id)}
              </button>
            ))}
          </div>
        </div>

        {/* Archive button */}
        <Button variant="outline" size="sm" className="mb-4 w-full" onClick={onArchive}>
          <Codicon name="archive" size="0.75rem" />
          {t.desktop.kanban.archiveTask}
        </Button>

        {/* Comments section */}
        <div className="border-t border-(--ui-stroke-secondary) pt-3">
          <h4 className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
            {t.desktop.kanban.comments(comments.length)}
          </h4>
          <div className="mb-3 flex flex-col gap-2">
            {comments.map(comment => (
              <div
                key={comment.id}
                className="group rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-2"
              >
                <div className="mb-0.5 flex items-center justify-between">
                  <span className="text-[0.6rem] font-medium text-(--ui-text-secondary)">
                    {comment.author || 'Anonymous'}
                  </span>
                  <button
                    className="flex h-4 w-4 items-center justify-center rounded text-(--ui-text-tertiary) opacity-0 transition-opacity hover:text-red-500 group-hover:opacity-100"
                    onClick={() => onDeleteComment(comment.id)}
                    title="Delete comment"
                  >
                    <Codicon name="close" size="0.625rem" />
                  </button>
                </div>
                <p className="text-xs text-(--ui-text-primary)">{comment.body}</p>
              </div>
            ))}
            {comments.length === 0 && (
              <p className="text-[0.6875rem] italic text-(--ui-text-tertiary/60)">{t.desktop.kanban.noComments}</p>
            )}
          </div>

          {/* Add comment */}
          <div className="flex gap-2">
            <Textarea
              value={commentText}
              onChange={e => setCommentText(e.target.value)}
              placeholder={t.desktop.kanban.addCommentPlaceholder}
              rows={2}
              className="min-h-0 flex-1 text-xs"
            />
            <Button
              variant="default"
              size="sm"
              className="self-end"
              onClick={handleSubmitComment}
              disabled={!commentText.trim()}
            >
              {t.desktop.kanban.send}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// KanbanView — main exported component
// ---------------------------------------------------------------------------

interface KanbanViewProps {
  setStatusbarItemGroup: SetStatusbarItemGroup
}

export function KanbanView({ setStatusbarItemGroup: _setStatusbarItemGroup }: KanbanViewProps) {
  const { t } = useI18n()
  const [boards, setBoards] = useState<KanbanBoard[]>([])
  const [activeBoardId, setActiveBoardId] = useState<string>('default')
  const [tasks, setTasks] = useState<KanbanTask[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedTask, setSelectedTask] = useState<KanbanTask | null>(null)
  const [taskComments, setTaskComments] = useState<KanbanComment[]>([])
  const [editingTask, setEditingTask] = useState<KanbanTask | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [deletingTask, setDeletingTask] = useState<KanbanTask | null>(null)
  const [showNewBoard, setShowNewBoard] = useState(false)
  const [newBoardTitle, setNewBoardTitle] = useState('')
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<KanbanTask | null>(null)

  const dndSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  // Load boards and tasks
  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [boardList, allTasks] = await Promise.all([
        window.hermesDesktop.kanban.boards(),
        window.hermesDesktop.kanban.allTasks()
      ])
      setBoards(boardList)
      setTasks(allTasks)
      if (boardList.length > 0 && !boardList.find(b => b.id === activeBoardId)) {
        setActiveBoardId(boardList[0].id)
      }
    } catch (err) {
      notifyError(new Error('Failed to load kanban data'), 'Failed to load kanban data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [activeBoardId])

  useEffect(() => {
    void loadData()
  }, [loadData])

  // Load comments for selected task
  useEffect(() => {
    if (!selectedTask) {
      setTaskComments([])
      return
    }
    let cancelled = false
    window.hermesDesktop.kanban.comments(selectedTask.id).then(comments => {
      if (!cancelled) setTaskComments(comments)
    }).catch(() => {
      if (!cancelled) setTaskComments([])
    })
    return () => { cancelled = true }
  }, [selectedTask])

  // Filter tasks by active board and group by status
  const boardTasks = useMemo(
    () => tasks.filter(t => t.boardId === activeBoardId),
    [tasks, activeBoardId]
  )

  const tasksByStatus = useMemo(() => {
    const map: Record<string, KanbanTask[]> = {}
    for (const col of STATUS_COLUMNS) {
      map[col.id] = [...boardTasks.filter(t => t.status === col.id)]
        .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    }
    return map
  }, [boardTasks])

  // Handle drag end — update task status and order with optimistic update
  const handleDragEnd = useCallback(
    async (event: DragEndEvent) => {
      const { active, over } = event
      if (!over || active.id === over.id) return

      const activeTask = tasks.find(t => t.id === active.id)
      if (!activeTask) return

      // Determine target status
      let targetStatus: KanbanStatus
      let overIndex: number

      if (typeof over.id === 'string' && over.id.startsWith('column:')) {
        // Dropped on an empty column
        targetStatus = over.id.replace('column:', '') as KanbanStatus
        overIndex = (tasksByStatus[targetStatus] ?? []).length
      } else {
        // Dropped on another task
        const overTask = tasks.find(t => t.id === over.id)
        if (!overTask) return
        targetStatus = overTask.status as KanbanStatus
        const columnTasks = tasksByStatus[targetStatus] ?? []
        overIndex = columnTasks.findIndex(t => t.id === over.id)
        if (overIndex < 0) return
      }

      const sourceStatus = activeTask.status as KanbanStatus
      const sourceColumn = [...(tasksByStatus[sourceStatus] ?? [])]
      const targetColumn = sourceStatus === targetStatus
        ? sourceColumn
        : [...(tasksByStatus[targetStatus] ?? [])]

      // Remove active from source
      const activeSourceIdx = sourceColumn.findIndex(t => t.id === active.id)
      if (activeSourceIdx >= 0) sourceColumn.splice(activeSourceIdx, 1)

      // Insert into target at position
      const insertAt = Math.min(overIndex, targetColumn.length)
      targetColumn.splice(insertAt, 0, activeTask)

      // Generate updates
      const updates: Array<{ id: string; status: KanbanStatus; order: number }> = []
      const collectUpdates = (col: KanbanTask[], status: KanbanStatus) => {
        col.forEach((t, idx) => {
          if (t.status !== status || t.order !== idx) {
            updates.push({ id: t.id, status, order: idx })
          }
        })
      }

      if (sourceStatus === targetStatus) {
        collectUpdates(sourceColumn, sourceStatus)
      } else {
        collectUpdates(sourceColumn, sourceStatus)
        collectUpdates(targetColumn, targetStatus)
      }

      if (updates.length === 0) return

      // Optimistic local update
      const prevTasks = tasks
      setTasks(prev => {
        const next = [...prev]
        for (const { id, status, order } of updates) {
          const idx = next.findIndex(t => t.id === id)
          if (idx >= 0) {
            next[idx] = { ...next[idx], status, order }
          }
        }
        return next
      })

      try {
        await window.hermesDesktop.kanban.reorderTasks(activeBoardId, updates)
      } catch {
        setTasks(prevTasks)
        notifyError(new Error('Failed to reorder tasks'), 'Failed to reorder tasks')
      }
    },
    [tasks, tasksByStatus, activeBoardId]
  )

  // Create task
  const handleCreateTask = useCallback(
    async (data: { title: string; description?: string; priority?: KanbanPriority; assignee?: string; status?: KanbanStatus }) => {
      try {
        const task = await window.hermesDesktop.kanban.createTask({
          boardId: activeBoardId,
          ...data
        })
        setTasks(prev => [...prev, task])
        notify({ message: 'Task created' })
      } catch {
        notifyError(new Error('Failed to create task'), 'Failed to create task')
      }
    },
    [activeBoardId]
  )

  // Update task
  const handleUpdateTask = useCallback(
    async (data: { title: string; description?: string; priority?: KanbanPriority; assignee?: string; status?: KanbanStatus }) => {
      if (!editingTask) return
      try {
        const updated = await window.hermesDesktop.kanban.updateTask(editingTask.id, data)
        setTasks(prev => prev.map(t => (t.id === updated.id ? updated : t)))
        setSelectedTask(prev => (prev?.id === updated.id ? updated : prev))
        notify({ message: 'Task updated' })
      } catch {
        notifyError(new Error('Failed to update task'), 'Failed to update task')
      }
    },
    [editingTask]
  )

  // Delete task
  const handleDeleteTask = useCallback(async () => {
    if (!deletingTask) return
    try {
      await window.hermesDesktop.kanban.deleteTask(deletingTask.id)
      setTasks(prev => prev.filter(t => t.id !== deletingTask.id))
      setSelectedTask(prev => (prev?.id === deletingTask.id ? null : prev))
      setDeletingTask(null)
      setShowDeleteConfirm(null)
      notify({ message: 'Task deleted' })
    } catch {
      notifyError(new Error('Failed to delete task'), 'Failed to delete task')
    }
  }, [deletingTask])

  // Archive task
  const handleArchiveTask = useCallback(async () => {
    if (!selectedTask) return
    try {
      await window.hermesDesktop.kanban.updateTask(selectedTask.id, { archived: true })
      setTasks(prev => prev.filter(t => t.id !== selectedTask.id))
      setSelectedTask(null)
      notify({ message: 'Task archived' })
    } catch {
      notifyError(new Error('Failed to archive task'), 'Failed to archive task')
    }
  }, [selectedTask])

  // Change task status (from detail panel)
  const handleStatusChange = useCallback(
    async (status: KanbanStatus) => {
      if (!selectedTask) return
      try {
        const updated = await window.hermesDesktop.kanban.updateTask(selectedTask.id, { status })
        setTasks(prev => prev.map(t => (t.id === updated.id ? updated : t)))
        setSelectedTask(updated)
      } catch {
        notifyError(new Error('Failed to update status'), 'Failed to update status')
      }
    },
    [selectedTask]
  )

  // Add comment
  const handleAddComment = useCallback(
    async (body: string) => {
      if (!selectedTask) return
      try {
        const comment = await window.hermesDesktop.kanban.addComment({
          taskId: selectedTask.id,
          author: selectedTask.assignee || 'User',
          body
        })
        setTaskComments(prev => [...prev, comment])
      } catch {
        notifyError(new Error('Failed to add comment'), 'Failed to add comment')
      }
    },
    [selectedTask]
  )

  // Delete comment
  const handleDeleteComment = useCallback(
    async (commentId: string) => {
      try {
        await window.hermesDesktop.kanban.deleteComment(commentId)
        setTaskComments(prev => prev.filter(c => c.id !== commentId))
      } catch {
        notifyError(new Error('Failed to delete comment'), 'Failed to delete comment')
      }
    },
    []
  )

  // Create board
  const handleCreateBoard = useCallback(async () => {
    if (!newBoardTitle.trim()) return
    try {
      const board = await window.hermesDesktop.kanban.createBoard({ title: newBoardTitle.trim() })
      setBoards(prev => [...prev, board])
      setActiveBoardId(board.id)
      setNewBoardTitle('')
      setShowNewBoard(false)
    } catch {
      notifyError(new Error('Failed to create board'), 'Failed to create board')
    }
  }, [newBoardTitle])

  if (loading) {
    return <PageLoader />
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-bg-primary)">
      {/* Header toolbar */}
      <div
        className={cn(
          'flex shrink-0 items-center gap-3 border-b border-(--ui-stroke-secondary)',
          'bg-(--ui-bg-primary) px-4 py-2'
        )}
      >
        {/* Board selector */}
        <div className="flex items-center gap-1.5">
          <Codicon name="project" size="1rem" className="text-(--ui-text-tertiary)" />
          <select
            className={cn(
              'h-7 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) px-2 text-xs text-(--ui-text-primary) outline-none',
              'focus:border-(--ui-stroke-tertiary) focus:ring-[0.125rem] focus:ring-ring/30'
            )}
            value={activeBoardId}
            onChange={e => setActiveBoardId(e.target.value)}
          >
            {boards.length === 0 && <option value="default">{t.desktop.kanban.defaultBoard}</option>}
            {boards.map(board => (
              <option key={board.id} value={board.id}>
                {board.title}
              </option>
            ))}
          </select>
        </div>

        {/* New board button */}
        <button
          className="flex h-7 w-7 items-center justify-center rounded text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)"
          onClick={() => setShowNewBoard(!showNewBoard)}
          title={t.desktop.kanban.newBoard}
        >
          <Codicon name="add" size="0.875rem" />
        </button>

        {/* New board inline form */}
        {showNewBoard && (
          <div className="flex items-center gap-1.5">
            <Input
              value={newBoardTitle}
              onChange={e => setNewBoardTitle(e.target.value)}
              placeholder={t.desktop.kanban.boardName}
              className="h-7 w-40 text-xs"
              autoFocus
            />
            <Button variant="default" size="xs" onClick={handleCreateBoard} disabled={!newBoardTitle.trim()}>
              Create
            </Button>
            <Button variant="ghost" size="xs" onClick={() => { setShowNewBoard(false); setNewBoardTitle('') }}>
              {t.desktop.kanban.cancel}
            </Button>
          </div>
        )}

        <span className="text-(--ui-stroke-secondary)">|</span>

        {/* New task button */}
        <Button
          variant="default"
          size="sm"
          onClick={() => {
            setEditingTask(null)
            setDialogOpen(true)
          }}
        >
          <Codicon name="add" size="0.75rem" />
          {t.desktop.kanban.newTask}
        </Button>

        {/* Stats */}
        <span className="ml-auto text-[0.65rem] text-(--ui-text-tertiary)">
          {t.desktop.kanban.taskCount(boardTasks.length)}
        </span>
      </div>

      {/* Workspace: board + detail panel */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Board pane */}
        <div className="min-w-0 flex-1 overflow-hidden">
          <div className="h-full overflow-x-auto overflow-y-hidden p-4">
            <DndContext
              sensors={dndSensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <div className="flex h-full min-w-max gap-4">
                {STATUS_COLUMNS.map(column => (
                  <KanbanColumn
                    key={column.id}
                    column={column}
                    tasks={tasksByStatus[column.id] ?? []}
                    onEditTask={task => {
                      setEditingTask(task)
                      setDialogOpen(true)
                    }}
                    onDeleteTask={task => {
                      setDeletingTask(task)
                      setShowDeleteConfirm(task)
                    }}
                    onSelectTask={task => setSelectedTask(task)}
                  />
                ))}
              </div>
            </DndContext>
          </div>
        </div>

        {/* Detail panel */}
        {selectedTask && (
          <aside className="h-full w-80 shrink-0 overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-bg-primary)">
            <TaskDetailPanel
              task={selectedTask}
              comments={taskComments}
              onClose={() => setSelectedTask(null)}
              onAddComment={handleAddComment}
              onDeleteComment={handleDeleteComment}
              onArchive={handleArchiveTask}
              onStatusChange={handleStatusChange}
            />
          </aside>
        )}
      </div>

      {/* Create / Edit dialog */}
      <TaskDialog
        open={dialogOpen}
        onOpenChange={open => {
          setDialogOpen(open)
          if (!open) setEditingTask(null)
        }}
        task={editingTask}
        onSave={data => {
          if (editingTask) {
            void handleUpdateTask(data)
          } else {
            void handleCreateTask(data)
          }
        }}
      />

      {/* Delete confirmation dialog */}
      <Dialog open={!!showDeleteConfirm} onOpenChange={open => { if (!open) { setShowDeleteConfirm(null); setDeletingTask(null) } }}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>{t.desktop.kanban.deleteTask}</DialogTitle>
            <DialogDescription>
              {t.desktop.kanban.deleteConfirm(showDeleteConfirm?.title ?? '')}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => { setShowDeleteConfirm(null); setDeletingTask(null) }}>
              {t.desktop.kanban.cancel}
            </Button>
            <Button variant="destructive" onClick={() => void handleDeleteTask()}>
              <Codicon name="trash" size="0.75rem" />
              {t.desktop.kanban.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
