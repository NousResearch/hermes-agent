/**
 * The Kanban board page — mounted at `/kanban` (a ROUTES_AREA contribution) in
 * the workspace pane. The desktop port of the dashboard board: one compact
 * header row (count, filter kebab, search, settings, new task — the board
 * SWITCHER lives in the titlebar, see board-switcher.tsx), columns in
 * BOARD_COLUMNS order, drag-to-move (optimistic, workflow-checked),
 * primary-modifier-click multi-select with a floating bulk bar, right-click
 * actions, and the detail drawer. Dispatch nudges ride every write (see api.ts).
 */

import {
  atom,
  Button,
  cn,
  Codicon,
  compactNumber,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
  Contribute,
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  ErrorState,
  formatModifierToken,
  host,
  Input,
  isSubmitEnter,
  Loader,
  SearchField,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Textarea,
  Tip,
  useGrabScroll,
  useMutation,
  useQuery,
  useQueryClient,
  useValue,
  WORKSPACE_PAGE_HEADER_AREA
} from '@hermes/plugin-sdk'
import {
  type CSSProperties,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import {
  $boardSlug,
  $collapsedLanes,
  $introDismissed,
  $lanesByProfile,
  addComment,
  boardKey,
  boardKeyPrefix,
  boardsKey,
  bulkTasks,
  createTask,
  deleteTask,
  estimateNew,
  fetchBoard,
  fetchBoards,
  fetchProfiles,
  patchTask,
  profilesKey,
  reassignTask,
  taskKey,
  useKanbanScope
} from './api'
import { BoardSwitcher } from './board-switcher'
import { TaskDrawer } from './drawer'
import { EMPTY_OVERRIDE, ModelOverrideField, overrideCreateFields, type TaskModelOverride } from './model-override'
import { OrchestrationPanel } from './orchestration'
import { columnMeta, type KanbanBoard, type KanbanTask, type TaskEstimate } from './types'
import {
  $newTaskLane,
  ago,
  type ArcState,
  arcState,
  Avatar,
  columnHelp,
  columnLabel,
  errText,
  FIELD_LABEL,
  fmtSecs,
  isLockedTarget,
  type KanbanText,
  lockedReason,
  PriorityGlyph,
  RunClock,
  runtimeCapBadge,
  shortId,
  staleBlocked,
  useDefaultAssignee,
  useKanban,
  useNowSecs,
  useOrchestration
} from './ui'

// ── optimistic board edits (reconciled by the follow-up refresh) ─────────────

function moveCard(board: KanbanBoard, id: string, toStatus: string): KanbanBoard {
  let moved: KanbanTask | undefined

  const columns = board.columns.map(col => ({
    ...col,
    tasks: col.tasks.filter(task => {
      if (task.id !== id) {
        return true
      }

      moved = { ...task, status: toStatus }

      return false
    })
  }))

  if (!moved) {
    return board
  }

  return {
    ...board,
    columns: columns.map(col => (col.name === toStatus ? { ...col, tasks: [moved!, ...col.tasks] } : col))
  }
}

function removeCard(board: KanbanBoard, id: string): KanbanBoard {
  return { ...board, columns: board.columns.map(col => ({ ...col, tasks: col.tasks.filter(t => t.id !== id) })) }
}

// ── card actions (right-click menu) ─────────────────────────────────────────

/**
 * The card menu's action set, data-tabled: one row per action, with its
 * visibility rule and whether it may fan out over a multi-card selection.
 * Non-batch-safe rows render disabled (not hidden) while several cards are
 * selected — the menu stays a stable map of what a card can do.
 */
export type CardActionKey =
  | 'addChild'
  | 'addLink'
  | 'block'
  | 'comment'
  | 'reassign'
  | 'requestChanges'
  | 'requestReview'
  | 'unblock'

export interface CardActionDef {
  batchSafe: boolean
  icon: string
  key: CardActionKey
  label: (k: KanbanText) => string
  when: (status: string) => boolean
}

export const CARD_ACTIONS: readonly CardActionDef[] = [
  // `when` mirrors the backend state machine's accepted source states, so the
  // menu never offers a verb the endpoint would reject with a 409:
  // block_task takes running/ready only; request_review takes running/ready.
  { batchSafe: false, icon: 'circle-slash', key: 'block', label: k => k.actBlock, when: s => s === 'running' || s === 'ready' },
  { batchSafe: true, icon: 'debug-continue', key: 'unblock', label: k => k.actUnblock, when: s => s === 'blocked' },
  { batchSafe: true, icon: 'eye', key: 'requestReview', label: k => k.actRequestReview, when: s => s === 'running' || s === 'ready' },
  {
    batchSafe: false,
    icon: 'request-changes',
    key: 'requestChanges',
    label: k => k.actRequestChanges,
    when: s => s === 'review'
  },
  { batchSafe: false, icon: 'comment', key: 'comment', label: k => k.actComment, when: () => true },
  { batchSafe: true, icon: 'account', key: 'reassign', label: k => k.actReassign, when: () => true },
  { batchSafe: false, icon: 'link', key: 'addLink', label: k => k.actAddLink, when: () => true },
  { batchSafe: false, icon: 'add', key: 'addChild', label: k => k.actAddChild, when: () => true }
]

type PromptKind = 'block' | 'changes' | 'comment'

/**
 * Copy + submit for the menu dialogs that collect a note. `block` flips status
 * with the reason; `changes` posts the must-fix note first, then hands the card
 * back to its queue (leaving review reopens it, per the backend transition).
 */
const PROMPTS: Record<
  PromptKind,
  {
    confirm: (k: KanbanText) => string
    label: (k: KanbanText) => string
    placeholder: (k: KanbanText) => string
    submit: (id: string, text: string) => Promise<unknown>
    title: (k: KanbanText) => string
  }
> = {
  block: {
    confirm: k => k.blockConfirm,
    label: k => k.reasonLabel,
    placeholder: k => k.blockPlaceholder,
    submit: (id, text) => patchTask(id, { block_reason: text, status: 'blocked' }),
    title: k => k.blockTitle
  },
  changes: {
    confirm: k => k.changesConfirm,
    label: k => k.changesLabel,
    placeholder: k => k.changesPlaceholder,
    submit: async (id, text) => {
      await addComment(id, text)

      return patchTask(id, { status: 'todo' })
    },
    title: k => k.changesTitle
  },
  comment: {
    confirm: k => k.send,
    label: k => k.comment,
    placeholder: k => k.addComment,
    submit: (id, text) => addComment(id, text),
    title: k => k.commentTitle
  }
}

/** One-shot note dialog shared by block / request-changes / comment. */
function CardPromptDialog({ entity, onClose }: { entity: { id: string; kind: PromptKind }; onClose: () => void }) {
  const k = useKanban()
  const qc = useQueryClient()
  const [text, setText] = useState('')
  const copy = PROMPTS[entity.kind]

  const submit = useMutation({
    mutationFn: () => copy.submit(entity.id, text.trim()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
      void qc.invalidateQueries({ queryKey: ['kanban', 'task'] })
      onClose()
    }
  })

  return (
    <Dialog onOpenChange={open => !open && onClose()} open>
      <DialogContent className="w-[min(30rem,92vw)]">
        <DialogHeader>
          <DialogTitle>{copy.title(k)}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          <span className={FIELD_LABEL}>{copy.label(k)}</span>
          <Textarea
            autoFocus
            className="min-h-24"
            onChange={event => setText(event.target.value)}
            placeholder={copy.placeholder(k)}
            value={text}
          />
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="ghost">
            {k.cancel}
          </Button>
          <Button disabled={!text.trim() || submit.isPending} onClick={() => submit.mutate()}>
            {copy.confirm(k)}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ── card ─────────────────────────────────────────────────────────────────────

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

function Card({
  assignees,
  columns,
  onAct,
  onDelete,
  onMove,
  onOpen,
  onToggleSelect,
  scope,
  selected,
  task
}: {
  assignees: string[]
  columns: string[]
  onAct: (key: CardActionKey, ids: string[], value?: string) => void
  onDelete: (id: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggleSelect: (id: string) => void
  scope: string[]
  selected: boolean
  task: KanbanTask
}) {
  const k = useKanban()
  const [dragging, setDragging] = useState(false)
  const meta = columnMeta(task.status)
  const summary = task.latest_summary || task.body
  const fallback = useDefaultAssignee()
  const arc = arcState(task, fallback)
  const now = useNowSecs(task.status === 'running')
  const cap = runtimeCapBadge(task, now)
  const stale = staleBlocked(task, now)

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
          style={
            {
              '--kanban-tone': meta.tone,
              borderLeftColor: cap?.kind === 'over' ? 'var(--destructive, #f87171)' : meta.tone
            } as CSSProperties
          }
        >
          {/* Machine-activity arc: animates ONLY while an agent is actually on
              the card (claimed + working; amber when the heartbeat is gone).
              Queued attachment is the footer's named-agent chip — a moving
              border on an idle card would lie. Hidden during drag/selection
              so those states stay legible. */}
          {(arc === 'running' || arc === 'stale') && !dragging && !selected && (
            <span aria-hidden className={cn('kanban-arc', arc === 'stale' && 'kanban-arc--stale')} />
          )}
          <span className="flex items-start gap-2 text-[0.8125rem] font-medium leading-snug text-foreground">
            <span className="line-clamp-2 min-w-0 flex-1">{task.title || task.id}</span>
            {cap && (
              <Tip label={cap.kind === 'over' ? k.overCapTip : k.nearCapTip}>
                <span
                  className={cn(
                    'shrink-0 cursor-help rounded px-1 py-px text-[0.5625rem] font-semibold tabular-nums',
                    cap.kind === 'over' ? 'bg-destructive/15 text-destructive' : 'bg-amber-500/15 text-amber-500'
                  )}
                >
                  {cap.kind === 'over' ? k.overCap : k.nearCap} · {fmtSecs(cap.elapsed)}/{fmtSecs(cap.cap)}
                </span>
              </Tip>
            )}
            {stale && !cap && (
              <Tip label={k.staleBlockedTip}>
                <span
                  className="mt-1 size-1.5 shrink-0 cursor-help rounded-full"
                  style={{ backgroundColor: 'var(--ui-text-quaternary)' }}
                />
              </Tip>
            )}
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
        {CARD_ACTIONS.filter(action => action.when(task.status)).map(action => {
          // A multi-card selection keeps non-batch-safe rows visible but inert.
          const inert = scope.length > 1 && !action.batchSafe

          if (action.key === 'reassign') {
            return (
              <ContextMenuSub key={action.key}>
                <ContextMenuSubTrigger>{action.label(k)}</ContextMenuSubTrigger>
                <ContextMenuSubContent>
                  {assignees.map(name => (
                    <ContextMenuItem key={name} onSelect={() => onAct('reassign', scope, name)}>
                      <Avatar name={name} size="0.875rem" />
                      {name}
                    </ContextMenuItem>
                  ))}
                </ContextMenuSubContent>
              </ContextMenuSub>
            )
          }

          return (
            <ContextMenuItem
              disabled={inert}
              key={action.key}
              onSelect={() => onAct(action.key, scope)}
              title={inert ? k.actMultiTip : undefined}
            >
              <Codicon name={action.icon} size="0.85rem" />
              {action.label(k)}
            </ContextMenuItem>
          )
        })}
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

// ── column ───────────────────────────────────────────────────────────────────

function Column({
  assignees,
  collapsed,
  column,
  columns,
  onAct,
  onAdd,
  onDelete,
  onDropTask,
  onMove,
  onOpen,
  onToggle,
  onToggleSelect,
  selected
}: {
  assignees: string[]
  collapsed: boolean
  column: { name: string; tasks: KanbanTask[] }
  columns: string[]
  onAct: (key: CardActionKey, ids: string[], value?: string) => void
  onAdd: (status: string) => void
  onDelete: (id: string) => void
  onDropTask: (id: string, status: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggle: () => void
  onToggleSelect: (id: string) => void
  selected: ReadonlySet<string>
}) {
  const k = useKanban()
  const [over, setOver] = useState(false)
  const meta = columnMeta(column.name)
  const label = columnLabel(k, column.name)
  const locked = isLockedTarget(column.name)
  const byProfile = useValue($lanesByProfile)

  // The dashboard's "lanes by profile": sub-group Running by assignee so a
  // fleet's in-flight work reads per-worker. Null = flat (off, or trivial).
  const lanes = useMemo(() => {
    if (!byProfile || column.name !== 'running' || column.tasks.length === 0) {
      return null
    }

    const groups = new Map<string, KanbanTask[]>()

    for (const task of column.tasks) {
      const key = task.assignee || UNASSIGNED_LANE
      groups.set(key, [...(groups.get(key) ?? []), task])
    }

    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [byProfile, column])

  const dragHandlers = {
    onDragLeave: () => setOver(false),
    onDragOver: (event: ReactDragEvent<HTMLElement>) => {
      // Locked lanes don't preventDefault → the OS shows the no-drop cursor
      // and the drop event never fires. The lane is honest about itself.
      if (locked) {
        event.dataTransfer.dropEffect = 'none'

        return
      }

      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      setOver(true)
    },
    onDrop: (event: ReactDragEvent<HTMLElement>) => {
      event.preventDefault()
      setOver(false)
      const id = event.dataTransfer.getData('text/plain')

      if (id) {
        onDropTask(id, column.name)
      }
    }
  }

  const wash = over && !locked ? 'bg-(--ui-bg-quinary)' : 'bg-[color-mix(in_srgb,var(--ui-bg-quinary)_50%,transparent)]'

  // Collapsed = a thin vertical rail: dot, sideways label, count. Still a live
  // drop target (drop straight onto the rail); click expands. The dot sits in
  // the same h-5 header row as an expanded lane's, so dots align across the
  // board regardless of collapse state.
  if (collapsed) {
    return (
      <button
        {...dragHandlers}
        aria-label={k.expand(label)}
        className={cn(
          'flex h-full w-8 shrink-0 flex-col items-center gap-1.5 rounded-lg p-2 transition-colors hover:bg-(--ui-bg-quinary)',
          wash
        )}
        onClick={onToggle}
        type="button"
      >
        <span className="grid h-5 shrink-0 place-items-center">
          <span className="size-1.5 rounded-full" style={{ backgroundColor: meta.tone }} />
        </span>
        <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary) [writing-mode:vertical-rl]">
          {label}
        </span>
        {column.tasks.length > 0 && (
          <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
        )}
      </button>
    )
  }

  return (
    <div
      {...dragHandlers}
      className={cn('group/col flex h-full w-64 shrink-0 flex-col rounded-lg p-2 transition-colors', wash)}
    >
      <header className="mb-1.5 flex h-5 items-center gap-1.5 px-1">
        <span className="size-1.5 rounded-full" style={{ backgroundColor: meta.tone }} />
        <Tip label={columnHelp(k, column.name)}>
          <span className="cursor-help text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
            {label}
          </span>
        </Tip>
        <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
        <button
          aria-label={k.collapse(label)}
          className="ml-auto grid size-5 place-items-center rounded text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100 group-hover/col:opacity-100"
          onClick={onToggle}
          type="button"
        >
          <Codicon name="chevron-left" size="0.75rem" />
        </button>
      </header>
      <div className="relative flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
        {lanes
          ? lanes.map(([assignee, tasks]) => (
              <div className="flex flex-col gap-2" key={assignee}>
                <div className="flex items-center gap-1.5 px-1 pt-1 text-[0.625rem] text-(--ui-text-quaternary)">
                  {assignee !== UNASSIGNED_LANE && <Avatar name={assignee} size="0.875rem" />}
                  {assignee}
                  <span className="tabular-nums">{tasks.length}</span>
                </div>
                {tasks.map(task => (
                  <Card
                    assignees={assignees}
                    columns={columns}
                    key={task.id}
                    onAct={onAct}
                    onDelete={onDelete}
                    onMove={onMove}
                    onOpen={onOpen}
                    onToggleSelect={onToggleSelect}
                    scope={selected.has(task.id) ? [...selected] : [task.id]}
                    selected={selected.has(task.id)}
                    task={task}
                  />
                ))}
              </div>
            ))
          : column.tasks.map(task => (
              <Card
                assignees={assignees}
                columns={columns}
                key={task.id}
                onAct={onAct}
                onDelete={onDelete}
                onMove={onMove}
                onOpen={onOpen}
                onToggleSelect={onToggleSelect}
                scope={selected.has(task.id) ? [...selected] : [task.id]}
                selected={selected.has(task.id)}
                task={task}
              />
            ))}
        {/* Jira-style lane add — dashed, faded in on lane hover. Opacity (not
            display) so it always holds its slot and never thrashes layout.
            Locked lanes get none: you can't create into a system state. */}
        {!locked && (
          <button
            aria-label={k.newTaskIn(label)}
            className="flex shrink-0 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary) py-1.5 text-(--ui-text-tertiary) opacity-0 transition-[opacity,color,border-color] group-hover/col:opacity-100 hover:border-(--ui-text-quaternary) hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100"
            onClick={() => onAdd(column.name)}
            type="button"
          >
            <Codicon name="add" size="0.8rem" />
          </button>
        )}
        {column.tasks.length === 0 && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-[0.6875rem] text-(--ui-text-quaternary)">
            {k.empty}
          </div>
        )}
      </div>
    </div>
  )
}

// ── dialogs ──────────────────────────────────────────────────────────────────

const NO_PARENT = '__none__'
const PARKED = '__parked__'
const WORKSPACE_KINDS = ['scratch', 'worktree', 'dir'] as const

function Field({ children, label }: { children: ReactNode; label: string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </label>
  )
}

function NewTaskDialog({
  onClose,
  parents,
  target
}: {
  onClose: () => void
  parents: Array<{ id: string; title: string }>
  target: null | string
}) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const { data: roster } = useQuery({ queryKey: profilesKey(scope), queryFn: fetchProfiles, staleTime: 60_000 })
  // Title-only creates must RUN: "auto" resolves to the orchestration default
  // (ultimately the active profile), applied at create time. Never silently
  // unassigned — parking a card is the explicit choice, not the default.
  const resolvedDefault = useOrchestration()?.resolved_default_assignee || 'default'

  // Board-level workspace default: a task inherits the current board's
  // configured project dir (scratch when unset, worktree in a git repo, else
  // dir) unless the operator overrides it below. Set the board default in the
  // board switcher's "Board settings…".
  const selectedSlug = useValue($boardSlug)
  const { data: boards } = useQuery({ queryKey: boardsKey(scope), queryFn: fetchBoards, staleTime: 30_000 })
  const currentBoard = boards?.boards.find(b => b.slug === (selectedSlug || boards.current))
  const boardDefaultKind = currentBoard?.default_workspace_kind || 'scratch'
  const boardDefaultDir = currentBoard?.default_workdir || ''

  const isTriage = target === 'triage'
  const [title, setTitle] = useState('')
  const [bodyText, setBodyText] = useState('')
  const [assignee, setAssignee] = useState('')
  const [priority, setPriority] = useState('0')
  const [skills, setSkills] = useState('')
  const [workspaceKind, setWorkspaceKind] = useState<string>(boardDefaultKind)
  // Empty = inherit the board's default project dir (backend resolves it);
  // a path here overrides just this task. Only meaningful for dir/worktree.
  const [workspacePath, setWorkspacePath] = useState('')
  const [parent, setParent] = useState('')
  const [modelOverride, setModelOverride] = useState<TaskModelOverride>(EMPTY_OVERRIDE)
  const [goalMode, setGoalMode] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const [estimate, setEstimate] = useState<null | TaskEstimate>(null)

  // Rough effort estimate from the typed title/body (before the task exists),
  // via the auto-routed auxiliary model. Makes a model call — explicit action.
  const estMut = useMutation({
    mutationFn: () => estimateNew(title.trim(), bodyText.trim()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: r => {
      if (r.ok) {
        setEstimate(r)
      } else {
        host.notify({ kind: 'warning', message: r.reason || k.couldNotEstimate })
      }
    }
  })

  // Reset per open — the dialog is externally controlled (open = target set),
  // so onOpenChange(true) never fires; key the reset off `target` (and the
  // resolved board default, which may arrive after the first open).
  useEffect(() => {
    if (target) {
      setTitle('')
      setBodyText('')
      setAssignee('')
      setPriority('0')
      setSkills('')
      setWorkspaceKind(boardDefaultKind)
      setWorkspacePath('')
      setParent('')
      setModelOverride(EMPTY_OVERRIDE)
      setGoalMode(false)
      setError(null)
      setBusy(false)
      setEstimate(null)
    }
  }, [target, boardDefaultKind])

  const submit = async () => {
    const trimmed = title.trim()

    if (!trimmed || !target || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const skillList = skills
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)

      // create() derives status (triage flag → 'triage', else 'ready'); move to
      // the requested column when they differ, so a per-column add lands right.
      const { task, warning } = await createTask({
        assignee: assignee === PARKED ? undefined : assignee || resolvedDefault,
        body: bodyText.trim() || undefined,
        goal_mode: goalMode,
        parents: parent ? [parent] : undefined,
        priority: Number(priority) || 0,
        skills: skillList.length ? skillList : undefined,
        title: trimmed,
        triage: isTriage,
        workspace_kind: workspaceKind,
        ...overrideCreateFields(modelOverride),
        // Empty → backend inherits the board's default project dir.
        workspace_path: workspaceKind !== 'scratch' && workspacePath.trim() ? workspacePath.trim() : undefined
      })

      if (task && task.status !== target) {
        await patchTask(task.id, { status: target })
      }

      // Dispatcher-presence warning ("this ready task will sit idle") — not an
      // error, but the user should know.
      if (warning) {
        host.notify({ kind: 'warning', message: warning })
      }

      await qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
      onClose()
    } catch (err) {
      setError(errText(err))
      setBusy(false)
    }
  }

  return (
    <Dialog onOpenChange={open => !open && onClose()} open={Boolean(target)}>
      {/* `overflow-visible`: DialogContent publishes ITSELF as the portal
          container for popovers opened inside it (dialog-portal-context), and
          its default `overflow-y-auto` then crops them at the dialog's edge —
          the model menu below is born inside that scroll box. This dialog
          already owns a scroller on its body div, so the shell's clip is
          redundant here and dropping it is safe. The general fix to
          DialogContent is in flight as #75600; when that lands this override
          becomes a no-op and can go. */}
      <DialogContent className="w-[min(42rem,94vw)] max-w-none overflow-visible">
        <DialogHeader>
          <DialogTitle>{target ? k.newTaskIn(columnLabel(k, target)) : k.newTask}</DialogTitle>
        </DialogHeader>
        <div className="flex max-h-[min(72vh,44rem)] flex-col gap-3 overflow-y-auto pr-0.5">
          <Input
            autoFocus
            onChange={event => setTitle(event.target.value)}
            onKeyDown={event => {
              if (isSubmitEnter(event)) {
                event.preventDefault()
                void submit()
              }
            }}
            placeholder={isTriage ? k.titlePlaceholderTriage : k.titlePlaceholder}
            value={title}
          />
          <Textarea
            className="min-h-20"
            onChange={event => setBodyText(event.target.value)}
            placeholder={k.descPlaceholder}
            value={bodyText}
          />

          <div className="grid grid-cols-2 gap-3">
            <Field label={k.priority}>
              <Input onChange={event => setPriority(event.target.value)} type="number" value={priority} />
            </Field>
            <Field label={k.workspace}>
              <Select onValueChange={setWorkspaceKind} value={workspaceKind}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {WORKSPACE_KINDS.map(kind => (
                    <SelectItem key={kind} value={kind}>
                      {kind}
                      {kind === boardDefaultKind ? k.boardDefaultSuffix : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          {workspaceKind !== 'scratch' && (
            <Field label={k.workspaceOverride}>
              <Input
                onChange={event => setWorkspacePath(event.target.value)}
                placeholder={boardDefaultDir || k.workspaceInherit}
                value={workspacePath}
              />
              <span className="text-[0.625rem] text-(--ui-text-quaternary)">
                {boardDefaultDir ? k.workspaceInheritDir(boardDefaultDir) : k.workspaceInheritGeneric}
              </span>
            </Field>
          )}

          <Field label={k.assignee}>
            <Select onValueChange={v => setAssignee(v === NO_PARENT ? '' : v)} value={assignee || NO_PARENT}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={NO_PARENT}>{k.defaultOption(resolvedDefault)}</SelectItem>
                {(roster?.profiles ?? [])
                  .filter(profile => profile.name !== resolvedDefault)
                  .map(profile => (
                    <SelectItem key={profile.name} value={profile.name}>
                      {profile.name}
                    </SelectItem>
                  ))}
                <SelectItem value={PARKED}>{k.parkedOption}</SelectItem>
              </SelectContent>
            </Select>
          </Field>

          <Field label={k.skills}>
            <Input onChange={event => setSkills(event.target.value)} placeholder={k.skillsPlaceholder} value={skills} />
          </Field>

          <Field label={k.model}>
            <ModelOverrideField onChange={setModelOverride} value={modelOverride} />
            <span className="text-[0.625rem] text-(--ui-text-quaternary)">{k.modelHint}</span>
          </Field>

          {parents.length > 0 && (
            <Field label={k.parent}>
              <Select onValueChange={v => setParent(v === NO_PARENT ? '' : v)} value={parent || NO_PARENT}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_PARENT}>{k.noParent}</SelectItem>
                  {parents.map(option => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.title || option.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          )}

          <label className="flex cursor-pointer items-center gap-2 text-[0.75rem] text-(--ui-text-secondary)">
            <Switch aria-label={k.goalMode} checked={goalMode} onCheckedChange={setGoalMode} size="xs" />
            {k.goalMode}
          </label>

          {error && <span className="text-[0.75rem] text-destructive">{error}</span>}
        </div>
        <DialogFooter>
          <div className="mr-auto flex items-center gap-1 text-[0.75rem] text-(--ui-text-tertiary)">
            {estimate?.ok ? (
              <>
                <Tip label={estimate.rationale || k.roughEstimate}>
                  <span className="font-medium tabular-nums text-(--ui-text-secondary)">
                    ~{compactNumber(estimate.est_tokens)} {k.tokUnit}
                    {estimate.complexity ? ` · ${k.complexity[estimate.complexity] ?? estimate.complexity}` : ''}
                  </span>
                </Tip>
                <Tip label={k.reEstimate}>
                  <Button
                    aria-label={k.reEstimate}
                    disabled={!title.trim() || estMut.isPending}
                    onClick={() => estMut.mutate()}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="refresh" size="0.7rem" spinning={estMut.isPending} />
                  </Button>
                </Tip>
              </>
            ) : (
              <Tip label={k.estimateTip}>
                <Button
                  disabled={!title.trim() || estMut.isPending}
                  onClick={() => estMut.mutate()}
                  size="xs"
                  variant="ghost"
                >
                  <Codicon
                    name={estMut.isPending ? 'loading' : 'dashboard'}
                    size="0.75rem"
                    spinning={estMut.isPending}
                  />
                  {estMut.isPending ? k.estimating : k.estimate}
                </Button>
              </Tip>
            )}
          </div>
          <Button onClick={onClose} variant="text">
            {k.cancel}
          </Button>
          <Button disabled={!title.trim() || busy} onClick={() => void submit()}>
            {busy ? k.creating : k.createTask}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ── intro ────────────────────────────────────────────────────────────────────

// One-time explainer for the board's core gotcha: this is a dispatcher queue,
// not a todo list. Dismissal persists via plugin storage.
function Intro() {
  const k = useKanban()
  const dismissed = useValue($introDismissed)

  if (dismissed) {
    return null
  }

  return (
    <div
      className="mx-4 mb-2 flex flex-col items-start gap-1.5 rounded-lg bg-(--ui-bg-quinary) px-3 py-2.5 text-[0.75rem] leading-relaxed text-(--ui-text-secondary)"
      data-selectable-text="true"
    >
      <p className="min-w-0">{k.introBody}</p>
      <Button onClick={() => $introDismissed.set(true)} size="inline" variant="textStrong">
        {k.introGotIt}
      </Button>
    </div>
  )
}

const UNASSIGNED_LANE = 'unassigned'

// ── filter facets (status / priority / triage) ───────────────────────────────

/**
 * The URL-facing facet filters. They live in atoms (the filter kebab writes,
 * the page reads) and round-trip through the hash query, so
 * `#/kanban?status=todo,blocked&priority=high&triage=1` opens the same view
 * in another window.
 */
export const $filterStatus = atom<string[]>([])
export const $filterPriority = atom<number[]>([])
export const $filterTriage = atom<boolean>(false)

export interface FilterQuery {
  priority: number[]
  status: string[]
  triage: boolean
}

interface PriorityFacet {
  key: string
  label: (k: KanbanText) => string
  value: number
}

export const PRIORITY_FACETS: readonly PriorityFacet[] = [
  { key: 'high', label: k => k.priorityHigh, value: 5 },
  { key: 'normal', label: k => k.priorityNormal, value: 0 },
  { key: 'low', label: k => k.priorityLow, value: -5 }
]

/** Read the facets out of a `#/route?query` hash; unknown values drop. */
export function parseFilterQuery(hash: string): FilterQuery {
  const query = new URLSearchParams(hash.split('?')[1] ?? '')
  const status = (query.get('status') ?? '').split(',').filter(Boolean)

  const priority = (query.get('priority') ?? '')
    .split(',')
    .flatMap(token => PRIORITY_FACETS.filter(facet => facet.key === token).map(facet => facet.value))

  return { priority, status, triage: query.get('triage') === '1' }
}

/** Merge the facets back into a hash, preserving the route and other params. */
export function buildFilterQuery(hash: string, facets: FilterQuery): string {
  const [route = '', query = ''] = hash.split('?')
  const path = route.replace(/^#/, '')
  const params = new URLSearchParams(query)
  const set = (key: string, value: string) => (value ? params.set(key, value) : params.delete(key))

  set('status', facets.status.join(','))
  set(
    'priority',
    facets.priority
      .flatMap(value => PRIORITY_FACETS.filter(facet => facet.value === value).map(facet => facet.key))
      .join(',')
  )
  set('triage', facets.triage ? '1' : '')

  const next = params.toString().replaceAll('%2C', ',')

  return next ? `${path}?${next}` : path
}

/** One task against the facet filters — an empty dimension passes everything. */
export function matchesFacetFilters(task: KanbanTask, facets: FilterQuery): boolean {
  return (
    (facets.status.length === 0 || facets.status.includes(task.status)) &&
    (facets.priority.length === 0 || facets.priority.includes(task.priority ?? 0)) &&
    (!facets.triage || task.triage_signal === true)
  )
}

// ── filter kebab ─────────────────────────────────────────────────────────────

function FilterMenu({
  archived,
  assignee,
  board,
  onArchived,
  onAssignee,
  onTenant,
  tenant
}: {
  archived: boolean
  assignee: string
  board: KanbanBoard
  onArchived: (v: boolean) => void
  onAssignee: (v: string) => void
  onTenant: (v: string) => void
  tenant: string
}) {
  const k = useKanban()
  const lanesByProfile = useValue($lanesByProfile)
  const statusFacets = useValue($filterStatus)
  const priorityFacets = useValue($filterPriority)
  const triageFacet = useValue($filterTriage)

  const active = Boolean(
    assignee || tenant || archived || statusFacets.length > 0 || priorityFacets.length > 0 || triageFacet
  )

  const toggle = <T,>(list: T[], value: T): T[] =>
    list.includes(value) ? list.filter(entry => entry !== value) : [...list, value]

  const check = (on: boolean) => (on ? <Codicon className="ml-auto" name="check" size="0.8rem" /> : null)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={k.filters}
          className={cn(active && 'bg-(--ui-control-active-background) text-foreground')}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="filter" size="0.85rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem onSelect={() => onAssignee('')}>
          {k.allProfiles}
          {check(!assignee)}
        </DropdownMenuItem>
        {board.assignees.map(name => (
          <DropdownMenuItem key={name} onSelect={() => onAssignee(name)}>
            <Avatar name={name} size="0.875rem" />
            {name}
            {check(assignee === name)}
          </DropdownMenuItem>
        ))}
        {board.tenants.length > 0 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => onTenant('')}>
              {k.allTenants}
              {check(!tenant)}
            </DropdownMenuItem>
            {board.tenants.map(name => (
              <DropdownMenuItem key={name} onSelect={() => onTenant(name)}>
                {name}
                {check(tenant === name)}
              </DropdownMenuItem>
            ))}
          </>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => onArchived(!archived)}>
          {k.showArchived}
          {check(archived)}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => $lanesByProfile.set(!lanesByProfile)}>
          {k.groupRunning}
          {check(lanesByProfile)}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <div className="px-2 py-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
          {k.filterStatus}
        </div>
        {board.columns.map(col => (
          <DropdownMenuItem key={col.name} onSelect={() => $filterStatus.set(toggle(statusFacets, col.name))}>
            <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(col.name).tone }} />
            {columnLabel(k, col.name)}
            {check(statusFacets.includes(col.name))}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <div className="px-2 py-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
          {k.priority}
        </div>
        {PRIORITY_FACETS.map(facet => (
          <DropdownMenuItem key={facet.key} onSelect={() => $filterPriority.set(toggle(priorityFacets, facet.value))}>
            {facet.label(k)}
            {check(priorityFacets.includes(facet.value))}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => $filterTriage.set(!triageFacet)} title={k.needsTriageTitle}>
          {k.filterNeedsTriage}
          {check(triageFacet)}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// ── selection bar ────────────────────────────────────────────────────────────

/**
 * Floating bulk-actions bar, shown while cards are ⌘-selected. Deliberately
 * leaner than the dashboard's always-on toolbar: move / assign / archive /
 * delete cover the real fleet chores (requeue a batch, archive a sweep of
 * done, reassign after a profile change) via POST /tasks/bulk, which applies
 * per-id and reports partial failures — failed cards stay selected.
 */
function SelectionBar({
  columns,
  onClear,
  onDone,
  selected
}: {
  columns: string[]
  onClear: () => void
  onDone: (failed: string[]) => void
  selected: ReadonlySet<string>
}) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const { data: roster } = useQuery({ queryKey: profilesKey(scope), queryFn: fetchProfiles, staleTime: 60_000 })

  const finish = (failed: Array<{ error?: string; id: string }>) => {
    void qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })

    if (failed.length > 0) {
      host.notify({
        kind: 'warning',
        message: k.bulkFailed(failed.length, selected.size, failed[0].error ?? k.refused)
      })
    }

    onDone(failed.map(f => f.id))
  }

  const bulk = useMutation({
    mutationFn: (patch: Record<string, unknown>) => bulkTasks([...selected], patch),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: data => finish(data.results.filter(r => !r.ok))
  })

  // No bulk-delete on the backend — fan out per id, same partial-failure story.
  const bulkDelete = useMutation({
    mutationFn: async () => {
      const ids = [...selected]
      const settled = await Promise.allSettled(ids.map(id => deleteTask(id)))

      return ids.flatMap((id, i) => {
        const result = settled[i]

        return result.status === 'rejected' ? [{ error: errText(result.reason), id }] : []
      })
    },
    onSuccess: finish
  })

  const busy = bulk.isPending || bulkDelete.isPending
  // One menu at a time — controlled, so a click on the second trigger can
  // never race Radix's dismiss layer into two open menus.
  const [menu, setMenu] = useState<'assign' | 'move' | null>(null)

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-10 flex justify-center px-4">
      {/* Flat overlay: stroke + elevated surface do the separating, no shadow. */}
      <div className="pointer-events-auto flex items-center gap-1 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) py-1 pr-1 pl-3">
        <span className="mr-1 text-xs tabular-nums text-(--ui-text-secondary)">{k.nSelected(selected.size)}</span>

        <DropdownMenu onOpenChange={open => setMenu(open ? 'move' : null)} open={menu === 'move'}>
          <DropdownMenuTrigger asChild>
            <Button disabled={busy} size="xs" variant="ghost">
              {k.moveToShort}
              <Codicon name="chevron-down" size="0.7rem" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center">
            {columns
              .filter(name => !isLockedTarget(name))
              .map(name => (
                <DropdownMenuItem key={name} onSelect={() => bulk.mutate({ status: name })}>
                  <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
                  {columnLabel(k, name)}
                </DropdownMenuItem>
              ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu onOpenChange={open => setMenu(open ? 'assign' : null)} open={menu === 'assign'}>
          <DropdownMenuTrigger asChild>
            <Button disabled={busy} size="xs" variant="ghost">
              {k.assign}
              <Codicon name="chevron-down" size="0.7rem" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center">
            {(roster?.profiles ?? []).map(profile => (
              <DropdownMenuItem
                key={profile.name}
                onSelect={() => bulk.mutate({ assignee: profile.name, reclaim_first: true })}
              >
                <Avatar name={profile.name} size="0.875rem" />
                {profile.name}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => bulk.mutate({ assignee: '', reclaim_first: true })}>
              {k.unassignAction}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <Button disabled={busy} onClick={() => bulk.mutate({ archive: true })} size="xs" variant="ghost">
          {k.archive}
        </Button>
        <Button
          className="text-destructive"
          disabled={busy}
          onClick={() => bulkDelete.mutate()}
          size="xs"
          variant="ghost"
        >
          {k.delete}
        </Button>

        <Tip label={k.clearSelection}>
          <Button aria-label={k.clearSelection} onClick={onClear} size="icon-xs" variant="ghost">
            <Codicon name="close" size="0.8rem" />
          </Button>
        </Tip>
      </div>
    </div>
  )
}

// ── page ─────────────────────────────────────────────────────────────────────

export function KanbanBoardPage() {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const slug = useValue($boardSlug)
  const [archived, setArchived] = useState(false)

  // Live updates ride the events socket (bindApi); this interval is only the
  // slow heartbeat for socketless paths (OAuth remotes, dropped connections).
  const { data: board, error } = useQuery({
    queryFn: () => fetchBoard(archived),
    queryKey: boardKey(scope, slug, archived),
    refetchInterval: 60_000
  })

  const [openId, setOpenId] = useState<null | string>(null)
  const [addStatus, setAddStatus] = useState<null | string>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [tenant, setTenant] = useState('')
  const [assignee, setAssignee] = useState('')
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  const [prompt, setPrompt] = useState<null | { id: string; kind: PromptKind }>(null)
  const [linksPing, setLinksPing] = useState(0)

  // The facets live in atoms (shared with the filter kebab) but seed from — and
  // write back to — the hash query, so a copied URL reopens the same view.
  // Seeding runs once per mount; writing tracks the atoms (and no-ops when the
  // hash already matches).
  const statusFacets = useValue($filterStatus)
  const priorityFacets = useValue($filterPriority)
  const triageFacet = useValue($filterTriage)

  const facets = useMemo<FilterQuery>(
    () => ({ priority: priorityFacets, status: statusFacets, triage: triageFacet }),
    [priorityFacets, statusFacets, triageFacet]
  )

  useEffect(() => {
    const seed = parseFilterQuery(window.location.hash)

    $filterStatus.set(seed.status)
    $filterPriority.set(seed.priority)
    $filterTriage.set(seed.triage)
  }, [])

  useEffect(() => {
    const next = buildFilterQuery(window.location.hash, facets)

    if (!window.location.hash || next === window.location.hash.replace(/^#/, '')) {
      return
    }

    try {
      window.history.replaceState(
        window.history.state,
        '',
        `${window.location.pathname}${window.location.search}#${next}`
      )
    } catch {
      // Sharing a view is a nicety; a locked-down scheme must not break the board.
    }
  }, [facets])

  // A new-task request raised from outside the page (⌘⌥N, the palette row).
  // The command navigates here and parks the lane; the page picks it up on
  // arrival — whether it was already mounted or is mounting for the first
  // time — then clears it so a later remount can't reopen the dialog.
  const requestedLane = useValue($newTaskLane)

  useEffect(() => {
    if (requestedLane === null) {
      return
    }

    setAddStatus(requestedLane)
    $newTaskLane.set(null)
  }, [requestedLane])

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev)

      if (!next.delete(id)) {
        next.add(id)
      }

      return next
    })
  }

  // Prune ids that left the board (completed elsewhere, deleted, filtered by
  // a board switch) so the bar's count never lies about what a bulk op hits.
  useEffect(() => {
    if (!board) {
      return
    }

    const alive = new Set(board.columns.flatMap(col => col.tasks.map(task => task.id)))

    setSelected(prev => {
      const kept = [...prev].filter(id => alive.has(id))

      return kept.length === prev.size ? prev : new Set(kept)
    })
  }, [board])

  useEffect(() => {
    if (selected.size === 0) {
      return
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSelected(new Set())
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [selected.size])

  const columnNames = board?.columns.map(col => col.name) ?? []

  const parentOptions = useMemo(
    () => board?.columns.flatMap(col => col.tasks).map(task => ({ id: task.id, title: task.title })) ?? [],
    [board]
  )

  // Client-side filters, mirroring the dashboard (search over title/body/id).
  const filtered = useMemo(() => {
    if (!board) {
      return null
    }

    const q = search.trim().toLowerCase()

    const keep = (task: KanbanTask) =>
      (!q || `${task.title} ${task.body ?? ''} ${task.id}`.toLowerCase().includes(q)) &&
      (!tenant || task.tenant === tenant) &&
      (!assignee || task.assignee === assignee) &&
      matchesFacetFilters(task, facets)

    return { ...board, columns: board.columns.map(col => ({ ...col, tasks: col.tasks.filter(keep) })) }
  }, [board, search, tenant, assignee, facets])

  const total = filtered?.columns.reduce((sum, col) => sum + col.tasks.length, 0) ?? 0

  const moveMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => patchTask(id, { status }),
    onMutate: async ({ id, status }) => {
      await qc.cancelQueries({ queryKey: boardKey(scope, slug, archived) })
      const previous = qc.getQueryData<KanbanBoard>(boardKey(scope, slug, archived))

      if (previous) {
        qc.setQueryData(boardKey(scope, slug, archived), moveCard(previous, id, status))
      }

      return { previous }
    },
    onError: (err, _vars, context) => {
      if (context?.previous) {
        qc.setQueryData(boardKey(scope, slug, archived), context.previous)
      }

      host.notify({ kind: 'error', message: errText(err) })
    },
    onSettled: (_data, _err, vars) => {
      void qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
      void qc.invalidateQueries({ queryKey: taskKey(scope, slug, vars.id) })
    }
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteTask(id),
    onMutate: async id => {
      await qc.cancelQueries({ queryKey: boardKey(scope, slug, archived) })
      const previous = qc.getQueryData<KanbanBoard>(boardKey(scope, slug, archived))

      if (previous) {
        qc.setQueryData(boardKey(scope, slug, archived), removeCard(previous, id))
      }

      return { previous }
    },
    onError: (err, _id, context) => {
      if (context?.previous) {
        qc.setQueryData(boardKey(scope, slug, archived), context.previous)
      }

      host.notify({ kind: 'error', message: errText(err) })
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
  })

  const onMove = (id: string, status: string) => {
    const task = board?.columns.flatMap(col => col.tasks).find(candidate => candidate.id === id)

    if (!task || task.status === status) {
      return
    }

    if (isLockedTarget(status)) {
      host.notify({ kind: 'info', message: lockedReason(k, status) })

      return
    }

    moveMut.mutate({ id, status })
  }

  // Card-menu writes fan out per id over the id-scoped endpoints (BulkTaskBody
  // carries no block_reason/note, so block/review still need the per-id route);
  // partial failures are named and
  // successes stand — the refresh then shows the true state.
  const runCardOps = (ops: Array<() => Promise<unknown>>) => {
    void Promise.allSettled(ops.map(op => op())).then(results => {
      const failures = results.flatMap(result => (result.status === 'rejected' ? [result.reason] : []))

      if (failures.length === 1) {
        host.notify({ kind: 'error', message: errText(failures[0]) })
      } else if (failures.length > 1) {
        host.notify({ kind: 'warning', message: k.bulkFailed(failures.length, ops.length, errText(failures[0])) })
      }

      void qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
    })
  }

  const onCardAct = (key: CardActionKey, ids: string[], value?: string) => {
    if (key === 'reassign' && value) {
      runCardOps(ids.map(id => () => reassignTask(id, value)))

      return
    }

    const statusFor: Partial<Record<CardActionKey, string>> = { requestReview: 'review', unblock: 'ready' }
    const status = statusFor[key]

    if (status) {
      runCardOps(ids.map(id => () => patchTask(id, { status })))

      return
    }

    if (key === 'block' || key === 'comment' || key === 'requestChanges') {
      setPrompt({ id: ids[0], kind: key === 'requestChanges' ? 'changes' : key })

      return
    }

    // add link / add child: land on the drawer's dependencies section instead
    // of growing an ID picker inside the menu.
    setOpenId(ids[0])
    setLinksPing(ping => ping + 1)
  }

  const anyFilter =
    Boolean(search || tenant || assignee) ||
    facets.status.length > 0 ||
    facets.priority.length > 0 ||
    facets.triage

  const errorMessage = error ? errText(error) : null

  // Grab-to-scrub the lane strip (shared primitive, same as the dashboard's pan).
  const lanesRef = useRef<HTMLDivElement>(null)
  const { grabbing, onMouseDown } = useGrabScroll(lanesRef)

  // Lane collapse: auto (empty → rail) unless the user overrode it. The map
  // stores only deviations from auto, so it stays tiny and self-heals. On a
  // board with no work at all, auto is disabled — a wall of rails teaches
  // nothing, so a fresh board shows its full structure instead.
  const laneOverrides = useValue($collapsedLanes)
  const boardHasWork = (board?.columns.reduce((sum, col) => sum + col.tasks.length, 0) ?? 0) > 0

  // An override only lives for the lane's current empty/non-empty phase: when
  // emptiness flips (last card dragged out, first card dropped in) the stale
  // override is dropped and auto takes over — so a drained lane collapses even
  // if it was manually expanded ages ago, while expanding an empty lane still
  // sticks for as long as it stays empty.
  //
  // The phase is a string signature held in state, not a ref: React bails out
  // when it's unchanged, so the common case (a poll where no lane's emptiness
  // moved) costs no extra render, and nothing lags a render behind the value
  // it mirrors.
  const lanePhase = filtered
    ? filtered.columns.map(col => `${col.name}:${col.tasks.length === 0 ? 'empty' : 'full'}`).join('|')
    : null

  const [prevLanePhase, setPrevLanePhase] = useState<null | string>(null)

  useEffect(() => {
    if (lanePhase === null || lanePhase === prevLanePhase) {
      return
    }

    setPrevLanePhase(lanePhase)

    if (prevLanePhase === null) {
      return
    }

    const before = new Map(prevLanePhase.split('|').map(entry => entry.split(':') as [string, string]))
    const overrides = { ...$collapsedLanes.get() }
    let changed = false

    for (const entry of lanePhase.split('|')) {
      const [name, phase] = entry.split(':')
      const was = before.get(name)

      if (was !== undefined && was !== phase && name in overrides) {
        delete overrides[name]
        changed = true
      }
    }

    if (changed) {
      $collapsedLanes.set(overrides)
    }
  }, [lanePhase, prevLanePhase])

  const toggleLane = (name: string, auto: boolean) => {
    const overrides = { ...laneOverrides }
    const next = !(overrides[name] ?? auto)

    if (next === auto) {
      delete overrides[name]
    } else {
      overrides[name] = next
    }

    $collapsedLanes.set(overrides)
  }

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-(--ui-surface-background)">
      {/* Page-owned header chrome: exists exactly while this page is mounted. */}
      <Contribute area={WORKSPACE_PAGE_HEADER_AREA} id="kanban:board-switcher">
        <BoardSwitcher />
      </Contribute>

      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-2">
        <h1 className="text-sm font-semibold text-foreground">{k.title}</h1>
        <span className="rounded-full bg-(--ui-bg-quaternary) px-1.5 py-px text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
          {total}
        </span>
        {board && (
          <FilterMenu
            archived={archived}
            assignee={assignee}
            board={board}
            onArchived={setArchived}
            onAssignee={setAssignee}
            onTenant={setTenant}
            tenant={tenant}
          />
        )}
        <SearchField aria-label={k.filterCards} onChange={setSearch} placeholder={k.filterCards} value={search} />
        <div className="ml-auto flex items-center gap-1">
          <Tip label={k.orchestrationSettings}>
            <Button
              aria-label={k.orchestrationSettings}
              className={cn(settingsOpen && 'bg-(--ui-control-active-background) text-foreground')}
              onClick={() => setSettingsOpen(!settingsOpen)}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="organization" size="0.85rem" />
            </Button>
          </Tip>
          <Button onClick={() => setAddStatus('triage')} size="sm">
            <Codicon name="add" size="0.8rem" />
            {k.newTask}
          </Button>
        </div>
      </header>

      {settingsOpen && <OrchestrationPanel />}

      {board && <Intro />}

      {errorMessage && !board ? (
        <div className="grid flex-1 place-items-center">
          <ErrorState title={errorMessage} />
        </div>
      ) : !filtered ? (
        <div className="grid flex-1 place-items-center">
          <Loader type="lemniscate-bloom" />
        </div>
      ) : total === 0 ? (
        <div className="grid flex-1 place-items-center px-4 text-center">
          <div className="flex flex-col items-center gap-2">
            <Codicon className="text-(--ui-text-quaternary)" name="project" size="1.25rem" />
            <p className="text-xs text-(--ui-text-tertiary)">{anyFilter ? k.noMatch : k.noTasks}</p>
            <Button className="mt-0.5" onClick={() => setAddStatus('triage')} size="sm" variant="outline">
              <Codicon name="add" size="0.75rem" />
              {k.newTask}
            </Button>
          </div>
        </div>
      ) : (
        <div
          className={cn('flex flex-1 gap-2 overflow-x-auto px-4 pt-1 pb-3', grabbing && 'cursor-grabbing')}
          onMouseDown={onMouseDown}
          ref={lanesRef}
        >
          {filtered.columns.map(col => {
            const auto = boardHasWork && col.tasks.length === 0

            return (
              <Column
                assignees={board?.assignees ?? []}
                collapsed={laneOverrides[col.name] ?? auto}
                column={col}
                columns={columnNames}
                key={col.name}
                onAct={onCardAct}
                onAdd={setAddStatus}
                onDelete={id => deleteMut.mutate(id)}
                onDropTask={onMove}
                onMove={onMove}
                onOpen={setOpenId}
                onToggle={() => toggleLane(col.name, auto)}
                onToggleSelect={toggleSelect}
                selected={selected}
              />
            )
          })}
        </div>
      )}

      {selected.size > 0 && (
        <SelectionBar
          columns={columnNames}
          onClear={() => setSelected(new Set())}
          onDone={failed => setSelected(new Set(failed))}
          selected={selected}
        />
      )}

      <NewTaskDialog onClose={() => setAddStatus(null)} parents={parentOptions} target={addStatus} />
      {prompt && <CardPromptDialog entity={prompt} onClose={() => setPrompt(null)} />}
      <TaskDrawer
        columns={columnNames}
        focusLinks={linksPing}
        id={openId}
        onClose={() => setOpenId(null)}
        onOpen={setOpenId}
      />
    </div>
  )
}
