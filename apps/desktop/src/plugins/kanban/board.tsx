/**
 * The Kanban board page — mounted at `/kanban` (a ROUTES_AREA contribution) in
 * the workspace pane. The desktop port of the dashboard board: one compact
 * header row (count, filter kebab, search, settings, new task — the board
 * SWITCHER lives in the titlebar, see board-switcher.tsx), columns in
 * BOARD_COLUMNS order, drag-to-move (optimistic, workflow-checked),
 * ⌘-click multi-select with a floating bulk bar, right-click actions, and
 * the detail drawer. Dispatch nudges ride every write (see api.ts).
 */

import {
  Button,
  cn,
  Codicon,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
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
  host,
  Input,
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
  TITLEBAR_AREAS,
  useGrabScroll,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
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
  boardKey,
  bulkTasks,
  createTask,
  deleteTask,
  fetchBoard,
  fetchProfiles,
  patchTask,
  PROFILES_KEY
} from './api'
import { BoardSwitcher } from './board-switcher'
import { TaskDrawer } from './drawer'
import { OrchestrationPanel } from './orchestration'
import { columnMeta, type KanbanBoard, type KanbanTask } from './types'
import {
  ago,
  ARC_TITLES,
  type ArcState,
  arcState,
  Avatar,
  errText,
  FIELD_LABEL,
  isLockedTarget,
  LOCKED_COLUMNS,
  RunClock,
  shortId,
  useDefaultAssignee,
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
              ? 'A review agent is checking the completed work.'
              : task.assignee
                ? `${attached} is attached — the dispatcher hands this over on its next tick (≤1m).`
                : task.status === 'triage'
                  ? `${attached} (the orchestrator) picks this up on the next tick and writes the spec.`
                  : `Auto-assigns to “${attached}” (kanban.default_assignee) on the next dispatch tick.`
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
        <Tip label={ARC_TITLES.running}>
          <span className="shrink-0 cursor-help">
            <RunClock task={task} />
          </span>
        </Tip>
      )}
      {arc === 'stale' && (
        <Tip label={ARC_TITLES.stale}>
          <span className="shrink-0 cursor-help font-medium text-amber-500">no heartbeat</span>
        </Tip>
      )}
      {unassignedReady && !fallback && (
        <Tip label="Ready cards only run once a profile is assigned. Open the card and set an assignee, or configure a default assignee in orchestration settings.">
          <span className="inline-flex shrink-0 cursor-help items-center gap-1 text-amber-500">
            <Codicon name="debug-disconnect" size="0.7rem" />
            won't run
          </span>
        </Tip>
      )}
      <div className="ml-auto flex min-w-0 shrink items-center gap-2">
        {typeof task.priority === 'number' && task.priority > 0 && (
          <span className="inline-flex items-center gap-0.5 text-amber-500">
            <Codicon name="arrow-up" size="0.7rem" />
            {task.priority}
          </span>
        )}
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
          Open
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => onToggleSelect(task.id)}>
          <Codicon name={selected ? 'close' : 'check-all'} size="0.85rem" />
          {selected ? 'Deselect' : 'Select (⌘-click)'}
        </ContextMenuItem>
        <ContextMenuSeparator />
        {columns
          .filter(name => name !== task.status && !isLockedTarget(name))
          .map(name => (
            <ContextMenuItem key={name} onSelect={() => onMove(task.id, name)}>
              <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
              Move to {columnMeta(name).label}
            </ContextMenuItem>
          ))}
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => onDelete(task.id)} variant="destructive">
          <Codicon name="trash" size="0.85rem" />
          Delete
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

// ── column ───────────────────────────────────────────────────────────────────

function Column({
  collapsed,
  column,
  columns,
  onAdd,
  onDelete,
  onDropTask,
  onMove,
  onOpen,
  onToggle,
  onToggleSelect,
  selected
}: {
  collapsed: boolean
  column: { name: string; tasks: KanbanTask[] }
  columns: string[]
  onAdd: (status: string) => void
  onDelete: (id: string) => void
  onDropTask: (id: string, status: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggle: () => void
  onToggleSelect: (id: string) => void
  selected: ReadonlySet<string>
}) {
  const [over, setOver] = useState(false)
  const meta = columnMeta(column.name)
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
        aria-label={`Expand ${meta.label}`}
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
          {meta.label}
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
        <Tip label={meta.help}>
          <span className="cursor-help text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
            {meta.label}
          </span>
        </Tip>
        <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
        <button
          aria-label={`Collapse ${meta.label}`}
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
                    columns={columns}
                    key={task.id}
                    onDelete={onDelete}
                    onMove={onMove}
                    onOpen={onOpen}
                    onToggleSelect={onToggleSelect}
                    selected={selected.has(task.id)}
                    task={task}
                  />
                ))}
              </div>
            ))
          : column.tasks.map(task => (
              <Card
                columns={columns}
                key={task.id}
                onDelete={onDelete}
                onMove={onMove}
                onOpen={onOpen}
                onToggleSelect={onToggleSelect}
                selected={selected.has(task.id)}
                task={task}
              />
            ))}
        {/* Jira-style lane add — dashed, faded in on lane hover. Opacity (not
            display) so it always holds its slot and never thrashes layout.
            Locked lanes get none: you can't create into a system state. */}
        {!locked && (
          <button
            aria-label={`New task in ${meta.label}`}
            className="flex shrink-0 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary) py-1.5 text-(--ui-text-tertiary) opacity-0 transition-[opacity,color,border-color] group-hover/col:opacity-100 hover:border-(--ui-text-quaternary) hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100"
            onClick={() => onAdd(column.name)}
            type="button"
          >
            <Codicon name="add" size="0.8rem" />
          </button>
        )}
        {column.tasks.length === 0 && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-[0.6875rem] text-(--ui-text-quaternary)">
            Empty
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
  const qc = useQueryClient()
  const { data: roster } = useQuery({ queryKey: PROFILES_KEY, queryFn: fetchProfiles, staleTime: 60_000 })
  // Title-only creates must RUN: "auto" resolves to the orchestration default
  // (ultimately the active profile), applied at create time. Never silently
  // unassigned — parking a card is the explicit choice, not the default.
  const resolvedDefault = useOrchestration()?.resolved_default_assignee || 'default'
  const isTriage = target === 'triage'
  const [title, setTitle] = useState('')
  const [bodyText, setBodyText] = useState('')
  const [assignee, setAssignee] = useState('')
  const [priority, setPriority] = useState('0')
  const [skills, setSkills] = useState('')
  const [workspaceKind, setWorkspaceKind] = useState<string>('scratch')
  const [parent, setParent] = useState('')
  const [goalMode, setGoalMode] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // Reset per open — the dialog is externally controlled (open = target set),
  // so onOpenChange(true) never fires; key the reset off `target` instead.
  useEffect(() => {
    if (target) {
      setTitle('')
      setBodyText('')
      setAssignee('')
      setPriority('0')
      setSkills('')
      setWorkspaceKind('scratch')
      setParent('')
      setGoalMode(false)
      setError(null)
      setBusy(false)
    }
  }, [target])

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
        workspace_kind: workspaceKind
      })

      if (task && task.status !== target) {
        await patchTask(task.id, { status: target })
      }

      // Dispatcher-presence warning ("this ready task will sit idle") — not an
      // error, but the user should know.
      if (warning) {
        host.notify({ kind: 'warning', message: warning })
      }

      await qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
      onClose()
    } catch (err) {
      setError(errText(err))
      setBusy(false)
    }
  }

  return (
    <Dialog onOpenChange={open => !open && onClose()} open={Boolean(target)}>
      <DialogContent className="w-[min(42rem,94vw)] max-w-none">
        <DialogHeader>
          <DialogTitle>New task{target ? ` in ${columnMeta(target).label}` : ''}</DialogTitle>
        </DialogHeader>
        <div className="flex max-h-[min(72vh,44rem)] flex-col gap-3 overflow-y-auto pr-0.5">
          <Input
            autoFocus
            onChange={event => setTitle(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                void submit()
              }
            }}
            placeholder={isTriage ? 'Rough idea — a specifier will flesh it out' : 'Title'}
            value={title}
          />
          <Textarea
            className="min-h-20"
            onChange={event => setBodyText(event.target.value)}
            placeholder="Description (optional)"
            value={bodyText}
          />

          <div className="grid grid-cols-2 gap-3">
            <Field label="Priority">
              <Input onChange={event => setPriority(event.target.value)} type="number" value={priority} />
            </Field>
            <Field label="Workspace">
              <Select onValueChange={setWorkspaceKind} value={workspaceKind}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {WORKSPACE_KINDS.map(kind => (
                    <SelectItem key={kind} value={kind}>
                      {kind}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          <Field label="Assignee">
            <Select onValueChange={v => setAssignee(v === NO_PARENT ? '' : v)} value={assignee || NO_PARENT}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={NO_PARENT}>{resolvedDefault} (default)</SelectItem>
                {(roster?.profiles ?? [])
                  .filter(profile => profile.name !== resolvedDefault)
                  .map(profile => (
                    <SelectItem key={profile.name} value={profile.name}>
                      {profile.name}
                    </SelectItem>
                  ))}
                <SelectItem value={PARKED}>unassigned (parked — won't run)</SelectItem>
              </SelectContent>
            </Select>
          </Field>

          <Field label="Skills (comma-separated)">
            <Input onChange={event => setSkills(event.target.value)} placeholder="translation, github" value={skills} />
          </Field>

          {parents.length > 0 && (
            <Field label="Parent (blocks until it's done)">
              <Select onValueChange={v => setParent(v === NO_PARENT ? '' : v)} value={parent || NO_PARENT}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_PARENT}>— no parent —</SelectItem>
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
            <Switch aria-label="Goal mode" checked={goalMode} onCheckedChange={setGoalMode} size="xs" />
            Goal mode (worker loops until a judge agrees it's done)
          </label>

          {error && <span className="text-[0.75rem] text-destructive">{error}</span>}
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="text">
            Cancel
          </Button>
          <Button disabled={!title.trim() || busy} onClick={() => void submit()}>
            {busy ? 'Creating…' : 'Create task'}
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
  const dismissed = useValue($introDismissed)

  if (dismissed) {
    return null
  }

  return (
    <div
      className="mx-4 mb-2 flex flex-col items-start gap-1.5 rounded-lg bg-(--ui-bg-quinary) px-3 py-2.5 text-[0.75rem] leading-relaxed text-(--ui-text-secondary)"
      data-selectable-text="true"
    >
      <p className="min-w-0">
        You don't run the cards — agents do. Put a card in <b>Ready</b> with an assignee and an agent picks it up within
        a minute. No assignee, no run. <b>Triage</b>: an agent rewrites the idea into a proper task first. <b>Todo</b>:
        waiting on other cards. <b>Scheduled</b>: waiting on a timer. <b>Running</b> and <b>Review</b>: the agents'
        lanes, hands off. <b>Blocked</b>: it's waiting on you. Results come back on the card.
      </p>
      <Button onClick={() => $introDismissed.set(true)} size="inline" variant="textStrong">
        Got it
      </Button>
    </div>
  )
}

const UNASSIGNED_LANE = 'unassigned'

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
  const active = Boolean(assignee || tenant || archived)
  const lanesByProfile = useValue($lanesByProfile)

  const check = (on: boolean) => (on ? <Codicon className="ml-auto" name="check" size="0.8rem" /> : null)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label="Filters"
          className={cn(active && 'bg-(--ui-control-active-background) text-foreground')}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="filter" size="0.85rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem onSelect={() => onAssignee('')}>All profiles{check(!assignee)}</DropdownMenuItem>
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
            <DropdownMenuItem onSelect={() => onTenant('')}>All tenants{check(!tenant)}</DropdownMenuItem>
            {board.tenants.map(name => (
              <DropdownMenuItem key={name} onSelect={() => onTenant(name)}>
                {name}
                {check(tenant === name)}
              </DropdownMenuItem>
            ))}
          </>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => onArchived(!archived)}>Show archived{check(archived)}</DropdownMenuItem>
        <DropdownMenuItem onSelect={() => $lanesByProfile.set(!lanesByProfile)}>
          Group Running by profile{check(lanesByProfile)}
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
  const qc = useQueryClient()
  const { data: roster } = useQuery({ queryKey: PROFILES_KEY, queryFn: fetchProfiles, staleTime: 60_000 })

  const finish = (failed: Array<{ error?: string; id: string }>) => {
    void qc.invalidateQueries({ queryKey: ['kanban', 'board'] })

    if (failed.length > 0) {
      host.notify({
        kind: 'warning',
        message: `${failed.length} of ${selected.size} failed — ${failed[0].error ?? 'refused'}. Failed cards stay selected.`
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
        <span className="mr-1 text-xs tabular-nums text-(--ui-text-secondary)">{selected.size} selected</span>

        <DropdownMenu onOpenChange={open => setMenu(open ? 'move' : null)} open={menu === 'move'}>
          <DropdownMenuTrigger asChild>
            <Button disabled={busy} size="xs" variant="ghost">
              Move to
              <Codicon name="chevron-down" size="0.7rem" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center">
            {columns
              .filter(name => !isLockedTarget(name))
              .map(name => (
                <DropdownMenuItem key={name} onSelect={() => bulk.mutate({ status: name })}>
                  <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
                  {columnMeta(name).label}
                </DropdownMenuItem>
              ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu onOpenChange={open => setMenu(open ? 'assign' : null)} open={menu === 'assign'}>
          <DropdownMenuTrigger asChild>
            <Button disabled={busy} size="xs" variant="ghost">
              Assign
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
              Unassign
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <Button disabled={busy} onClick={() => bulk.mutate({ archive: true })} size="xs" variant="ghost">
          Archive
        </Button>
        <Button
          className="text-destructive"
          disabled={busy}
          onClick={() => bulkDelete.mutate()}
          size="xs"
          variant="ghost"
        >
          Delete
        </Button>

        <Tip label="Clear selection (Esc)">
          <Button aria-label="Clear selection" onClick={onClear} size="icon-xs" variant="ghost">
            <Codicon name="close" size="0.8rem" />
          </Button>
        </Tip>
      </div>
    </div>
  )
}

// ── page ─────────────────────────────────────────────────────────────────────

export function KanbanBoardPage() {
  const qc = useQueryClient()
  const slug = useValue($boardSlug)
  const [archived, setArchived] = useState(false)

  // Live updates ride the events socket (bindApi); this interval is only the
  // slow heartbeat for socketless paths (OAuth remotes, dropped connections).
  const { data: board, error } = useQuery({
    queryFn: () => fetchBoard(archived),
    queryKey: boardKey(slug, archived),
    refetchInterval: 60_000
  })

  const [openId, setOpenId] = useState<null | string>(null)
  const [addStatus, setAddStatus] = useState<null | string>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [tenant, setTenant] = useState('')
  const [assignee, setAssignee] = useState('')
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())

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
      (!assignee || task.assignee === assignee)

    return { ...board, columns: board.columns.map(col => ({ ...col, tasks: col.tasks.filter(keep) })) }
  }, [board, search, tenant, assignee])

  const total = filtered?.columns.reduce((sum, col) => sum + col.tasks.length, 0) ?? 0

  const moveMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => patchTask(id, { status }),
    onMutate: async ({ id, status }) => {
      await qc.cancelQueries({ queryKey: boardKey(slug, archived) })
      const previous = qc.getQueryData<KanbanBoard>(boardKey(slug, archived))

      if (previous) {
        qc.setQueryData(boardKey(slug, archived), moveCard(previous, id, status))
      }

      return { previous }
    },
    onError: (err, _vars, context) => {
      if (context?.previous) {
        qc.setQueryData(boardKey(slug, archived), context.previous)
      }

      host.notify({ kind: 'error', message: errText(err) })
    },
    onSettled: (_data, _err, vars) => {
      void qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
      void qc.invalidateQueries({ queryKey: ['kanban', 'task', slug, vars.id] })
    }
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteTask(id),
    onMutate: async id => {
      await qc.cancelQueries({ queryKey: boardKey(slug, archived) })
      const previous = qc.getQueryData<KanbanBoard>(boardKey(slug, archived))

      if (previous) {
        qc.setQueryData(boardKey(slug, archived), removeCard(previous, id))
      }

      return { previous }
    },
    onError: (err, _id, context) => {
      if (context?.previous) {
        qc.setQueryData(boardKey(slug, archived), context.previous)
      }

      host.notify({ kind: 'error', message: errText(err) })
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: ['kanban', 'board'] })
  })

  const onMove = (id: string, status: string) => {
    const task = board?.columns.flatMap(col => col.tasks).find(candidate => candidate.id === id)

    if (!task || task.status === status) {
      return
    }

    if (isLockedTarget(status)) {
      host.notify({ kind: 'info', message: LOCKED_COLUMNS[status] })

      return
    }

    moveMut.mutate({ id, status })
  }

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
  const laneCounts = useRef<null | Record<string, number>>(null)

  useEffect(() => {
    if (!filtered) {
      return
    }

    const counts = Object.fromEntries(filtered.columns.map(col => [col.name, col.tasks.length]))
    const prev = laneCounts.current
    laneCounts.current = counts

    if (!prev) {
      return
    }

    const overrides = { ...$collapsedLanes.get() }
    let changed = false

    for (const [name, count] of Object.entries(counts)) {
      const before = prev[name]

      if (before !== undefined && (before === 0) !== (count === 0) && name in overrides) {
        delete overrides[name]
        changed = true
      }
    }

    if (changed) {
      $collapsedLanes.set(overrides)
    }
  }, [filtered])

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
      {/* Page-owned titlebar chrome: exists exactly while this page is mounted. */}
      <Contribute area={TITLEBAR_AREAS.center} id="kanban:board-switcher">
        <BoardSwitcher />
      </Contribute>

      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-2">
        <h1 className="text-sm font-semibold text-foreground">Kanban</h1>
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
        <SearchField aria-label="Filter cards" onChange={setSearch} placeholder="Filter cards…" value={search} />
        <div className="ml-auto flex items-center gap-1">
          <Tip label="Orchestration settings">
            <Button
              aria-label="Orchestration settings"
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
            New task
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
            <p className="text-xs text-(--ui-text-tertiary)">
              {search || tenant || assignee ? 'No tasks match the filters' : 'No tasks on this board'}
            </p>
            <Button className="mt-0.5" onClick={() => setAddStatus('triage')} size="sm" variant="outline">
              <Codicon name="add" size="0.75rem" />
              New task
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
                collapsed={laneOverrides[col.name] ?? auto}
                column={col}
                columns={columnNames}
                key={col.name}
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
      <TaskDrawer columns={columnNames} id={openId} onClose={() => setOpenId(null)} onOpen={setOpenId} />
    </div>
  )
}
