/**
 * The Kanban board page — mounted at `/kanban` (a ROUTES_AREA contribution) in
 * the workspace pane or a split route tile. The desktop port of the dashboard
 * board: one compact header row (count, board switcher, filter kebab, search,
 * settings, new task — on the full page the switcher is projected into the
 * page header instead, see WorkspacePageHeaderControl), columns in
 * BOARD_COLUMNS order, drag-to-move (optimistic, workflow-checked),
 * primary-modifier-click multi-select with a floating bulk bar, right-click
 * actions, and the detail drawer. Dispatch nudges ride every write (see api.ts).
 */

import {
  Button,
  cn,
  Codicon,
  compactNumber,
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
  WorkspacePageHeaderControl
} from '@hermes/plugin-sdk'
import { type DragEvent as ReactDragEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import {
  $boardSlug,
  $collapsedLanes,
  $introDismissed,
  $lanesByProfile,
  $swimlaneBy,
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
  fetchProjects,
  patchTask,
  profilesKey,
  projectsKey,
  taskKey,
  useKanbanScope
} from './api'
import { BoardSwitcher } from './board-switcher'
import { Card } from './card'
import { TaskDrawer } from './drawer'
import { EMPTY_OVERRIDE, ModelOverrideField, overrideCreateFields, type TaskModelOverride } from './model-override'
import { OrchestrationPanel } from './orchestration'
import { laneTitle, SwimlaneGrid } from './swimlane-grid'
import {
  applyPatch,
  dropPatch,
  groupSwimlanes,
  type LanePreset,
  lanePreset,
  normalizeSwimlaneBy,
  SWIMLANE_OPTIONS,
  type TaskPatch
} from './swimlanes'
import { columnMeta, type KanbanBoard, type KanbanTask, type TaskEstimate } from './types'
import {
  $newTaskLane,
  Avatar,
  columnHelp,
  columnLabel,
  errText,
  FIELD_LABEL,
  isLockedTarget,
  lockedReason,
  useKanban,
  useOrchestration
} from './ui'

// ── optimistic board edits (reconciled by the follow-up refresh) ─────────────

function removeCard(board: KanbanBoard, id: string): KanbanBoard {
  return { ...board, columns: board.columns.map(col => ({ ...col, tasks: col.tasks.filter(t => t.id !== id) })) }
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

/** A new-task request: the column it lands in, plus — when raised from a
 *  swimlane cell — the lane's create-time fields and display name. */
interface NewTaskTarget {
  status: string
  preset?: LanePreset
  laneTitle?: string
}

function NewTaskDialog({
  onClose,
  parents,
  request
}: {
  onClose: () => void
  parents: Array<{ id: string; title: string }>
  request: NewTaskTarget | null
}) {
  const target = request?.status ?? null
  const preset = request?.preset
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
    if (request) {
      setTitle('')
      setBodyText('')
      // A lane preset seeds the form; the default assignee already IS the
      // "default" option, so it maps to that rather than a duplicate row.
      setAssignee(preset?.assignee && preset.assignee !== resolvedDefault ? preset.assignee : '')
      setPriority(String(preset?.priority ?? 0))
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
  }, [request, preset, resolvedDefault, boardDefaultKind])

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
        // Lane identity the form doesn't edit (project, tenant) rides along.
        project_id: preset?.project_id,
        tenant: preset?.tenant,
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
          <DialogTitle>
            {target
              ? k.newTaskIn(
                  request?.laneTitle ? `${columnLabel(k, target)} · ${request.laneTitle}` : columnLabel(k, target)
                )
              : k.newTask}
          </DialogTitle>
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
                {[
                  ...(roster?.profiles ?? []).map(profile => profile.name),
                  // A lane's assignee that isn't a local profile still has to be selectable.
                  ...(assignee && assignee !== PARKED && !roster?.profiles.some(p => p.name === assignee)
                    ? [assignee]
                    : [])
                ]
                  .filter(name => name !== resolvedDefault)
                  .map(name => (
                    <SelectItem key={name} value={name}>
                      {name}
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
  const active = Boolean(assignee || tenant || archived)
  const lanesByProfile = useValue($lanesByProfile)
  // Swimlane cells don't sub-group, so the toggle would do nothing there.
  const swimlanes = normalizeSwimlaneBy(useValue($swimlaneBy)) !== 'none'

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
        {!swimlanes && (
          <DropdownMenuItem onSelect={() => $lanesByProfile.set(!lanesByProfile)}>
            {k.groupRunning}
            {check(lanesByProfile)}
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// ── swimlane picker ──────────────────────────────────────────────────────────

function SwimlaneMenu() {
  const k = useKanban()
  const by = normalizeSwimlaneBy(useValue($swimlaneBy))

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={k.swimlanes}
          className={cn(by !== 'none' && 'bg-(--ui-control-active-background) text-foreground')}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="layers" size="0.85rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {SWIMLANE_OPTIONS.map(option => (
          <DropdownMenuItem key={option} onSelect={() => $swimlaneBy.set(option)}>
            {k.swimlaneBy[option]}
            {by === option && <Codicon className="ml-auto" name="check" size="0.8rem" />}
          </DropdownMenuItem>
        ))}
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
  const [adding, setAdding] = useState<NewTaskTarget | null>(null)
  const setAddStatus = (status: string) => setAdding({ status })
  const swimlaneBy = normalizeSwimlaneBy(useValue($swimlaneBy))
  const laneBy = swimlaneBy === 'none' ? null : swimlaneBy
  // Names for project lanes; the board switcher reads the same cached query.
  const { data: projectList } = useQuery({ queryKey: projectsKey(scope), queryFn: fetchProjects, staleTime: 30_000 })

  const projectNames = useMemo(
    () => new Map((projectList?.projects ?? []).map(project => [project.id, project.name])),
    [projectList]
  )

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [tenant, setTenant] = useState('')
  const [assignee, setAssignee] = useState('')
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())

  // A new-task request raised from outside the page (⌘⌥N, the palette row).
  // The command navigates here and parks the lane; the page picks it up on
  // arrival — whether it was already mounted or is mounting for the first
  // time — then clears it so a later remount can't reopen the dialog.
  const requestedLane = useValue($newTaskLane)

  useEffect(() => {
    if (requestedLane === null) {
      return
    }

    setAdding({ status: requestedLane })
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
      (!assignee || task.assignee === assignee)

    return { ...board, columns: board.columns.map(col => ({ ...col, tasks: col.tasks.filter(keep) })) }
  }, [board, search, tenant, assignee])

  const total = filtered?.columns.reduce((sum, col) => sum + col.tasks.length, 0) ?? 0

  const swimlanes = useMemo(
    () =>
      laneBy && filtered
        ? groupSwimlanes(filtered.columns, laneBy, key => laneTitle(laneBy, key, k, projectNames))
        : null,
    [filtered, laneBy, k, projectNames]
  )

  const moveMut = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: TaskPatch }) => patchTask(id, { ...patch }),
    onMutate: async ({ id, patch }) => {
      await qc.cancelQueries({ queryKey: boardKey(scope, slug, archived) })
      const previous = qc.getQueryData<KanbanBoard>(boardKey(scope, slug, archived))

      if (previous) {
        qc.setQueryData(boardKey(scope, slug, archived), applyPatch(previous, id, patch))
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

  const findTask = (id: string) => board?.columns.flatMap(col => col.tasks).find(candidate => candidate.id === id)

  // One drop rule for the flat board (lane = null: status only) and the
  // swimlane grid (a cross-lane drop also writes the lane's field).
  const dropCard = (id: string, lane: null | string, status: string) => {
    const task = findTask(id)
    const patch = task ? dropPatch(swimlaneBy, task, lane, status) : null

    if (!patch || Object.keys(patch).length === 0) {
      return
    }

    if (patch.status && isLockedTarget(patch.status)) {
      host.notify({ kind: 'info', message: lockedReason(k, patch.status) })

      return
    }

    moveMut.mutate({ id, patch })
  }

  const canDropCard = (id: string, lane: string, status: string) => {
    const task = findTask(id)
    const patch = task ? dropPatch(swimlaneBy, task, lane, status) : null

    return patch !== null && !(patch.status && isLockedTarget(patch.status))
  }

  const onMove = (id: string, status: string) => dropCard(id, null, status)

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

  const autoCollapsed = (col: { tasks: KanbanTask[] }) => boardHasWork && col.tasks.length === 0

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
      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-2">
        <h1 className="text-sm font-semibold text-foreground">{k.title}</h1>
        <span className="rounded-full bg-(--ui-bg-quaternary) px-1.5 py-px text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
          {total}
        </span>
        {/* The full page projects this into its page header; a split tile has
            none, so the switcher stays here in the row. */}
        <WorkspacePageHeaderControl id="kanban:board-switcher">
          <BoardSwitcher />
        </WorkspacePageHeaderControl>
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
        <SwimlaneMenu />
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
            <p className="text-xs text-(--ui-text-tertiary)">{search || tenant || assignee ? k.noMatch : k.noTasks}</p>
            <Button className="mt-0.5" onClick={() => setAddStatus('triage')} size="sm" variant="outline">
              <Codicon name="add" size="0.75rem" />
              {k.newTask}
            </Button>
          </div>
        </div>
      ) : laneBy && swimlanes ? (
        <div
          className={cn('min-h-0 flex-1 overflow-auto px-4 pb-3', grabbing && 'cursor-grabbing')}
          onMouseDown={onMouseDown}
          ref={lanesRef}
        >
          <SwimlaneGrid
            by={laneBy}
            canDrop={canDropCard}
            collapsedColumn={col => laneOverrides[col.name] ?? autoCollapsed(col)}
            columns={filtered.columns}
            lanes={swimlanes}
            onAdd={(status, lane) =>
              setAdding({
                status,
                preset: lanePreset(laneBy, lane),
                laneTitle: laneTitle(laneBy, lane, k, projectNames)
              })
            }
            onDelete={id => deleteMut.mutate(id)}
            onDrop={dropCard}
            onMove={onMove}
            onOpen={setOpenId}
            onToggleColumn={col => toggleLane(col.name, autoCollapsed(col))}
            onToggleSelect={toggleSelect}
            projectNames={projectNames}
            selected={selected}
          />
        </div>
      ) : (
        <div
          className={cn('flex flex-1 gap-2 overflow-x-auto px-4 pt-1 pb-3', grabbing && 'cursor-grabbing')}
          onMouseDown={onMouseDown}
          ref={lanesRef}
        >
          {filtered.columns.map(col => {
            const auto = autoCollapsed(col)

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

      <NewTaskDialog onClose={() => setAdding(null)} parents={parentOptions} request={adding} />
      <TaskDrawer columns={columnNames} id={openId} onClose={() => setOpenId(null)} onOpen={setOpenId} />
    </div>
  )
}
