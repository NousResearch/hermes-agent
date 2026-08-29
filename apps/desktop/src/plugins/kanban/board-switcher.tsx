/**
 * Titlebar board switcher — the board page projects this into `titleBar.center`
 * (where chat shows the session-title dropdown) via `<Contribute>`, so it
 * exists exactly while the page is mounted — no route sniffing. Same chrome as
 * the session title: quiet label + chevron, menu on click.
 */

import {
  Button,
  Codicon,
  ConfirmDialog,
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
  host,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  useI18n,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useState } from 'react'

import {
  $boardSlug,
  BOARDS_KEY,
  createBoard,
  deleteBoard,
  fetchBoards,
  fetchProjects,
  pluginOs,
  PROJECTS_KEY,
  updateBoard
} from './api'
import { runExportBoardFlow, runImportBoardFlow } from './transfer'
import type { BoardMeta } from './types'
import { errText, FIELD_LABEL, useKanban } from './ui'

const NO_PROJECT = '__none__'
/** Mirrors `kanban_db.DEFAULT_BOARD` — the board that always exists. */
const DEFAULT_BOARD = 'default'

/** Board scope = a first-class Hermes project. Its primary repo becomes the
 *  board's default workspace root; new tasks inherit it as a worktree with a
 *  deterministic branch. "No project" falls back to scratch sandboxes. */
function ProjectPicker({ onChange, value }: { onChange: (id: string) => void; value: string }) {
  const k = useKanban()
  const { data } = useQuery({ queryKey: PROJECTS_KEY, queryFn: fetchProjects, staleTime: 30_000 })
  const projects = data?.projects ?? []

  return (
    <label className="flex flex-col gap-1">
      <span className={FIELD_LABEL}>{k.project}</span>
      <Select onValueChange={id => onChange(id === NO_PROJECT ? '' : id)} value={value || NO_PROJECT}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NO_PROJECT}>{k.noProject}</SelectItem>
          {projects.map(project => (
            <SelectItem key={project.id} value={project.id}>
              {project.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-[0.6875rem] leading-relaxed text-(--ui-text-quaternary)">
        {k.projectHintPre}
        <span className="font-mono">{k.projectHintCmd}</span>.
      </span>
    </label>
  )
}

/** Every board write ends the same way: refresh the switcher's list and let
 *  the caller finish, or surface the error and leave the dialog open. */
function useBoardWrite<T>(mutationFn: () => Promise<T>, onDone: (result: T) => void) {
  const qc = useQueryClient()

  return useMutation({
    mutationFn,
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: result => {
      void qc.invalidateQueries({ queryKey: BOARDS_KEY })
      onDone(result)
    }
  })
}

/** Shared chrome for the board dialogs — same width, same Cancel/confirm pair. */
function BoardDialog({
  children,
  confirmLabel,
  disabled,
  onClose,
  onConfirm,
  open,
  title
}: {
  children: ReactNode
  confirmLabel: string
  disabled: boolean
  onClose: () => void
  onConfirm: () => void
  open: boolean
  title: string
}) {
  const k = useKanban()

  return (
    <Dialog onOpenChange={o => !o && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3">{children}</div>
        <DialogFooter>
          <Button onClick={onClose} variant="text">
            {k.cancel}
          </Button>
          <Button disabled={disabled} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Display name, with the slug it maps to shown underneath. */
function BoardNameField({
  onChange,
  onEnter,
  slug,
  value
}: {
  onChange: (name: string) => void
  onEnter: () => void
  slug: string
  value: string
}) {
  const k = useKanban()

  return (
    <label className="flex flex-col gap-1">
      <span className={FIELD_LABEL}>{k.name}</span>
      <Input
        autoFocus
        onChange={event => onChange(event.target.value)}
        onKeyDown={event => event.key === 'Enter' && onEnter()}
        placeholder={k.boardNamePlaceholder}
        value={value}
      />
      {slug && <span className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.slug(slug)}</span>}
    </label>
  )
}

function NewBoardDialog({ onClose, open }: { onClose: () => void; open: boolean }) {
  const k = useKanban()
  const [name, setName] = useState('')
  const [project, setProject] = useState('')

  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  useEffect(() => {
    if (open) {
      setName('')
      setProject('')
    }
  }, [open])

  const create = useBoardWrite(
    () => createBoard(slug, name.trim(), project || undefined),
    result => {
      $boardSlug.set(result.board.slug)
      onClose()
    }
  )

  return (
    <BoardDialog
      confirmLabel={k.createBoard}
      disabled={!slug || create.isPending}
      onClose={onClose}
      onConfirm={() => create.mutate()}
      open={open}
      title={k.newBoard}
    >
      {/* Enter submits only while the scope is untouched — once a project is
          picked the choice is worth a deliberate click. */}
      <BoardNameField onChange={setName} onEnter={() => slug && !project && create.mutate()} slug={slug} value={name} />
      <ProjectPicker onChange={setProject} value={project} />
    </BoardDialog>
  )
}

/** Name-only edit, matching how projects rename (a dedicated dialog, separate
 *  from the settings surface that owns scope). The slug is immutable — it is
 *  the board's directory name — so this touches the display name alone. */
function RenameBoardDialog({ board, onClose }: { board: BoardMeta | null; onClose: () => void }) {
  const k = useKanban()
  const [name, setName] = useState('')
  // The dialog stays mounted while closed, so `board` is null most of the
  // time. Resolve the slug here rather than inside the mutation: the React
  // Compiler lifts a callback's property reads into its render-time
  // dependency check, which would deref that null on every closed render.
  const slug = board?.slug ?? ''

  useEffect(() => {
    if (board) {
      setName(board.name || board.slug)
    }
  }, [board])

  const save = useBoardWrite(() => updateBoard(slug, { name: name.trim() }), onClose)
  const disabled = !name.trim() || save.isPending

  return (
    <BoardDialog
      confirmLabel={k.save}
      disabled={disabled}
      onClose={onClose}
      onConfirm={() => save.mutate()}
      open={Boolean(board)}
      title={k.renameBoardTitle}
    >
      <BoardNameField
        onChange={setName}
        onEnter={() => !disabled && save.mutate()}
        slug={slug}
        value={name}
      />
    </BoardDialog>
  )
}

function BoardSettingsDialog({ board, onClose }: { board: BoardMeta | null; onClose: () => void }) {
  const k = useKanban()
  const [project, setProject] = useState('')
  // Null while closed — see RenameBoardDialog on why this can't live inside
  // the mutation callback.
  const slug = board?.slug ?? ''

  useEffect(() => {
    if (board) {
      setProject(board.project_id || '')
    }
  }, [board])

  // The name lives in the rename dialog; '' clears the scope, which also
  // drops the mirrored default_workdir on the backend.
  const save = useBoardWrite(() => updateBoard(slug, { project_id: project }), onClose)

  return (
    <BoardDialog
      confirmLabel={k.save}
      disabled={save.isPending}
      onClose={onClose}
      onConfirm={() => save.mutate()}
      open={Boolean(board)}
      title={board ? k.boardSettingsFor(board.name || board.slug) : k.settingsDots}
    >
      <ProjectPicker onChange={setProject} value={project} />
    </BoardDialog>
  )
}

export function BoardSwitcher() {
  const k = useKanban()
  // Delete reuses the app-wide label, the way sessions and profiles do.
  const { t } = useI18n()
  const qc = useQueryClient()
  const slug = useValue($boardSlug)
  const { data: boards } = useQuery({ queryFn: fetchBoards, queryKey: BOARDS_KEY, staleTime: 30_000 })
  const [adding, setAdding] = useState(false)
  const [settingsFor, setSettingsFor] = useState<BoardMeta | null>(null)
  const [renameFor, setRenameFor] = useState<BoardMeta | null>(null)
  const [deleteFor, setDeleteFor] = useState<BoardMeta | null>(null)

  // Archive rather than erase, so a mis-click stays recoverable. The backend
  // reverts the active board to default; drop our override to follow it.
  const confirmDelete = async (target: BoardMeta) => {
    const { result } = await deleteBoard(target.slug)

    $boardSlug.set('')
    void qc.invalidateQueries({ queryKey: BOARDS_KEY })
    host.notify({ kind: 'success', message: k.boardArchived(result.new_path) })
  }

  const runExport = async (target: string) => {
    const os = pluginOs()

    if (!os) {
      return
    }

    await runExportBoardFlow(os, k, target)
  }

  const runImport = async () => {
    const os = pluginOs()

    if (!os) {
      return
    }

    const imported = await runImportBoardFlow(os, k)

    if (imported) {
      $boardSlug.set(imported)
      void qc.invalidateQueries({ queryKey: BOARDS_KEY })
    }
  }

  if (!boards) {
    return null
  }

  const currentSlug = slug || boards.current
  const current = boards.boards.find(meta => meta.slug === currentSlug)
  const label = current?.name || current?.slug || k.board

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button className="h-7 max-w-56 gap-1.5 px-2" size="sm" variant="ghost">
            <span className="min-w-0 flex-1 truncate text-[0.75rem] font-medium leading-none">{label}</span>
            {typeof current?.total === 'number' && (
              <span className="text-[0.6875rem] tabular-nums text-(--ui-text-quaternary)">{current.total}</span>
            )}
            <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.8125rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="center">
          {boards.boards.map(meta => (
            <DropdownMenuItem
              key={meta.slug}
              onSelect={() => $boardSlug.set(meta.slug === boards.current ? '' : meta.slug)}
            >
              {meta.name || meta.slug}
              {typeof meta.total === 'number' && (
                <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{meta.total}</span>
              )}
              {meta.slug === currentSlug && <Codicon className="ml-auto" name="check" size="0.8rem" />}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          {current && (
            <>
              <DropdownMenuItem onSelect={() => setRenameFor(current)}>
                <Codicon name="edit" size="0.8rem" />
                {k.renameDots}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => setSettingsFor(current)}>
                <Codicon name="settings-gear" size="0.8rem" />
                {k.settingsDots}
              </DropdownMenuItem>
            </>
          )}
          <DropdownMenuItem onSelect={() => setAdding(true)}>
            <Codicon name="add" size="0.8rem" />
            {k.newBoardDots}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {current && (
            <DropdownMenuItem onSelect={() => void runExport(current.slug)}>
              <Codicon name="package" size="0.8rem" />
              {k.exportDots}
            </DropdownMenuItem>
          )}
          <DropdownMenuItem onSelect={() => void runImport()}>
            <Codicon name="cloud-download" size="0.8rem" />
            {k.importDots}
          </DropdownMenuItem>
          {/* `default` is the fallback every board reverts to — the backend
              refuses to remove it, so it never offers the action. */}
          {current && current.slug !== DEFAULT_BOARD && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => setDeleteFor(current)} variant="destructive">
                <Codicon name="trash" size="0.8rem" />
                {t.common.delete}
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <NewBoardDialog onClose={() => setAdding(false)} open={adding} />
      <RenameBoardDialog board={renameFor} onClose={() => setRenameFor(null)} />
      <BoardSettingsDialog board={settingsFor} onClose={() => setSettingsFor(null)} />
      <ConfirmDialog
        confirmLabel={t.common.delete}
        description={k.deleteBoardConfirm}
        destructive
        onClose={() => setDeleteFor(null)}
        onConfirm={() => confirmDelete(deleteFor!)}
        open={Boolean(deleteFor)}
        title={deleteFor ? k.deleteBoardTitle(deleteFor.name || deleteFor.slug) : t.common.delete}
      />
    </>
  )
}
