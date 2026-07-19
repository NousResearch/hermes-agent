/**
 * Titlebar board switcher — the board page projects this into `titleBar.center`
 * (where chat shows the session-title dropdown) via `<Contribute>`, so it
 * exists exactly while the page is mounted — no route sniffing. Same chrome as
 * the session title: quiet label + chevron, menu on click.
 */

import {
  Button,
  Codicon,
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
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { $boardSlug, BOARDS_KEY, createBoard, fetchBoards, fetchProjects, PROJECTS_KEY, updateBoard } from './api'
import type { BoardMeta } from './types'
import { errText, FIELD_LABEL } from './ui'

const NO_PROJECT = '__none__'

/** Board scope = a first-class Hermes project. Its primary repo becomes the
 *  board's default workspace root; new tasks inherit it as a worktree with a
 *  deterministic branch. "No project" falls back to scratch sandboxes. */
function ProjectPicker({ onChange, value }: { onChange: (id: string) => void; value: string }) {
  const { data } = useQuery({ queryKey: PROJECTS_KEY, queryFn: fetchProjects, staleTime: 30_000 })
  const projects = data?.projects ?? []

  return (
    <label className="flex flex-col gap-1">
      <span className={FIELD_LABEL}>Project</span>
      <Select onValueChange={id => onChange(id === NO_PROJECT ? '' : id)} value={value || NO_PROJECT}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NO_PROJECT}>No project (scratch sandboxes)</SelectItem>
          {projects.map(project => (
            <SelectItem key={project.id} value={project.id}>
              {project.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-[0.6875rem] leading-relaxed text-(--ui-text-quaternary)">
        New tasks run in the project’s repo (a worktree per task); each task can still override its workspace at
        creation. Manage projects with <span className="font-mono">hermes project</span>.
      </span>
    </label>
  )
}

function NewBoardDialog({ onClose, open }: { onClose: () => void; open: boolean }) {
  const qc = useQueryClient()
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

  const create = useMutation({
    mutationFn: () => createBoard(slug, name.trim(), project || undefined),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: result => {
      $boardSlug.set(result.board.slug)
      void qc.invalidateQueries({ queryKey: BOARDS_KEY })
      onClose()
    }
  })

  return (
    <Dialog onOpenChange={o => !o && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>New board</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className={FIELD_LABEL}>Name</span>
            <Input
              autoFocus
              onChange={event => setName(event.target.value)}
              onKeyDown={event => event.key === 'Enter' && slug && !project && create.mutate()}
              placeholder="Board name"
              value={name}
            />
            {slug && <span className="text-[0.6875rem] text-(--ui-text-quaternary)">slug: {slug}</span>}
          </label>
          <ProjectPicker onChange={setProject} value={project} />
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="text">
            Cancel
          </Button>
          <Button disabled={!slug || create.isPending} onClick={() => create.mutate()}>
            Create board
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function BoardSettingsDialog({ board, onClose }: { board: BoardMeta | null; onClose: () => void }) {
  const qc = useQueryClient()
  const [name, setName] = useState('')
  const [project, setProject] = useState('')

  useEffect(() => {
    if (board) {
      setName(board.name || '')
      setProject(board.project_id || '')
    }
  }, [board])

  const save = useMutation({
    // Slug is immutable; send name + project_id ('' clears the scope, which
    // also drops the mirrored default_workdir on the backend).
    mutationFn: () => updateBoard(board!.slug, { name: name.trim(), project_id: project }),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: BOARDS_KEY })
      onClose()
    }
  })

  return (
    <Dialog onOpenChange={o => !o && onClose()} open={Boolean(board)}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Board settings{board ? ` — ${board.name || board.slug}` : ''}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className={FIELD_LABEL}>Name</span>
            <Input onChange={event => setName(event.target.value)} placeholder="Board name" value={name} />
            {board && <span className="text-[0.6875rem] text-(--ui-text-quaternary)">slug: {board.slug}</span>}
          </label>
          <ProjectPicker onChange={setProject} value={project} />
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="text">
            Cancel
          </Button>
          <Button disabled={save.isPending} onClick={() => save.mutate()}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function BoardSwitcher() {
  const slug = useValue($boardSlug)
  const { data: boards } = useQuery({ queryFn: fetchBoards, queryKey: BOARDS_KEY, staleTime: 30_000 })
  const [adding, setAdding] = useState(false)
  const [settingsFor, setSettingsFor] = useState<BoardMeta | null>(null)

  if (!boards) {
    return null
  }

  const currentSlug = slug || boards.current
  const current = boards.boards.find(meta => meta.slug === currentSlug)
  const label = current?.name || current?.slug || 'Board'

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
            <DropdownMenuItem onSelect={() => setSettingsFor(current)}>
              <Codicon name="settings-gear" size="0.8rem" />
              Board settings…
            </DropdownMenuItem>
          )}
          <DropdownMenuItem onSelect={() => setAdding(true)}>
            <Codicon name="add" size="0.8rem" />
            New board…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <NewBoardDialog onClose={() => setAdding(false)} open={adding} />
      <BoardSettingsDialog board={settingsFor} onClose={() => setSettingsFor(null)} />
    </>
  )
}
