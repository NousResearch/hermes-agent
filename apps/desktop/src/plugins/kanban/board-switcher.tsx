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
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { $boardSlug, BOARDS_KEY, createBoard, fetchBoards } from './api'
import { errText } from './ui'

function NewBoardDialog({ onClose, open }: { onClose: () => void; open: boolean }) {
  const qc = useQueryClient()
  const [name, setName] = useState('')

  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  useEffect(() => {
    if (open) {
      setName('')
    }
  }, [open])

  const create = useMutation({
    mutationFn: () => createBoard(slug, name.trim()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: result => {
      $boardSlug.set(result.board.slug)
      void qc.invalidateQueries({ queryKey: BOARDS_KEY })
      onClose()
    }
  })

  return (
    <Dialog onOpenChange={o => !o && onClose()} open={open}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>New board</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-1.5">
          <Input
            autoFocus
            onChange={event => setName(event.target.value)}
            onKeyDown={event => event.key === 'Enter' && slug && create.mutate()}
            placeholder="Board name"
            value={name}
          />
          {slug && <span className="text-[0.6875rem] text-(--ui-text-quaternary)">slug: {slug}</span>}
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

export function BoardSwitcher() {
  const slug = useValue($boardSlug)
  const { data: boards } = useQuery({ queryFn: fetchBoards, queryKey: BOARDS_KEY, staleTime: 30_000 })
  const [adding, setAdding] = useState(false)

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
          <DropdownMenuItem onSelect={() => setAdding(true)}>
            <Codicon name="add" size="0.8rem" />
            New board…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <NewBoardDialog onClose={() => setAdding(false)} open={adding} />
    </>
  )
}
