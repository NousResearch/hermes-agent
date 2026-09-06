/**
 * Titlebar workflow switcher — the page projects this into `titleBar.center`
 * (where chat shows the session-title dropdown, and Kanban shows the board)
 * via `<Contribute>`, so it exists exactly while the page is mounted.
 *
 * A dropdown rather than tabs: tabs spend a whole row of chrome on a list that
 * is usually one item long, and they have nowhere to put "delete" that isn't a
 * hover-revealed × on the thing you're currently editing. A menu lists them,
 * marks the open one, and keeps new/rename/delete in the same place whether
 * you have one workflow or nine.
 *
 * Delete is confirmed, because a workflow is the whole document and there's no
 * undo across a switch — `useUndoRedo` is per-canvas and dies with it.
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
  Field,
  Input,
  useValue
} from '@hermes/plugin-sdk'
import { type FormEvent, useState } from 'react'

import {
  $currentId,
  $runCounts,
  $workflows,
  createWorkflow,
  removeWorkflow,
  renameWorkflow,
  type WorkflowDoc
} from './documents'
import { blankScenario } from './scenario'

export function WorkflowSwitcher() {
  const docs = useValue($workflows)
  const currentId = useValue($currentId)
  const runs = useValue($runCounts)
  const current = docs.find(d => d.id === currentId)

  const [naming, setNaming] = useState<'new' | 'rename' | null>(null)
  const [deleting, setDeleting] = useState<WorkflowDoc | null>(null)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button className="h-7 max-w-56 gap-1.5 px-2" size="sm" variant="ghost">
            <span className="min-w-0 flex-1 truncate text-[0.75rem] font-medium leading-none">
              {current?.name ?? 'Workflows'}
            </span>
            <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.8125rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="center">
          {docs.map(doc => (
            <DropdownMenuItem key={doc.id} onSelect={() => $currentId.set(doc.id)}>
              <span className="min-w-0 truncate">{doc.name}</span>
              <span className="ml-auto flex items-center gap-1.5">
                {(runs[doc.id] ?? 0) > 0 && (
                  <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{runs[doc.id]}</span>
                )}
                {doc.id === currentId && <Codicon name="check" size="0.8rem" />}
              </span>
            </DropdownMenuItem>
          ))}
          {docs.length > 0 && <DropdownMenuSeparator />}
          <DropdownMenuItem onSelect={() => setNaming('new')}>
            <Codicon name="add" size="0.8rem" />
            New workflow…
          </DropdownMenuItem>
          {current && (
            <>
              <DropdownMenuItem onSelect={() => setNaming('rename')}>
                <Codicon name="edit" size="0.8rem" />
                Rename…
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => setDeleting(current)} variant="destructive">
                <Codicon name="trash" size="0.8rem" />
                Delete
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Mounted only while open, so the field starts from the current name
          every time rather than from whatever the last open left behind. */}
      {naming && (
        <NameDialog
          initial={naming === 'rename' ? (current?.name ?? '') : ''}
          onClose={() => setNaming(null)}
          onSubmit={name => {
            if (naming === 'rename' && current) {
              renameWorkflow(current.id, name)
            } else {
              createWorkflow(name, blankScenario())
            }
          }}
          verb={naming === 'rename' ? 'Rename' : 'Create'}
        />
      )}

      <ConfirmDialog
        confirmLabel="Delete"
        description={`${deleting?.name} and its ${deleting?.scenario.steps.length ?? 0} steps are gone for good.`}
        destructive
        dismissOnConfirm
        onClose={() => setDeleting(null)}
        onConfirm={() => {
          if (deleting) {
            removeWorkflow(deleting.id)
          }
        }}
        open={deleting !== null}
        title="Delete this workflow?"
      />
    </>
  )
}

/** One dialog for both namings — they differ by a verb and a starting value. */
function NameDialog({
  initial,
  onClose,
  onSubmit,
  verb
}: {
  initial: string
  onClose: () => void
  onSubmit: (name: string) => void
  verb: string
}) {
  const [name, setName] = useState(initial)

  const commit = (e: FormEvent) => {
    e.preventDefault()

    if (name.trim()) {
      onSubmit(name.trim())
      onClose()
    }
  }

  return (
    <Dialog onOpenChange={o => !o && onClose()} open>
      <DialogContent>
        <form onSubmit={commit}>
          <DialogHeader>
            <DialogTitle>{verb} workflow</DialogTitle>
          </DialogHeader>
          <Field htmlFor="wf-name" label="Name">
            <Input
              autoFocus
              id="wf-name"
              onChange={e => setName(e.target.value)}
              placeholder="Design review"
              value={name}
            />
          </Field>
          <DialogFooter>
            <Button onClick={onClose} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={!name.trim()} type="submit">
              {verb}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
