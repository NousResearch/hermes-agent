import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { createKanbanTask, getKanbanAssignees, parseKanbanApiError } from '@/kanban-api'
import { $kanbanActiveBoard, $kanbanBoard, refreshKanbanBoard } from '@/store/kanban'
import { notify } from '@/store/notifications'

// Radix Select items may not use '' — sentinel for "no parent link".
const NO_PARENT = '__none__'

interface NewCardDialogProps {
  onClose: () => void
  open: boolean
}

export function NewCardDialog({ onClose, open }: NewCardDialogProps) {
  const { t } = useI18n()
  const k = t.kanban
  const board = useStore($kanbanBoard)
  const activeBoard = useStore($kanbanActiveBoard)

  const [title, setTitle] = useState('')
  const [assignee, setAssignee] = useState('')
  const [body, setBody] = useState('')
  const [priority, setPriority] = useState('0')
  const [parent, setParent] = useState(NO_PARENT)
  const [assignees, setAssignees] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setTitle('')
    setAssignee('')
    setBody('')
    setPriority('0')
    setParent(NO_PARENT)
    setError(null)
    setSaving(false)

    let cancelled = false

    // Assignee choices: the /assignees union (on-disk profiles + names already
    // on the board), falling back to the board payload's assignees.
    getKanbanAssignees($kanbanActiveBoard.get())
      .then(list => {
        if (!cancelled) {
          setAssignees(list.map(entry => entry.name))
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAssignees($kanbanBoard.get()?.assignees ?? [])
        }
      })

    return () => {
      cancelled = true
    }
  }, [open, activeBoard])

  const parentOptions = useMemo(
    () => (board?.columns ?? []).flatMap(column => column.tasks.map(task => ({ id: task.id, title: task.title }))),
    [board]
  )

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    const trimmedTitle = title.trim()

    if (!trimmedTitle) {
      setError(k.titleRequired)

      return
    }

    if (!assignee) {
      setError(k.assigneeRequired)

      return
    }

    setSaving(true)
    setError(null)

    try {
      const parsedPriority = Number.parseInt(priority, 10)

      const response = await createKanbanTask(
        {
          title: trimmedTitle,
          assignee,
          body: body.trim() || undefined,
          priority: Number.isInteger(parsedPriority) ? parsedPriority : 0,
          parents: parent === NO_PARENT ? [] : [parent]
        },
        $kanbanActiveBoard.get()
      )

      notify({ kind: 'success', message: k.created })

      if (response.warning) {
        notify({ kind: 'warning', message: response.warning })
      }

      void refreshKanbanBoard()
      onClose()
    } catch (err) {
      setError(parseKanbanApiError(err).detail || k.createFailed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{k.newCardTitle}</DialogTitle>
          <DialogDescription>{k.newCardDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <Field htmlFor="kanban-card-title" label={k.titleLabel}>
            <Input
              autoFocus
              id="kanban-card-title"
              onChange={event => setTitle(event.target.value)}
              placeholder={k.titlePlaceholder}
              value={title}
            />
          </Field>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="kanban-card-assignee" label={k.assigneeLabel}>
              <Select onValueChange={setAssignee} value={assignee || undefined}>
                <SelectTrigger className="h-9 rounded-md" id="kanban-card-assignee">
                  <SelectValue placeholder={k.assigneePlaceholder} />
                </SelectTrigger>
                <SelectContent>
                  {assignees.map(name => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field htmlFor="kanban-card-priority" label={k.priorityLabel}>
              <Input
                className="tabular-nums"
                id="kanban-card-priority"
                onChange={event => setPriority(event.target.value)}
                type="number"
                value={priority}
              />
            </Field>
          </div>

          <Field htmlFor="kanban-card-body" label={k.bodyLabel} optional optionalLabel={k.optional}>
            <Textarea
              className="min-h-24"
              id="kanban-card-body"
              onChange={event => setBody(event.target.value)}
              placeholder={k.bodyPlaceholder}
              value={body}
            />
          </Field>

          {parentOptions.length > 0 && (
            <Field htmlFor="kanban-card-parent" label={k.parentLabel} optional optionalLabel={k.optional}>
              <Select onValueChange={setParent} value={parent}>
                <SelectTrigger className="h-9 rounded-md" id="kanban-card-parent">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_PARENT}>{k.parentNone}</SelectItem>
                  {parentOptions.map(option => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.title || option.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          )}

          {error && <div className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div>}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={saving} type="submit">
              {saving ? k.creating : k.createCard}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function Field({
  children,
  htmlFor,
  label,
  optional,
  optionalLabel
}: {
  children: React.ReactNode
  htmlFor: string
  label: string
  optional?: boolean
  optionalLabel?: string
}) {
  return (
    <div className="grid gap-1.5">
      <label className="flex items-baseline gap-2 text-xs font-medium text-foreground" htmlFor={htmlFor}>
        {label}
        {optional && <span className="text-[0.65rem] font-normal text-muted-foreground">{optionalLabel}</span>}
      </label>
      {children}
    </div>
  )
}
