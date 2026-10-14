import { useEffect, useRef, useState } from 'react'

import { ArchiveSkillConfirmDialog, fireOptimistic } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  type ApiRequestScopeToken,
  captureApiRequestScope,
  deleteLearningNode,
  editLearningNode,
  getLearningNode,
  isApiRequestScopeCurrent,
  onApiRequestScopeChange
} from '@/hermes'
import { notifyError } from '@/store/notifications'
import { evictStarmapNode, loadStarmapGraph } from '@/store/starmap'

export interface NodeMenuTarget {
  id: string
  kind: 'memory' | 'skill'
  label: string
  x: number
  y: number
}

interface NodeContextMenuProps {
  onClose: () => void
  onNodeRemoved: () => void
  target: NodeMenuTarget | null
}

interface EditState {
  content: string
  id: string
  label: string
  owner: ApiRequestScopeToken
}

interface DeleteState extends Omit<NodeMenuTarget, 'x' | 'y'> {
  owner: ApiRequestScopeToken
}

/** Right-click actions for a star-map node: edit (modal) or delete (confirm). */
export function NodeContextMenu({ onClose, onNodeRemoved, target }: NodeContextMenuProps) {
  const [editing, setEditing] = useState<EditState | null>(null)
  const [deleting, setDeleting] = useState<DeleteState | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // Bumped on owner switch so an in-flight openEdit fetch from A can't reopen
  // the editor with A's node content on a same-named profile from connection B.
  const editEpoch = useRef(0)

  useEffect(
    () =>
      onApiRequestScopeChange(() => {
        editEpoch.current += 1
        setEditing(null)
        setDeleting(null)
        setLoading(false)
        setSaving(false)
        setError(null)
      }),
    []
  )

  const noun = target?.kind === 'memory' ? 'memory' : 'skill'

  const openEdit = async () => {
    if (!target) {
      return
    }

    const epoch = editEpoch.current
    const owner = captureApiRequestScope()
    setLoading(true)
    setError(null)

    try {
      const detail = await getLearningNode(target.id, owner)

      if (editEpoch.current !== epoch || !isApiRequestScopeCurrent(owner)) {
        return
      }

      setEditing({ content: detail.content, id: target.id, label: target.label, owner })
      onClose()
    } catch (e) {
      if (isApiRequestScopeCurrent(owner)) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (isApiRequestScopeCurrent(owner)) {
        setLoading(false)
      }
    }
  }

  const save = async () => {
    if (!editing) {
      return
    }

    const { owner } = editing

    setSaving(true)
    setError(null)

    try {
      const res = await editLearningNode(editing.id, editing.content, owner)

      if (!isApiRequestScopeCurrent(owner)) {
        return
      }

      if (!res.ok) {
        throw new Error(res.message)
      }

      setEditing(null)
      void loadStarmapGraph(true)
    } catch (e) {
      if (isApiRequestScopeCurrent(owner)) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (isApiRequestScopeCurrent(owner)) {
        setSaving(false)
      }
    }
  }

  const menuOpen = target && !editing && !deleting

  return (
    <>
      {menuOpen ? (
        <DropdownMenu onOpenChange={open => !open && onClose()} open>
          <DropdownMenuTrigger asChild>
            {/* A zero-size anchor at the canvas click point, as AppContextMenu
                does: Radix positions against it like a real trigger and flips or
                shifts the menu back inside the viewport near the window edges,
                so the destructive row can never be clipped off-window. */}
            <span aria-hidden style={{ left: target.x, position: 'fixed', top: target.y }} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} side="bottom">
            <DropdownMenuLabel className="truncate text-[0.68rem] font-normal text-muted-foreground">
              {target.label}
            </DropdownMenuLabel>
            <DropdownMenuItem
              disabled={loading}
              onSelect={e => {
                // Keep the menu up while the node content loads; openEdit closes it.
                e.preventDefault()
                void openEdit()
              }}
            >
              Edit {noun}…
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() =>
                setDeleting({
                  id: target.id,
                  kind: target.kind,
                  label: target.label,
                  owner: captureApiRequestScope()
                })
              }
              variant="destructive"
            >
              {target.kind === 'skill' ? 'Archive skill' : 'Delete memory'}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}

      <Dialog onOpenChange={value => !value && !saving && setEditing(null)} open={Boolean(editing)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Edit {editing?.label}</DialogTitle>
          </DialogHeader>
          <div className="h-80">
            {editing && (
              <CodeEditor
                filePath={noun === 'skill' ? 'SKILL.md' : 'memory.md'}
                framed
                initialValue={editing.content}
                key={editing.id}
                onCancel={() => !saving && setEditing(null)}
                onChange={content => setEditing(prev => (prev ? { ...prev, content } : prev))}
                onSave={() => void save()}
              />
            )}
          </div>
          {error ? <p className="text-xs text-destructive">{error}</p> : null}
          <DialogFooter>
            <Button disabled={saving} onClick={() => setEditing(null)} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={saving} onClick={() => void save()}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {deleting?.kind === 'skill' ? (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            onNodeRemoved()

            return evictStarmapNode(deleting.id, deleting.owner)
          }}
          onClose={() => setDeleting(null)}
          onFailure={(err, name) => notifyError(err, name)}
          open
          profile={deleting.owner}
          skillId={deleting.id}
          skillName={deleting.label}
        />
      ) : (
        <ConfirmDialog
          confirmLabel="Delete"
          description="This memory is removed permanently."
          destructive
          dismissOnConfirm
          onClose={() => setDeleting(null)}
          onConfirm={() => {
            if (!deleting) {
              return
            }

            const { id, label, owner } = deleting
            const rollback = evictStarmapNode(id, owner)
            onNodeRemoved()

            fireOptimistic(
              deleteLearningNode(id, owner).then(res => {
                if (!res.ok) {
                  throw new Error(res.message)
                }
              }),
              rollback,
              err => notifyError(err, label)
            )
          }}
          open={Boolean(deleting)}
          title={`Delete ${deleting?.label ?? ''}?`}
        />
      )}
    </>
  )
}
