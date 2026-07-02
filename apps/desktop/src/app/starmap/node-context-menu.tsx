import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { deleteLearningNode, editLearningNode, getLearningNode } from '@/hermes'

export interface NodeMenuTarget {
  id: string
  kind: 'memory' | 'skill'
  label: string
  x: number
  y: number
}

interface NodeContextMenuProps {
  onChanged: () => void
  onClose: () => void
  target: NodeMenuTarget | null
}

interface EditState {
  content: string
  id: string
  label: string
}

/** Right-click actions for a star-map node: edit (modal) or delete (confirm). */
export function NodeContextMenu({ onChanged, onClose, target }: NodeContextMenuProps) {
  const [editing, setEditing] = useState<EditState | null>(null)
  const [deleting, setDeleting] = useState<{ id: string; label: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  const noun = target?.kind === 'memory' ? 'memory' : 'skill'

  const openEdit = async () => {
    if (!target) {
      return
    }

    setLoading(true)
    setError(null)
    try {
      const detail = await getLearningNode(target.id)
      setEditing({ content: detail.content, id: target.id, label: target.label })
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const save = async () => {
    if (!editing) {
      return
    }

    setSaving(true)
    setError(null)
    try {
      const res = await editLearningNode(editing.id, editing.content)
      if (!res.ok) {
        throw new Error(res.message)
      }
      setEditing(null)
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const menuOpen = target && !editing && !deleting

  return (
    <>
      {menuOpen ? (
        <>
          <div className="fixed inset-0 z-50" onClick={onClose} onContextMenu={e => e.preventDefault()} />
          <div
            className="fixed z-50 min-w-36 overflow-hidden rounded-md border border-border bg-popover py-1 text-sm shadow-md"
            style={{ left: target.x, top: target.y }}
          >
            <div className="truncate px-3 py-1 text-xs text-muted-foreground">{target.label}</div>
            <button
              className="block w-full px-3 py-1 text-left hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
              disabled={loading}
              onClick={() => void openEdit()}
              type="button"
            >
              Edit {noun}…
            </button>
            <button
              className="block w-full px-3 py-1 text-left text-destructive hover:bg-destructive/10"
              onClick={() => {
                setDeleting({ id: target.id, label: target.label })
                onClose()
              }}
              type="button"
            >
              Delete {noun}
            </button>
          </div>
        </>
      ) : null}

      <Dialog onOpenChange={value => !value && !saving && setEditing(null)} open={Boolean(editing)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Edit {editing?.label}</DialogTitle>
          </DialogHeader>
          <Textarea
            className="h-80 font-mono text-xs"
            onChange={e => setEditing(prev => (prev ? { ...prev, content: e.target.value } : prev))}
            value={editing?.content ?? ''}
          />
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

      <ConfirmDialog
        confirmLabel="Delete"
        description={
          noun === 'skill'
            ? 'The skill is archived and can be restored with `hermes curator restore`.'
            : 'This memory is removed permanently.'
        }
        destructive
        onClose={() => setDeleting(null)}
        onConfirm={async () => {
          if (!deleting) {
            return
          }

          const res = await deleteLearningNode(deleting.id)
          if (!res.ok) {
            throw new Error(res.message)
          }
          onChanged()
        }}
        open={Boolean(deleting)}
        title={`Delete ${deleting?.label ?? ''}?`}
      />
    </>
  )
}
