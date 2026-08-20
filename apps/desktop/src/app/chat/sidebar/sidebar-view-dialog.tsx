import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import {
  applySavedSidebarView,
  deleteSavedSidebarView,
  renameSavedSidebarView,
  saveCurrentSidebarView,
  type SavedSidebarView,
  updateSavedSidebarView
} from '@/store/sidebar-views'

export type SidebarViewDialogState =
  | { kind: 'apply'; view: SavedSidebarView }
  | { kind: 'delete'; view: SavedSidebarView }
  | { kind: 'rename'; view: SavedSidebarView }
  | { kind: 'save' }
  | { kind: 'update'; view: SavedSidebarView }

interface SidebarViewDialogProps {
  dialog: SidebarViewDialogState | null
  onClose: () => void
}

export function SidebarViewDialog({ dialog, onClose }: SidebarViewDialogProps) {
  const { t } = useI18n()
  const copy = t.sidebar.viewMenu
  const [name, setName] = useState('')

  useEffect(() => {
    setName(dialog?.kind === 'rename' ? dialog.view.name : '')
  }, [dialog])

  if (!dialog) {
    return null
  }

  if (dialog.kind === 'apply' || dialog.kind === 'delete' || dialog.kind === 'update') {
    const applying = dialog.kind === 'apply'
    const deleting = dialog.kind === 'delete'

    return (
      <ConfirmDialog
        confirmLabel={applying ? copy.apply : deleting ? t.common.delete : copy.update}
        description={
          applying
            ? copy.applyDescription(dialog.view.name, dialog.view.state.profileScope)
            : deleting
              ? copy.deleteDescription(dialog.view.name)
              : copy.updateDescription(dialog.view.name)
        }
        destructive={deleting}
        onClose={onClose}
        onConfirm={() => {
          if (applying) {
            applySavedSidebarView(dialog.view.id)
          } else if (deleting) {
            deleteSavedSidebarView(dialog.view.id)
          } else {
            updateSavedSidebarView(dialog.view.id)
          }
        }}
        open
        title={applying ? copy.applyTitle : deleting ? copy.deleteTitle : copy.updateTitle}
      />
    )
  }

  const title = dialog.kind === 'save' ? copy.saveTitle : copy.renameTitle

  const submit = () => {
    if (dialog.kind === 'save') {
      if (!saveCurrentSidebarView(name)) {
        return
      }
    } else {
      if (!renameSavedSidebarView(dialog.view.id, name)) {
        return
      }
    }

    onClose()
  }

  return (
    <Dialog onOpenChange={open => !open && onClose()} open>
      <DialogContent>
        <form
          className="contents"
          onSubmit={event => {
            event.preventDefault()
            submit()
          }}
        >
          <DialogHeader>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>
              {dialog.kind === 'save' ? copy.saveDescription : copy.renameDescription}
            </DialogDescription>
          </DialogHeader>

          <label className="flex flex-col gap-1 text-xs text-(--ui-text-secondary)">
            <span>{copy.nameLabel}</span>
            <Input
              autoFocus
              maxLength={80}
              onChange={event => setName(event.target.value)}
              onFocus={event => dialog.kind === 'rename' && event.currentTarget.select()}
              placeholder={copy.namePlaceholder}
              value={name}
            />
          </label>

          <DialogFooter>
            <Button onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={!name.trim()} type="submit">
              {dialog.kind === 'save' ? t.common.save : copy.rename}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
