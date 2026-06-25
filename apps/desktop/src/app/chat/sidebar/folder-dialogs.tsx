import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'

interface FolderNameDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  // 'create' starts blank; 'rename' seeds + selects the current name.
  mode: 'create' | 'rename'
  initialName?: string
  onSubmit: (name: string) => void
}

// Name-entry dialog shared by "New folder" and "Rename folder", built on the
// same Dialog/Input shape as RenameSessionDialog (Enter submits, Escape cancels,
// autofocus + select). The store mutation itself is synchronous, so there is no
// submitting/async state to track here.
export function FolderNameDialog({ open, onOpenChange, mode, initialName = '', onSubmit }: FolderNameDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const [value, setValue] = useState(initialName)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(initialName)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [initialName, open])

  const submit = () => {
    const next = value.trim()

    if (!next) {
      onOpenChange(false)

      return
    }

    onSubmit(next)
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{mode === 'create' ? s.newFolderTitle : s.renameFolderTitle}</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={s.folderNamePlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button onClick={submit} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
