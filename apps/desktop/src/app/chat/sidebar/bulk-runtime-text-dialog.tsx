import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'

export type BulkRuntimeTextMode = 'prompt' | 'steer'

interface BulkRuntimeTextDialogProps {
  count: number
  mode: BulkRuntimeTextMode | null
  onOpenChange: (open: boolean) => void
  onSubmit: (mode: BulkRuntimeTextMode, text: string) => void
  pending: boolean
}

export function BulkRuntimeTextDialog({ count, mode, onOpenChange, onSubmit, pending }: BulkRuntimeTextDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar.bulk
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const open = mode !== null

  useEffect(() => {
    if (open) {
      setValue('')
      window.setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open, mode])

  if (!mode) {
    return null
  }

  const submit = () => {
    const text = value.trim()

    if (!text || pending) {
      return
    }

    onSubmit(mode, text)
  }

  const title = mode === 'prompt' ? s.promptDialogTitle(count) : s.steerDialogTitle(count)
  const desc = mode === 'prompt' ? s.promptDialogDesc : s.steerDialogDesc
  const placeholder = mode === 'prompt' ? s.promptPlaceholder : s.steerPlaceholder
  const submitLabel = mode === 'prompt' ? s.promptSubmit : s.steerSubmit

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{desc}</DialogDescription>
        </DialogHeader>
        <Textarea
          autoFocus
          className="min-h-28 resize-y"
          disabled={pending}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              event.preventDefault()
              submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={placeholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={pending} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={pending || !value.trim()} onClick={submit} type="button">
            {submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
