import type * as React from 'react'

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
import { useI18n } from '@/i18n'
import { Globe, Pencil } from '@/lib/icons'

const URL_HINT = /^https?:\/\//i

/** A value that will serialize as an `@kind:` reference or a slash chip —
 *  non-empty after trimming. Edit mode applies to any reference kind, so this
 *  is intentionally looser than attach mode's URL check. */
function isUsableValue(value: string, editMode: boolean): boolean {
  const trimmed = value.trim()

  if (trimmed.length === 0) {
    return false
  }

  if (editMode) {
    return true
  }

  return URL_HINT.test(trimmed)
}

export function UrlDialog({
  chipEdit,
  inputRef,
  onChange,
  onOpenChange,
  onSubmit,
  open,
  value
}: {
  chipEdit: { chip: HTMLElement; value: string } | null
  inputRef: React.RefObject<HTMLInputElement | null>
  onChange: (value: string) => void
  onOpenChange: (open: boolean) => void
  onSubmit: () => void
  open: boolean
  value: string
}) {
  const { t } = useI18n()
  const c = t.composer
  const editMode = chipEdit !== null
  const trimmed = value.trim()
  const looksLikeUrl = trimmed.length > 0 && URL_HINT.test(trimmed)
  const usable = isUsableValue(value, editMode)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent bodyClassName="gap-5" className="max-w-md">
        <DialogHeader>
          <DialogTitle icon={editMode ? Pencil : Globe}>
            {editMode ? c.editRefTitle : c.attachUrlTitle}
          </DialogTitle>
          <DialogDescription>{editMode ? c.editRefDesc : c.attachUrlDesc}</DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-4"
          onSubmit={e => {
            e.preventDefault()
            onSubmit()
          }}
        >
          <div className="grid gap-1.5">
            <Input
              autoComplete="off"
              autoCorrect="off"
              inputMode={editMode ? 'text' : 'url'}
              onChange={e => onChange(e.target.value)}
              placeholder={editMode ? (chipEdit?.value ?? '') : c.urlPlaceholder}
              ref={inputRef}
              spellCheck={false}
              value={value}
            />
            {!editMode && trimmed.length > 0 && !looksLikeUrl && (
              <p className="text-xs text-muted-foreground/85">
                {c.urlHintPre}
                <span className="font-mono">https://…</span>
              </p>
            )}
          </div>
          <DialogFooter>
            <Button onClick={() => onOpenChange(false)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={!usable} type="submit">
              {editMode ? t.common.save : c.attach}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
