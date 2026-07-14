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
import { Globe } from '@/lib/icons'

const URL_HINT = /^https?:\/\//i

export function UrlDialog({
  inputRef,
  onChange,
  onOpenChange,
  onSubmit,
  open,
  value
}: {
  inputRef: React.RefObject<HTMLInputElement | null>
  onChange: (value: string) => void
  onOpenChange: (open: boolean) => void
  onSubmit: () => void
  open: boolean
  value: string
}) {
  const { t } = useI18n()
  const c = t.composer
  const trimmed = value.trim()
  const looksLikeUrl = trimmed.length > 0 && URL_HINT.test(trimmed)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md gap-5">
        <DialogHeader>
          <DialogTitle icon={Globe}>{c.attachUrlTitle}</DialogTitle>
          <DialogDescription>{c.attachUrlDesc}</DialogDescription>
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
              inputMode="url"
              onChange={e => onChange(e.target.value)}
              placeholder={c.urlPlaceholder}
              ref={inputRef}
              spellCheck={false}
              value={value}
            />
            {trimmed.length > 0 && !looksLikeUrl && (
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
            <Button disabled={!looksLikeUrl} type="submit">
              {c.attach}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
