import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { Upload } from '@/lib/icons'

interface ShareControlsProps {
  // True when the shown map was loaded from a pasted code (not the live scan).
  imported?: boolean
  // Decode + apply a pasted code. Returns an error string to show inline, or null.
  onImport?: (code: string) => null | string
  onResetMap?: () => void
  // The current map serialized as a WoW-style share code (the copy target).
  shareCode?: string
}

// Share / import a map as a single code. The textarea shows the current map's
// code (copy it to share); edit/replace it and hit Load to view someone else's.
// One field, one button — a standard Dialog matching rename/create.
export function ShareControls({ imported = false, onImport, onResetMap, shareCode }: ShareControlsProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState('')
  const [error, setError] = useState<null | string>(null)

  const own = (shareCode ?? '').trim()
  const code = value.trim()
  const canLoad = code !== '' && code !== own

  const load = () => {
    if (!code) {
      setError(t.starmap.importEmpty)

      return
    }

    const err = onImport?.(code) ?? null
    setError(err)

    if (err === null) {
      setOpen(false)
    }
  }

  return (
    <div className="flex items-center gap-1">
      {imported && (
        <Button
          className="text-muted-foreground hover:text-foreground"
          onClick={() => onResetMap?.()}
          size="xs"
          variant="text"
        >
          {t.starmap.resetToMine}
        </Button>
      )}

      <Dialog
        onOpenChange={next => {
          setOpen(next)
          setError(null)

          if (next) {
            setValue(shareCode ?? '')
          }
        }}
        open={open}
      >
        <DialogTrigger asChild>
          <Button
            aria-label={t.starmap.shareTitle}
            className="text-muted-foreground hover:text-foreground"
            size="icon"
            title={t.starmap.shareTitle}
            variant="ghost"
          >
            <Upload className="size-3.5" />
          </Button>
        </DialogTrigger>

        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t.starmap.shareTitle}</DialogTitle>
            <DialogDescription>{t.starmap.shareHint}</DialogDescription>
          </DialogHeader>

          {/* One code field: pre-filled with this map's code (copy to share); edit
            or paste another and Load. Copy button floats in on hover, like a
            thread code block. */}
          <div className="group/code relative">
            <textarea
              aria-label={t.starmap.shareTitle}
              className="h-24 w-full resize-none rounded-md bg-foreground/5 p-2.5 pr-9 font-mono text-xs leading-relaxed break-all text-muted-foreground/90 outline-none transition placeholder:text-muted-foreground/50 focus-visible:text-foreground focus-visible:ring-1 focus-visible:ring-ring/40"
              onChange={e => {
                setValue(e.target.value)
                setError(null)
              }}
              placeholder={t.starmap.sharePlaceholder}
              spellCheck={false}
              value={value}
            />
            {code !== '' && (
              <CopyButton
                appearance="inline"
                className="absolute right-1.5 top-1.5 h-5 gap-0 rounded-md px-1 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/code:opacity-100 hover:opacity-100"
                iconClassName="size-3"
                label={t.starmap.copy}
                showLabel={false}
                text={value}
              />
            )}
          </div>

          {error && <p className="text-[0.7rem] text-destructive">{error}</p>}

          <Button className="w-full" disabled={!canLoad} onClick={load} type="button">
            {t.starmap.importBtn}
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  )
}
