import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'

interface ShareControlsProps {
  // True when the shown map was loaded from a pasted code (not the live scan).
  imported?: boolean
  // Decode + apply a pasted code. Returns an error string to show inline, or null.
  onImport?: (code: string) => null | string
  onResetMap?: () => void
  // The current map serialized as a WoW-style share code (the copy target).
  shareCode?: string
}

const SECTION_LABEL = 'text-[0.6rem] font-medium uppercase tracking-wider text-muted-foreground/55'

// WoW-talent-loadout style sharing: one icon button opens a popover with the
// current map's code (copy/export) and a paste box (import) — drop a string,
// see the build. Lives bottom-right of the map, mirroring the legend.
export function ShareControls({ imported = false, onImport, onResetMap, shareCode }: ShareControlsProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<null | string>(null)

  const apply = () => {
    const code = draft.trim()

    if (!code) {
      setError(t.starmap.importEmpty)

      return
    }

    const err = onImport?.(code) ?? null
    setError(err)

    if (err === null) {
      setOpen(false)
      setDraft('')
    }
  }

  return (
    <Popover
      onOpenChange={next => {
        setOpen(next)

        if (!next) {
          setError(null)
        }
      }}
      open={open}
    >
      <PopoverTrigger asChild>
        <Button
          aria-label={t.starmap.shareTitle}
          className="size-7 text-muted-foreground hover:bg-(--ui-row-hover-background) hover:text-foreground data-[state=open]:bg-(--ui-row-hover-background) data-[state=open]:text-foreground"
          size="icon"
          title={t.starmap.shareTitle}
          variant="ghost"
        >
          <Codicon name="send" size="0.8rem" />
        </Button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-72 p-0" side="top" sideOffset={8}>
        <div className="space-y-2 px-3 py-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className={SECTION_LABEL}>{t.starmap.share}</span>
            {imported && (
              <button
                className="text-[0.62rem] text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline"
                onClick={() => {
                  onResetMap?.()
                  setOpen(false)
                }}
                type="button"
              >
                {t.starmap.resetToMine}
              </button>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <div className="flex h-7 min-w-0 flex-1 items-center rounded-md bg-foreground/5 px-2">
              <span className="truncate font-mono text-[0.62rem] text-muted-foreground/90">{shareCode || '—'}</span>
            </div>
            <CopyButton
              appearance="button"
              buttonSize="icon"
              className="size-7 shrink-0 text-muted-foreground hover:bg-(--ui-row-hover-background) hover:text-foreground"
              disabled={!shareCode}
              label={t.starmap.copy}
              showLabel={false}
              text={shareCode ?? ''}
            />
          </div>
        </div>

        <div className="h-px bg-(--ui-stroke-secondary)" />

        <div className="space-y-2 px-3 py-2.5">
          <span className={SECTION_LABEL}>{t.starmap.importMap}</span>

          <div className="flex items-center gap-1.5">
            <Input
              aria-label={t.starmap.sharePlaceholder}
              className="h-7 flex-1 font-mono text-[0.62rem]"
              onChange={e => {
                setDraft(e.target.value)
                setError(null)
              }}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  apply()
                }
              }}
              placeholder={t.starmap.sharePlaceholder}
              value={draft}
            />
            <Button className="h-7 shrink-0 px-2.5 text-[0.7rem]" disabled={!draft.trim()} onClick={apply} size="sm" type="button">
              {t.starmap.importBtn}
            </Button>
          </div>

          {error && <p className="text-[0.62rem] text-destructive">{error}</p>}
        </div>
      </PopoverContent>
    </Popover>
  )
}
