import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { copyTextToClipboard } from '@/lib/desktop-fs'
import { detectSelectionLanguage, languageLabel, type SelectionLanguageCode } from '@/lib/selection-language'
import type { SelectionTranslateMode } from '@/lib/selection-language'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $selectionTranslate,
  closeSelectionTranslate,
  retrySelectionTranslate,
  setSelectionTranslateTarget
} from '@/store/selection-translate'
import { $selectionTranslateMode, setSelectionTranslateMode } from '@/store/selection-translate-prefs'

const TARGETS: SelectionLanguageCode[] = ['ar', 'en']
const MODES: SelectionTranslateMode[] = ['auto', 'ar', 'en']

export function SelectionTranslateDialog() {
  const { t } = useI18n()
  const state = useStore($selectionTranslate)
  const mode = useStore($selectionTranslateMode)
  const copy = t.selectionTranslate

  useEffect(() => {
    if (!state.open) {
      return
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeSelectionTranslate()
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [state.open])

  const detected = state.source ? detectSelectionLanguage(state.source) : 'en'
  const dir = state.target === 'ar' ? 'rtl' : 'ltr'

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          closeSelectionTranslate()
        }
      }}
      open={state.open}
    >
      <DialogContent className="max-w-lg gap-3" fitContent showCloseButton>
        <DialogHeader>
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription>{copy.providerNote}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>
            {copy.detected}: <span className="font-medium text-foreground">{languageLabel(detected)}</span>
          </span>
          <span aria-hidden>·</span>
          <label className="flex items-center gap-1.5">
            <span>{copy.target}</span>
            <select
              aria-label={copy.target}
              className="h-7 rounded-md border border-(--ui-stroke-secondary) bg-background px-2 text-xs text-foreground"
              onChange={event => setSelectionTranslateTarget(event.target.value as SelectionLanguageCode)}
              value={state.target}
            >
              {TARGETS.map(code => (
                <option key={code} value={code}>
                  {languageLabel(code)}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-(--ui-stroke-secondary) bg-muted/30 px-2.5 py-2">
          <span className="text-xs font-medium text-foreground/85">{copy.defaultPreference}</span>
          <div className="flex flex-wrap gap-1">
            {MODES.map(item => (
              <Button
                className={cn('h-7 rounded-full px-2.5 text-[0.7rem]', mode === item && 'bg-primary/15 text-primary')}
                key={item}
                onClick={() => setSelectionTranslateMode(item)}
                size="sm"
                type="button"
                variant={mode === item ? 'secondary' : 'ghost'}
              >
                {item === 'auto' ? copy.modeAuto : item === 'ar' ? copy.modeArabic : copy.modeEnglish}
              </Button>
            ))}
          </div>
        </div>

        <section className="space-y-1">
          <h3 className="text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">{copy.source}</h3>
          <p className="max-h-28 overflow-auto whitespace-pre-wrap rounded-lg border border-(--ui-stroke-secondary) bg-background/60 px-3 py-2 text-sm text-foreground/90">
            {state.source}
          </p>
        </section>

        <section className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">{copy.translation}</h3>
            {state.status === 'ready' && state.result ? (
              <Button
                className="h-7 px-2 text-[0.7rem]"
                onClick={() => {
                  void copyTextToClipboard(state.result)
                    .then(() => notify({ kind: 'info', message: copy.copied, durationMs: 1200 }))
                    .catch(error => notifyError(error, copy.copyFailed))
                }}
                size="sm"
                type="button"
                variant="ghost"
              >
                {copy.copy}
              </Button>
            ) : null}
          </div>

          <div
            className={cn(
              'min-h-24 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg border border-(--ui-stroke-secondary) bg-background px-3 py-2 text-sm',
              state.status === 'error' && 'border-destructive/40'
            )}
            dir={dir}
            lang={state.target}
          >
            {state.status === 'loading' ? (
              <span className="text-muted-foreground">{copy.translating}</span>
            ) : state.status === 'error' ? (
              <div className="space-y-2">
                <p className="text-destructive">{state.error || copy.failed}</p>
                <Button onClick={() => retrySelectionTranslate()} size="sm" type="button" variant="secondary">
                  {copy.retry}
                </Button>
              </div>
            ) : (
              <span className="text-foreground">{state.result}</span>
            )}
          </div>
        </section>
      </DialogContent>
    </Dialog>
  )
}
