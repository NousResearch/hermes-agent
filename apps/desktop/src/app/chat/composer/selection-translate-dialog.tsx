import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Command, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { copyTextToClipboard } from '@/lib/desktop-fs'
import { Check, ChevronDown } from '@/lib/icons'
import {
  languageLabel,
  listTranslationLanguages,
  normalizeTranslationLanguageCode,
  type SelectionLanguageCode,
  type TranslationLanguageOption
} from '@/lib/selection-language'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $selectionTranslate,
  closeSelectionTranslate,
  retrySelectionTranslate,
  setSelectionTranslateTarget
} from '@/store/selection-translate'

export function SelectionTranslateDialog() {
  const { locale, t } = useI18n()
  const state = useStore($selectionTranslate)
  const copy = t.selectionTranslate
  const languages = useMemo(() => listTranslationLanguages(locale), [locale])

  const errorCopy =
    state.error === 'too-long' ? copy.tooLong : state.error === 'empty-result' ? copy.emptyResult : copy.failed

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

        <div className="space-y-1">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-foreground">{copy.target}</span>
            <TranslationLanguagePicker
              ariaLabel={copy.target}
              customHint={copy.languageTagHint}
              formatCustomLanguage={copy.useLanguageTag}
              languages={languages}
              locale={locale}
              noResults={copy.noLanguages}
              onSelect={setSelectionTranslateTarget}
              searchPlaceholder={copy.searchLanguages}
              value={state.target}
            />
          </div>
          <p className="text-xs text-muted-foreground">{copy.preferredHint}</p>
        </div>

        <section className="space-y-1">
          <h3 className="text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">{copy.source}</h3>
          <p
            className="max-h-28 overflow-auto whitespace-pre-wrap rounded-lg border border-(--ui-stroke-secondary) bg-background/60 px-3 py-2 text-sm text-foreground/90"
            dir="auto"
          >
            {state.source}
          </p>
        </section>

        <section className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">
              {copy.translation}
            </h3>
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
            aria-atomic="true"
            aria-busy={state.status === 'loading'}
            aria-live={state.status === 'error' ? 'assertive' : 'polite'}
            className={cn(
              'min-h-24 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg border border-(--ui-stroke-secondary) bg-background px-3 py-2 text-sm',
              state.status === 'error' && 'border-destructive/40'
            )}
            dir="auto"
            role={state.status === 'error' ? 'alert' : 'status'}
          >
            {state.status === 'loading' ? (
              <span className="text-muted-foreground">{copy.translating}</span>
            ) : state.status === 'error' ? (
              <div className="space-y-2">
                <p className="text-destructive">{errorCopy}</p>
                {state.error !== 'too-long' ? (
                  <Button onClick={() => retrySelectionTranslate()} size="sm" type="button" variant="secondary">
                    {copy.retry}
                  </Button>
                ) : null}
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

function TranslationLanguagePicker({
  ariaLabel,
  customHint,
  formatCustomLanguage,
  languages,
  locale,
  noResults,
  onSelect,
  searchPlaceholder,
  value
}: {
  ariaLabel: string
  customHint: string
  formatCustomLanguage: (name: string, tag: string) => string
  languages: TranslationLanguageOption[]
  locale: string
  noResults: string
  onSelect: (code: SelectionLanguageCode) => void
  searchPlaceholder: string
  value: SelectionLanguageCode
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const query = normalize(search)

  const filtered = languages.filter(
    language =>
      !query ||
      normalize(language.name).includes(query) ||
      normalize(languageLabel(language.code)).includes(query) ||
      language.code.includes(query)
  )

  const current = languages.find(language => language.code === value)
  const canonicalSearch = normalizeTranslationLanguageCode(search)
  const isSuggestedTarget = canonicalSearch ? languages.some(language => language.code === canonicalSearch) : false

  const customTarget =
    canonicalSearch &&
    !isSuggestedTarget &&
    (filtered.length === 0 || search.trim().length === 3 || search.includes('-'))
      ? canonicalSearch
      : null

  const customName = customTarget ? languageLabel(customTarget, locale) : null

  const selectTarget = (target: SelectionLanguageCode) => {
    onSelect(target)
    setOpen(false)
    setSearch('')
  }

  return (
    <Popover
      onOpenChange={nextOpen => {
        setOpen(nextOpen)

        if (!nextOpen) {
          setSearch('')
        }
      }}
      open={open}
    >
      <PopoverTrigger asChild>
        <Button
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-label={ariaLabel}
          className="min-w-40 justify-between"
          role="combobox"
          size="sm"
          type="button"
          variant="outline"
        >
          <span className="truncate">{current?.name ?? languageLabel(value, locale)}</span>
          <ChevronDown className="size-3 shrink-0 opacity-70" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-64 p-0">
        <Command className="bg-transparent" shouldFilter={false}>
          <CommandInput
            aria-describedby="selection-translate-language-tag-hint"
            onKeyDown={event => {
              if (event.key === 'Enter' && customTarget) {
                event.preventDefault()
                event.stopPropagation()
                selectTarget(customTarget)
              }
            }}
            onValueChange={setSearch}
            placeholder={searchPlaceholder}
            value={search}
          />
          <CommandList className="max-h-72 p-1">
            {customTarget && customName ? (
              <CommandItem onSelect={() => selectTarget(customTarget)} value={`custom-${customTarget}`}>
                <Check className="invisible size-3.5 shrink-0" />
                <span className="min-w-0 flex-1 truncate">{formatCustomLanguage(customName, customTarget)}</span>
              </CommandItem>
            ) : null}
            {filtered.length === 0 && !customTarget ? (
              <div className="py-6 text-center text-sm text-muted-foreground">{noResults}</div>
            ) : null}
            {filtered.map(language => (
              <CommandItem key={language.code} onSelect={() => selectTarget(language.code)} value={language.code}>
                <Check className={cn('size-3.5 shrink-0 text-primary', language.code !== value && 'invisible')} />
                <span className="min-w-0 flex-1 truncate">{language.name}</span>
                <span className="font-mono text-[0.65rem] uppercase text-(--ui-text-tertiary)">{language.code}</span>
              </CommandItem>
            ))}
          </CommandList>
          <p
            className="border-t border-border px-3 py-2 text-[0.7rem] text-muted-foreground"
            id="selection-translate-language-tag-hint"
          >
            {customHint}
          </p>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
