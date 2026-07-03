import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Command, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { useIsMobile } from '@/hooks/use-mobile'
import { type Locale, LOCALE_META, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, ChevronDown, Globe } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

export interface LanguageSwitcherProps {
  className?: string
  collapsed?: boolean
  dropUp?: boolean
}

interface LanguageCommandProps {
  allLocales: Array<[Locale, (typeof LOCALE_META)[Locale]]>
  autoFocus?: boolean
  disabled?: boolean
  locale: Locale
  noResults: string
  onSelect: (code: Locale) => void
  searchPlaceholder: string
}

export function LanguageSwitcher({ className, collapsed = false, dropUp = false }: LanguageSwitcherProps) {
  const { isSavingLocale, locale, setLocale, t } = useI18n()
  const [open, setOpen] = useState(false)
  const isMobile = useIsMobile()
  const useMobileSheet = Boolean(dropUp && isMobile)
  const current = LOCALE_META[locale]
  const allLocales = Object.entries(LOCALE_META) as Array<[Locale, typeof current]>
  const title = t.language.switchTo

  const selectLocale = async (code: Locale) => {
    if (code === locale || isSavingLocale) {
      setOpen(false)

      return
    }

    triggerHaptic('selection')

    try {
      await setLocale(code)
      setOpen(false)
      triggerHaptic('success')
    } catch (error) {
      notifyError(error, t.language.saveError)
    }
  }

  const trigger = (
    <Button
      aria-expanded={open}
      aria-label={title}
      className={cn(
        'min-w-32 justify-between gap-2 border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-2.5 text-left text-muted-foreground hover:text-foreground',
        collapsed && 'min-w-0 px-2',
        className
      )}
      disabled={isSavingLocale}
      size="sm"
      title={title}
      type="button"
      variant="outline"
    >
      <span className="inline-flex min-w-0 items-center gap-2">
        <Globe className="size-3.5 shrink-0" />
        {!collapsed && <span className="truncate">{current.name}</span>}
      </span>
      {!collapsed && <ChevronDown className="size-3 shrink-0 opacity-70" />}
    </Button>
  )

  if (useMobileSheet) {
    return (
      <Sheet onOpenChange={setOpen} open={open}>
        <SheetTrigger asChild>{trigger}</SheetTrigger>
        <SheetContent className="max-h-[min(28rem,80vh)] rounded-t-xl" side="bottom">
          <SheetHeader>
            <SheetTitle>{title}</SheetTitle>
            <SheetDescription>{t.language.description}</SheetDescription>
          </SheetHeader>
          <LanguageCommand
            allLocales={allLocales}
            disabled={isSavingLocale}
            locale={locale}
            noResults={t.language.noResults}
            onSelect={code => void selectLocale(code)}
            searchPlaceholder={t.language.searchPlaceholder}
          />
        </SheetContent>
      </Sheet>
    )
  }

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent align="end" className="w-56 p-0" side={dropUp ? 'top' : 'bottom'}>
        <LanguageCommand
          allLocales={allLocales}
          autoFocus
          disabled={isSavingLocale}
          locale={locale}
          noResults={t.language.noResults}
          onSelect={code => void selectLocale(code)}
          searchPlaceholder={t.language.searchPlaceholder}
        />
      </PopoverContent>
    </Popover>
  )
}

function LanguageCommand({
  allLocales,
  autoFocus,
  disabled,
  locale,
  noResults,
  onSelect,
  searchPlaceholder
}: LanguageCommandProps) {
  const [search, setSearch] = useState('')

  // Own the search term and filter manually. cmdk's built-in shouldFilter
  // reorders items by its fuzzy-match score (≈alphabetical with an empty
  // query), which destroys the curated en→zh→zh-hant→ja→ko order. We disable it
  // and do a plain substring filter that preserves array order — matching
  // model-picker.tsx. Match against the endonym, the (hidden) English name,
  // and the locale code so "한국"/"korean"/"ko" all find Korean.
  const q = search.trim().toLowerCase()

  const filtered = allLocales.filter(
    ([code, meta]) =>
      !q ||
      meta.name.toLowerCase().includes(q) ||
      meta.englishName.toLowerCase().includes(q) ||
      code.toLowerCase().includes(q)
  )

  return (
    <Command className="bg-transparent" shouldFilter={false}>
      <CommandInput autoFocus={autoFocus} onValueChange={setSearch} placeholder={searchPlaceholder} value={search} />
      <CommandList className="max-h-80 p-1">
        {filtered.length === 0 ? (
          <div className="py-6 text-center text-sm text-muted-foreground">{noResults}</div>
        ) : (
          filtered.map(([code, meta]) => {
            const selected = code === locale

            return (
              <CommandItem
                className={cn(selected ? 'font-medium text-foreground' : 'text-muted-foreground')}
                disabled={disabled}
                key={code}
                onSelect={() => onSelect(code)}
                value={code}
              >
                <Check className={cn('size-3.5 shrink-0 text-primary', !selected && 'invisible')} />
                <span className="min-w-0 flex-1 truncate">{meta.name}</span>
                <span className="font-mono text-[0.65rem] uppercase text-(--ui-text-tertiary)">{code}</span>
              </CommandItem>
            )
          })
        )}
      </CommandList>
    </Command>
  )
}
