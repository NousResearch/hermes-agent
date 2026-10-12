import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandItemCheck,
  CommandList
} from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { useIsMobile } from '@/hooks/use-mobile'
import { $appLocaleVersion, type LanguageOption, languageOptions, type Locale, localeMeta, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { ChevronDown, Globe } from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

export interface LanguageSwitcherProps {
  className?: string
  collapsed?: boolean
  dropUp?: boolean
}

interface LanguageCommandProps {
  allLocales: LanguageOption[]
  autoFocus?: boolean
  disabled?: boolean
  locale: Locale
  noResults: string
  onSelect: (code: Locale) => void
  searchPlaceholder: string
  /** `menu` inside the desktop popover; the mobile sheet keeps the palette rows. */
  variant?: 'default' | 'menu'
}

export function LanguageSwitcher({ className, collapsed = false, dropUp = false }: LanguageSwitcherProps) {
  const { isSavingLocale, locale, setLocale, t } = useI18n()
  const [open, setOpen] = useState(false)
  const isMobile = useIsMobile()
  const useMobileSheet = Boolean(dropUp && isMobile)
  // Bundled ∪ plugin-registered ∪ backend `i18n.languages`; the registry
  // version re-lists when a pack lands after first paint.
  const registryVersion = useStore($appLocaleVersion)
  // eslint-disable-next-line react-hooks/exhaustive-deps -- registryVersion is the registry's change token
  const allLocales = useMemo(() => languageOptions(), [registryVersion])
  const current = allLocales.find(option => option.id === locale) ?? { id: locale, ...localeMeta(locale) }
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
      type="button"
      variant="outline"
    >
      <span className="inline-flex min-w-0 items-center gap-2">
        <Globe className="size-3.5 shrink-0" />
        {!collapsed && <span className="truncate">{current.endonym}</span>}
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
      <PopoverContent align="end" className="w-56" side={dropUp ? 'top' : 'bottom'} variant="menu">
        <LanguageCommand
          allLocales={allLocales}
          autoFocus
          disabled={isSavingLocale}
          locale={locale}
          noResults={t.language.noResults}
          onSelect={code => void selectLocale(code)}
          searchPlaceholder={t.language.searchPlaceholder}
          variant="menu"
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
  searchPlaceholder,
  variant = 'default'
}: LanguageCommandProps) {
  const [search, setSearch] = useState('')

  // Own the search term and filter manually. cmdk's built-in shouldFilter
  // reorders items by its fuzzy-match score (≈alphabetical with an empty
  // query), which destroys the curated en→zh→zh-hant→ja order. We disable it
  // and do a plain substring filter that preserves array order — matching
  // model-picker.tsx. Match against the endonym, the (hidden) English name,
  // and the locale code so "日本"/"japanese"/"ja" all find Japanese.
  const q = normalize(search)

  const filtered = allLocales.filter(
    option =>
      !q ||
      option.endonym.toLowerCase().includes(q) ||
      (option.englishName?.toLowerCase().includes(q) ?? false) ||
      option.id.toLowerCase().includes(q)
  )

  return (
    <Command className="bg-transparent" shouldFilter={false} variant={variant}>
      <CommandInput autoFocus={autoFocus} onValueChange={setSearch} placeholder={searchPlaceholder} value={search} />
      <CommandList className={variant === 'menu' ? undefined : 'max-h-80 p-1'}>
        <CommandEmpty>{noResults}</CommandEmpty>
        {filtered.map(option => {
          const selected = option.id === locale

          return (
            <CommandItem
              className={cn(selected && 'font-medium')}
              disabled={disabled}
              key={option.id}
              onSelect={() => onSelect(option.id)}
              value={option.id}
            >
              <span className="min-w-0 flex-1 truncate">{option.endonym}</span>
              <span className="font-mono text-[0.65rem] uppercase text-(--ui-text-tertiary)">{option.id}</span>
              <CommandItemCheck checked={selected} />
            </CommandItem>
          )
        })}
      </CommandList>
    </Command>
  )
}
