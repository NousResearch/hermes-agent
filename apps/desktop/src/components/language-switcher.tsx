import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { ScrollArea } from '@/components/ui/scroll-area'
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

interface LanguageSwitcherOptionsProps {
  allLocales: Array<[Locale, (typeof LOCALE_META)[Locale]]>
  disabled?: boolean
  label: string
  locale: Locale
  onSelect: (code: Locale) => void
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
      aria-label={title}
      aria-expanded={open}
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
        {!collapsed && <span className="truncate">{locale === 'en' ? 'EN' : current.name}</span>}
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
          <ScrollArea className="max-h-80 px-2 pb-3">
            <LanguageSwitcherOptions
              allLocales={allLocales}
              disabled={isSavingLocale}
              label={title}
              locale={locale}
              onSelect={code => void selectLocale(code)}
            />
          </ScrollArea>
        </SheetContent>
      </Sheet>
    )
  }

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent align="end" className="w-48 p-1" side={dropUp ? 'top' : 'bottom'}>
        <ScrollArea className="max-h-80">
          <LanguageSwitcherOptions
            allLocales={allLocales}
            disabled={isSavingLocale}
            label={title}
            locale={locale}
            onSelect={code => void selectLocale(code)}
          />
        </ScrollArea>
      </PopoverContent>
    </Popover>
  )
}

function LanguageSwitcherOptions({ allLocales, disabled, label, locale, onSelect }: LanguageSwitcherOptionsProps) {
  return (
    <div aria-label={label} className="py-1" role="listbox">
      {allLocales.map(([code, meta]) => {
        const selected = code === locale

        return (
          <button
            aria-selected={selected}
            className={cn(
              'flex w-full cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[length:var(--conversation-caption-font-size)] transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:pointer-events-none disabled:opacity-50',
              selected ? 'font-medium text-foreground' : 'text-muted-foreground'
            )}
            disabled={disabled}
            key={code}
            onClick={() => onSelect(code)}
            role="option"
            type="button"
          >
            <span className="min-w-0 flex-1 truncate">{meta.name}</span>
            <span className="font-mono text-[0.65rem] uppercase text-(--ui-text-tertiary)">{code}</span>
            {selected && <Check className="size-3.5 shrink-0 text-primary" />}
          </button>
        )
      })}
    </div>
  )
}
