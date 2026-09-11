import { type Locale, LOCALE_META, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

export interface LanguageSwitcherProps {
  className?: string
  collapsed?: boolean
}

/**
 * Language picker — a native <select>.
 *
 * 2026-09-02: rewritten from a Popover/Sheet + cmdk `Command` searchable list
 * (role="listbox"/"option" items reached via custom keyboard handling) to a
 * plain native <select>. Reported as inaccessible: Space, Ctrl+Up/Down, and
 * Alt+Up/Down either didn't move through options or moved focus to the
 * control itself without reading any option aloud. A native <select> hands
 * all of that (open, move, announce, choose) to the OS/browser's own
 * combobox implementation, which every screen reader already drives
 * correctly — no custom keyboard code needed. See the web dashboard's
 * `web/src/components/LanguageSwitcher.tsx` for the same fix applied there
 * first, and `D:\접근성\KWCAG_WCAG_조사.md` for why native controls are the
 * safer default for accessibility here.
 *
 * `setLocale` remains async — it optimistically updates the UI, then
 * persists to `config.yaml`'s `display.language` via the desktop config
 * client, reverting and surfacing `notifyError` on failure.
 */
export function LanguageSwitcher({ className, collapsed = false }: LanguageSwitcherProps) {
  const { isSavingLocale, locale, setLocale, t } = useI18n()
  const current = LOCALE_META[locale]
  const allLocales = Object.entries(LOCALE_META) as Array<[Locale, typeof current]>
  const title = t.language.switchTo

  const handleChange = async (code: Locale) => {
    if (code === locale || isSavingLocale) return

    triggerHaptic('selection')

    try {
      await setLocale(code)
      triggerHaptic('success')
    } catch (error) {
      notifyError(error, t.language.saveError)
    }
  }

  return (
    <select
      aria-label={title}
      className={cn(
        'min-w-32 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-2.5 py-1.5',
        'text-left text-muted-foreground hover:text-foreground disabled:opacity-60',
        collapsed && 'min-w-0 px-2',
        className,
      )}
      disabled={isSavingLocale}
      onChange={(e) => void handleChange(e.target.value as Locale)}
      title={title}
      value={locale}
    >
      {allLocales.map(([code, meta]) => (
        <option key={code} value={code}>
          {meta.name}
        </option>
      ))}
    </select>
  )
}
