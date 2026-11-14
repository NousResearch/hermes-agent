import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesign's keyboard-operable pane seam: its accessible name. Classic draws no focusable
 *  seam (as on main), so this stays beside the one call site instead of in the base catalog. */
const RESIZE_LABEL: Record<BundledLocale, string> = {
  en: 'Resize panes',
  zh: '调整窗格大小',
  'zh-hant': '調整窗格大小',
  ja: 'ペインのサイズを変更',
  ar: 'تغيير حجم الأجزاء',
  ru: 'Изменить размер панелей',
  fr: 'Redimensionner les panneaux',
  de: 'Bereiche anpassen',
  es: 'Cambiar tamaño de los paneles'
}

export function useSashLabel(): string {
  const { locale } = useI18n()

  return RESIZE_LABEL[isBundledLocale(locale) ? locale : 'en']
}
