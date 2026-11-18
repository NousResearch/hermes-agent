import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

interface StatusBarCopy {
  collapse: string
  open: string
}

const copy = {
  en: { collapse: 'Collapse status bar', open: 'Open status bar' },
  fr: { collapse: 'Réduire la barre d’état', open: 'Ouvrir la barre d’état' },
  de: { collapse: 'Statusleiste einklappen', open: 'Statusleiste öffnen' },
  es: { collapse: 'Contraer la barra de estado', open: 'Abrir la barra de estado' },
  ja: { collapse: 'ステータスバーを折りたたむ', open: 'ステータスバーを開く' },
  zh: { collapse: '收起状态栏', open: '打开状态栏' },
  'zh-hant': { collapse: '收合狀態列', open: '開啟狀態列' },
  ru: { collapse: 'Свернуть строку состояния', open: 'Открыть строку состояния' },
  ar: { collapse: 'طي شريط الحالة', open: 'فتح شريط الحالة' }
} satisfies Record<BundledLocale, StatusBarCopy>

export function usePortalStatusBarCopy(): StatusBarCopy {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
