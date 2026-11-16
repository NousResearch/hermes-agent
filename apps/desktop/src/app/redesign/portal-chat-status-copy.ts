import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: { line: 'Status line under the chat' },
  fr: { line: 'Ligne d’état sous la discussion' },
  de: { line: 'Statuszeile unter dem Chat' },
  es: { line: 'Línea de estado bajo el chat' },
  ja: { line: 'チャットの下にステータス行を表示' },
  zh: { line: '在对话下方显示状态行' },
  'zh-hant': { line: '在對話下方顯示狀態列' },
  ru: { line: 'Строка состояния под чатом' },
  ar: { line: 'سطر الحالة أسفل المحادثة' }
} satisfies Record<BundledLocale, { line: string }>

export function usePortalChatStatusCopy() {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
