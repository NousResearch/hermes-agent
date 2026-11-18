import { Codicon } from '@/components/ui/codicon'
import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: ['Pin your go-to chats.', 'Use a chat’s menu to pin it.'],
  de: ['Wichtige Chats anpinnen.', 'Im Chat-Menü „Anpinnen“ wählen.'],
  es: ['Fija tus chats habituales.', 'Usa el menú del chat para fijarlo.'],
  fr: ['Épinglez vos conversations favorites.', 'Utilisez le menu de la conversation.'],
  ja: ['よく使うチャットをピン留め。', 'チャットのメニューからピン留めできます。'],
  zh: ['置顶常用聊天。', '在聊天菜单中选择置顶。'],
  'zh-hant': ['釘選常用聊天。', '在聊天選單中選擇釘選。'],
  ru: ['Закрепите нужные чаты.', 'Выберите «Закрепить» в меню чата.'],
  ar: ['ثبّت محادثاتك المفضلة.', 'استخدم قائمة المحادثة لتثبيتها.']
} satisfies Record<BundledLocale, [string, string]>

export function PortalPinnedEmpty() {
  const { locale } = useI18n()
  const [title, hint] = copy[isBundledLocale(locale) ? locale : 'en']

  return (
    <div className="portal-pinned-empty" data-slot="pinned-empty">
      <span aria-hidden className="portal-pinned-empty-icon"><Codicon name="pinned" size={16} /></span>
      <div><p>{title}</p><p className="text-(--ui-text-tertiary)">{hint}</p></div>
    </div>
  )
}
