import { Codicon } from '@/components/ui/codicon'
import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: ['Pin your go-to chats.', 'Drop a chat here to pin it.'],
  de: ['Wichtige Chats anpinnen.', 'Chat hierher ziehen zum Anpinnen.'],
  es: ['Fija tus chats habituales.', 'Arrastra un chat aquí para fijarlo.'],
  fr: ['Épinglez vos conversations favorites.', 'Glissez une conversation ici pour l’épingler.'],
  ja: ['よく使うチャットをピン留め。', 'チャットをここにドロップしてピン留め。'],
  zh: ['置顶常用聊天。', '将聊天拖到此处即可置顶。'],
  'zh-hant': ['釘選常用聊天。', '將聊天拖到此處即可釘選。'],
  ru: ['Закрепите нужные чаты.', 'Перетащите чат сюда, чтобы закрепить.'],
  ar: ['ثبّت محادثاتك المفضلة.', 'أسقط محادثة هنا لتثبيتها.']
} satisfies Record<BundledLocale, [string, string]>

/** What an empty Pinned shows while a session drag is in flight (HADESK-132): the section stands
 *  down at rest, so this hint comes out with its drop target and goes away with it. */
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
