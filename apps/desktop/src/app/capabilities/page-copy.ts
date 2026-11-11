import { type BundledLocale, isBundledLocale, useI18n } from '@/i18n'

/** The redesigned Capabilities heading's description (Figma 919:9095). */
const DESCRIPTION: Record<BundledLocale, string> = {
  en: 'Everything Hermes can use: what exists, what’s allowed, and what’s signed in. To turn something on or off for one agent, open that agent’s settings.',
  fr: 'Tout ce que Hermes peut utiliser : ce qui existe, ce qui est autorisé et ce qui est connecté. Pour activer ou désactiver un élément pour un agent, ouvrez les réglages de cet agent.',
  de: 'Alles, was Hermes nutzen kann: was es gibt, was erlaubt ist und wo du angemeldet bist. Um etwas für einen Agenten ein- oder auszuschalten, öffne dessen Einstellungen.',
  es: 'Todo lo que Hermes puede usar: lo que existe, lo que está permitido y dónde has iniciado sesión. Para activar o desactivar algo en un agente, abre los ajustes de ese agente.',
  ja: 'Hermes が使えるもの：存在するもの、許可されているもの、サインイン済みのもの。特定のエージェントでオン・オフするには、そのエージェントの設定を開いてください。',
  zh: 'Hermes 能使用的一切：现有的、已允许的和已登录的。要为某个智能体开启或关闭某项内容，请打开该智能体的设置。',
  'zh-hant': 'Hermes 能使用的一切：現有的、已允許的和已登入的。若要為某個代理開啟或關閉某項內容，請開啟該代理的設定。',
  ru: 'Всё, что может использовать Hermes: что есть, что разрешено и где выполнен вход. Чтобы включить или выключить что-то для одного агента, откройте настройки этого агента.',
  ar: 'كل ما يمكن لـ Hermes استخدامه: ما هو موجود، وما هو مسموح، وما تم تسجيل الدخول إليه. لتشغيل شيء أو إيقافه لوكيل واحد، افتح إعدادات ذلك الوكيل.'
}

export function useCapabilitiesPageDescription(): string {
  const { locale } = useI18n()

  return DESCRIPTION[isBundledLocale(locale) ? locale : 'en']
}
