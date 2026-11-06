import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: {
    newSessionIn: 'New session in',
    blank: 'Start blank session',
    blankHint: 'Uses the default agent on this gateway.'
  },
  fr: {
    newSessionIn: 'Nouvelle session avec',
    blank: 'Démarrer une session vierge',
    blankHint: 'Utilise l’agent par défaut de cette passerelle.'
  },
  de: {
    newSessionIn: 'Neue Sitzung mit',
    blank: 'Leere Sitzung starten',
    blankHint: 'Verwendet den Standardagenten dieses Gateways.'
  },
  es: {
    newSessionIn: 'Nueva sesión con',
    blank: 'Iniciar sesión en blanco',
    blankHint: 'Usa el agente predeterminado de esta puerta de enlace.'
  },
  ja: {
    newSessionIn: '新しいセッションのエージェント',
    blank: '空のセッションを開始',
    blankHint: 'このゲートウェイのデフォルトエージェントを使用します。'
  },
  zh: {
    newSessionIn: '新建会话使用',
    blank: '开始空白会话',
    blankHint: '使用此网关的默认智能体。'
  },
  'zh-hant': {
    newSessionIn: '新建工作階段使用',
    blank: '開始空白工作階段',
    blankHint: '使用此閘道的預設代理。'
  },
  ru: {
    newSessionIn: 'Новый сеанс с',
    blank: 'Начать пустой сеанс',
    blankHint: 'Используется агент по умолчанию на этом шлюзе.'
  },
  ar: {
    newSessionIn: 'جلسة جديدة مع',
    blank: 'بدء جلسة فارغة',
    blankHint: 'يستخدم الوكيل الافتراضي على هذه البوابة.'
  }
} satisfies Record<BundledLocale, { newSessionIn: string; blank: string; blankHint: string }>

export function useNewSessionCopy() {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
