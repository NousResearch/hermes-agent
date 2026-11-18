import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: {
    newSessionIn: 'New chat in',
    blank: 'Start blank chat',
    blankHint: 'Uses the default agent on this machine.'
  },
  fr: {
    newSessionIn: 'Nouvelle discussion avec',
    blank: 'Démarrer une discussion vierge',
    blankHint: 'Utilise l’agent par défaut de cette machine.'
  },
  de: {
    newSessionIn: 'Neuer Chat mit',
    blank: 'Leeren Chat starten',
    blankHint: 'Verwendet den Standardagenten dieses Rechners.'
  },
  es: {
    newSessionIn: 'Nuevo chat en',
    blank: 'Iniciar chat en blanco',
    blankHint: 'Usa el agente predeterminado de esta máquina.'
  },
  ja: {
    newSessionIn: '新しいチャットの開始先',
    blank: '空のチャットを開始',
    blankHint: 'このマシンのデフォルトエージェントを使用します。'
  },
  zh: {
    newSessionIn: '新建对话位于',
    blank: '开始空白对话',
    blankHint: '使用此机器的默认智能体。'
  },
  'zh-hant': {
    newSessionIn: '新對話使用',
    blank: '開始空白對話',
    blankHint: '使用此機器的預設智慧體。'
  },
  ru: {
    newSessionIn: 'Новый чат в',
    blank: 'Начать пустой чат',
    blankHint: 'Используется агент по умолчанию на этом компьютере.'
  },
  ar: {
    newSessionIn: 'محادثة جديدة مع',
    blank: 'بدء محادثة فارغة',
    blankHint: 'يستخدم الوكيل الافتراضي على هذا الجهاز.'
  }
} satisfies Record<BundledLocale, { newSessionIn: string; blank: string; blankHint: string }>

export function useNewSessionCopy() {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
