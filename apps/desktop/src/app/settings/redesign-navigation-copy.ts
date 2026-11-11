import { type BundledLocale, isBundledLocale } from '@/i18n'
import { desktopTerms } from '@/i18n/desktop-terms'

/** Destination names of the redesigned global Settings rail (Figma 856:1834). Classic keeps its own. */
export interface RedesignNavigationCopy {
  accountPlan: string
  gateways: string
  sessions: string
  about: string
  agentSettings: string
  agentSettingsDescription: string
}

type LocalCopy = Omit<RedesignNavigationCopy, 'gateways'> & { gateways: (machines: string) => string }

const COPY: Record<BundledLocale, LocalCopy> = {
  en: {
    accountPlan: 'Account & Plan',
    gateways: machines => `Gateways & ${machines}`,
    sessions: 'Sessions & History',
    about: 'About & Updates',
    agentSettings: 'Agent settings',
    agentSettingsDescription: 'Model, safety, memory and more, per agent'
  },
  fr: {
    accountPlan: 'Compte et forfait',
    gateways: machines => `Passerelles et ${machines.toLocaleLowerCase('fr')}`,
    sessions: 'Sessions et historique',
    about: 'À propos et mises à jour',
    agentSettings: 'Réglages de l’agent',
    agentSettingsDescription: 'Modèle, sécurité, mémoire et plus, par agent'
  },
  de: {
    accountPlan: 'Konto & Tarif',
    gateways: machines => `Gateways & ${machines}`,
    sessions: 'Sitzungen & Verlauf',
    about: 'Info & Updates',
    agentSettings: 'Agent-Einstellungen',
    agentSettingsDescription: 'Modell, Sicherheit, Gedächtnis und mehr, je Agent'
  },
  es: {
    accountPlan: 'Cuenta y plan',
    gateways: machines => `Gateways y ${machines.toLocaleLowerCase('es')}`,
    sessions: 'Sesiones e historial',
    about: 'Acerca de y actualizaciones',
    agentSettings: 'Ajustes del agente',
    agentSettingsDescription: 'Modelo, seguridad, memoria y más, por agente'
  },
  ja: {
    accountPlan: 'アカウントとプラン',
    gateways: machines => `ゲートウェイと${machines}`,
    sessions: 'セッションと履歴',
    about: '情報とアップデート',
    agentSettings: 'エージェント設定',
    agentSettingsDescription: 'モデル、安全性、メモリーなどをエージェントごとに'
  },
  zh: {
    accountPlan: '账户与套餐',
    gateways: machines => `网关与${machines}`,
    sessions: '会话与历史',
    about: '关于与更新',
    agentSettings: '智能体设置',
    agentSettingsDescription: '模型、安全、记忆等，按智能体设置'
  },
  'zh-hant': {
    accountPlan: '帳戶與方案',
    gateways: machines => `閘道與${machines}`,
    sessions: '工作階段與記錄',
    about: '關於與更新',
    agentSettings: '智慧體設定',
    agentSettingsDescription: '模型、安全、記憶等，依智慧體設定'
  },
  ru: {
    accountPlan: 'Аккаунт и тариф',
    gateways: machines => `Шлюзы и ${machines.toLocaleLowerCase('ru')}`,
    sessions: 'Сеансы и история',
    about: 'О программе и обновления',
    agentSettings: 'Настройки агента',
    agentSettingsDescription: 'Модель, безопасность, память и другое для каждого агента'
  },
  ar: {
    accountPlan: 'الحساب والخطة',
    gateways: machines => `البوابات و${machines}`,
    sessions: 'الجلسات والسجل',
    about: 'حول والتحديثات',
    agentSettings: 'إعدادات الوكيل',
    agentSettingsDescription: 'النموذج والأمان والذاكرة والمزيد، لكل وكيل'
  }
}

export function redesignNavigationCopy(locale: string): RedesignNavigationCopy {
  const bundled = isBundledLocale(locale) ? locale : 'en'
  const copy = COPY[bundled]

  return { ...copy, gateways: copy.gateways(desktopTerms[bundled].machines) }
}
