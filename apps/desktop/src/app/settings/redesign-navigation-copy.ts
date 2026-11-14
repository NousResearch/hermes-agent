import { type BundledLocale, isBundledLocale } from '@/i18n'

/** Destination names of the redesigned global Settings rail (Figma 856:1834). Classic keeps its own. */
export interface RedesignNavigationCopy {
  accountPlan: string
  gateways: string
  sessions: string
  about: string
  agentSettings: string
  agentSettingsDescription: string
}

const COPY: Record<BundledLocale, RedesignNavigationCopy> = {
  en: {
    accountPlan: 'Account & Plan',
    gateways: 'Machines',
    sessions: 'Chats & History',
    about: 'About & Updates',
    agentSettings: 'Agent settings',
    agentSettingsDescription: 'Model, safety, memory and more, per agent'
  },
  fr: {
    accountPlan: 'Compte et forfait',
    gateways: 'Machines',
    sessions: 'Discussions et historique',
    about: 'À propos et mises à jour',
    agentSettings: 'Réglages de l’agent',
    agentSettingsDescription: 'Modèle, sécurité, mémoire et plus, par agent'
  },
  de: {
    accountPlan: 'Konto & Tarif',
    gateways: 'Rechner',
    sessions: 'Chats & Verlauf',
    about: 'Info & Updates',
    agentSettings: 'Agent-Einstellungen',
    agentSettingsDescription: 'Modell, Sicherheit, Gedächtnis und mehr, je Agent'
  },
  es: {
    accountPlan: 'Cuenta y plan',
    gateways: 'Máquinas',
    sessions: 'Chats e historial',
    about: 'Acerca de y actualizaciones',
    agentSettings: 'Ajustes del agente',
    agentSettingsDescription: 'Modelo, seguridad, memoria y más, por agente'
  },
  ja: {
    accountPlan: 'アカウントとプラン',
    gateways: 'マシン',
    sessions: 'チャットと履歴',
    about: '情報とアップデート',
    agentSettings: 'エージェント設定',
    agentSettingsDescription: 'モデル、安全性、メモリーなどをエージェントごとに'
  },
  zh: {
    accountPlan: '账户与套餐',
    gateways: '机器',
    sessions: '对话与历史',
    about: '关于与更新',
    agentSettings: '智能体设置',
    agentSettingsDescription: '模型、安全、记忆等，按智能体设置'
  },
  'zh-hant': {
    accountPlan: '帳戶與方案',
    gateways: '機器',
    sessions: '對話與記錄',
    about: '關於與更新',
    agentSettings: '智慧體設定',
    agentSettingsDescription: '模型、安全、記憶等，依智慧體設定'
  },
  ru: {
    accountPlan: 'Аккаунт и тариф',
    gateways: 'Компьютеры',
    sessions: 'Чаты и история',
    about: 'О программе и обновления',
    agentSettings: 'Настройки агента',
    agentSettingsDescription: 'Модель, безопасность, память и другое для каждого агента'
  },
  ar: {
    accountPlan: 'الحساب والخطة',
    gateways: 'الأجهزة',
    sessions: 'المحادثات والسجل',
    about: 'حول والتحديثات',
    agentSettings: 'إعدادات الوكيل',
    agentSettingsDescription: 'النموذج والأمان والذاكرة والمزيد، لكل وكيل'
  }
}

export function redesignNavigationCopy(locale: string): RedesignNavigationCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}
