import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: {
    general: 'General',
    access: 'Access',
    runtime: 'Runtime',
    limits: 'Runtime & Limits',
    browser: 'Browser & Logins',
    providers: 'Providers & Keys',
    safety: 'Safety & Approvals',
    invalid: 'This setting is not part of this agent. Choose a section below.',
    modelDescription: 'Which models this agent uses, and what happens when one fails.'
  },
  fr: {
    general: 'Général',
    access: 'Accès',
    runtime: 'Exécution',
    limits: 'Exécution et limites',
    browser: 'Navigateur et connexions',
    providers: 'Fournisseurs et clés',
    safety: 'Sécurité et approbations',
    invalid: 'Ce réglage ne fait pas partie de cet agent. Choisissez une section ci-dessous.',
    modelDescription: 'Les modèles utilisés par cet agent et les solutions de secours en cas d’échec.'
  },
  de: {
    general: 'Allgemein',
    access: 'Zugriff',
    runtime: 'Laufzeit',
    limits: 'Laufzeit & Limits',
    browser: 'Browser & Anmeldungen',
    providers: 'Anbieter & Schlüssel',
    safety: 'Sicherheit & Freigaben',
    invalid: 'Diese Einstellung gehört nicht zu diesem Agenten. Wähle unten einen Bereich.',
    modelDescription: 'Welche Modelle dieser Agent verwendet und was passiert, wenn eines ausfällt.'
  },
  es: {
    general: 'General',
    access: 'Acceso',
    runtime: 'Ejecución',
    limits: 'Ejecución y límites',
    browser: 'Navegador e inicios de sesión',
    providers: 'Proveedores y claves',
    safety: 'Seguridad y aprobaciones',
    invalid: 'Este ajuste no pertenece a este agente. Elige una sección a continuación.',
    modelDescription: 'Qué modelos usa este agente y qué ocurre cuando uno falla.'
  },
  ja: {
    general: '一般',
    access: 'アクセス',
    runtime: '実行環境',
    limits: '実行環境と制限',
    browser: 'ブラウザーとログイン',
    providers: 'プロバイダーとキー',
    safety: '安全性と承認',
    invalid: 'この設定はこのエージェントの設定ではありません。下のセクションを選択してください。',
    modelDescription: 'このエージェントが使用するモデルと、失敗した場合の動作。'
  },
  zh: {
    general: '常规',
    access: '访问',
    runtime: '运行环境',
    limits: '运行环境与限制',
    browser: '浏览器与登录',
    providers: '提供商与密钥',
    safety: '安全与审批',
    invalid: '此设置不属于此智能体。请选择下方的设置分区。',
    modelDescription: '此智能体使用的模型，以及模型失败时的处理方式。'
  },
  'zh-hant': {
    general: '一般',
    access: '存取',
    runtime: '執行環境',
    limits: '執行環境與限制',
    browser: '瀏覽器與登入',
    providers: '供應商與金鑰',
    safety: '安全與核准',
    invalid: '此設定不屬於此代理。請選擇下方的設定區段。',
    modelDescription: '此代理使用的模型，以及模型失敗時的處理方式。'
  },
  ru: {
    general: 'Общие',
    access: 'Доступ',
    runtime: 'Среда выполнения',
    limits: 'Среда выполнения и ограничения',
    browser: 'Браузер и вход',
    providers: 'Провайдеры и ключи',
    safety: 'Безопасность и разрешения',
    invalid: 'Эта настройка не относится к этому агенту. Выберите раздел ниже.',
    modelDescription: 'Какие модели использует агент и что происходит при сбое одной из них.'
  },
  ar: {
    general: 'عام',
    access: 'الوصول',
    runtime: 'بيئة التشغيل',
    limits: 'بيئة التشغيل والحدود',
    browser: 'المتصفح وتسجيل الدخول',
    providers: 'المزوّدون والمفاتيح',
    safety: 'الأمان والموافقات',
    invalid: 'هذا الإعداد ليس جزءًا من هذا الوكيل. اختر قسمًا أدناه.',
    modelDescription: 'النماذج التي يستخدمها هذا الوكيل وما يحدث عند فشل أحدها.'
  }
} satisfies Record<BundledLocale, Record<string, string>>

const duplicate = {
  en: 'Duplicate',
  fr: 'Dupliquer',
  de: 'Duplizieren',
  es: 'Duplicar',
  ja: '複製',
  zh: '复制',
  'zh-hant': '複製',
  ru: 'Дублировать',
  ar: 'تكرار'
} satisfies Record<BundledLocale, string>

export function useAgentSettingsEditorCopy() {
  const { locale } = useI18n()
  const bundled = isBundledLocale(locale) ? locale : 'en'

  return { ...copy[bundled], duplicate: duplicate[bundled] }
}
