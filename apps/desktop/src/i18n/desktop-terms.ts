import type { BundledLocale } from './types'

export const desktopTerms: Record<BundledLocale, { agents: string; schedules: string }> = {
  en: { agents: 'Agents', schedules: 'Schedules' },
  fr: { agents: 'Agents', schedules: 'Planifications' },
  de: { agents: 'Agenten', schedules: 'Zeitpläne' },
  es: { agents: 'Agentes', schedules: 'Programaciones' },
  ja: { agents: 'エージェント', schedules: 'スケジュール' },
  zh: { agents: '智能体', schedules: '计划' },
  'zh-hant': { agents: '智慧體', schedules: '排程' },
  ru: { agents: 'Агенты', schedules: 'Расписания' },
  ar: { agents: 'الوكلاء', schedules: 'الجداول' }
}
