import type { BundledLocale } from './types'

export const desktopTerms: Record<BundledLocale, { agents: string; machines: string; schedules: string }> = {
  en: { agents: 'Agents', machines: 'Machines', schedules: 'Schedules' },
  fr: { agents: 'Agents', machines: 'Machines', schedules: 'Planifications' },
  de: { agents: 'Agenten', machines: 'Rechner', schedules: 'Zeitpläne' },
  es: { agents: 'Agentes', machines: 'Máquinas', schedules: 'Programaciones' },
  ja: { agents: 'エージェント', machines: 'マシン', schedules: 'スケジュール' },
  zh: { agents: '智能体', machines: '机器', schedules: '计划' },
  'zh-hant': { agents: '智慧體', machines: '機器', schedules: '排程' },
  ru: { agents: 'Агенты', machines: 'Компьютеры', schedules: 'Расписания' },
  ar: { agents: 'الوكلاء', machines: 'الأجهزة', schedules: 'الجداول' }
}
