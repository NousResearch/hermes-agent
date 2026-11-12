import type { BundledLocale } from './types'

export const desktopTerms: Record<BundledLocale, { agents: string; machines: string; routines: string }> = {
  en: { agents: 'Agents', machines: 'Machines', routines: 'Routines' },
  fr: { agents: 'Agents', machines: 'Machines', routines: 'Routines' },
  de: { agents: 'Agenten', machines: 'Rechner', routines: 'Routinen' },
  es: { agents: 'Agentes', machines: 'Máquinas', routines: 'Rutinas' },
  ja: { agents: 'エージェント', machines: 'マシン', routines: 'ルーティン' },
  zh: { agents: '智能体', machines: '机器', routines: '例行任务' },
  'zh-hant': { agents: '智慧體', machines: '機器', routines: '例行任務' },
  ru: { agents: 'Агенты', machines: 'Компьютеры', routines: 'Регулярные задачи' },
  ar: { agents: 'الوكلاء', machines: 'الأجهزة', routines: 'المهام الروتينية' }
}
