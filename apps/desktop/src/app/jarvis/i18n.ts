import type { Translations } from '@/i18n'

/**
 * The one navigation of the product: the shell's rail owns every destination,
 * the runtime's sidebar keeps only the conversations (see `$jarvisShellNav`).
 * Grouped so a longer rail still reads at a glance; every entry opens a real
 * runtime page (see `JARVIS_VIEW_TARGETS`).
 */
export const JARVIS_NAV_GROUPS = [
  { id: 'work', views: ['jarvis', 'tasks', 'agents'] },
  { id: 'knowledge', views: ['starmap', 'prompts', 'artifacts', 'memory'] },
  { id: 'connections', views: ['connections', 'webhooks'] },
  { id: 'system', views: ['tools', 'insights', 'settings'] }
] as const

export const JARVIS_MAIN_VIEWS = JARVIS_NAV_GROUPS.flatMap(group => group.views)
export const JARVIS_AUXILIARY_VIEWS = ['profile', 'messaging'] as const

export type JarvisNavGroup = (typeof JARVIS_NAV_GROUPS)[number]['id']
export type JarvisMainView = (typeof JARVIS_NAV_GROUPS)[number]['views'][number]
export type JarvisAuxiliaryView = (typeof JARVIS_AUXILIARY_VIEWS)[number]
export type JarvisShellView = JarvisMainView | JarvisAuxiliaryView

export type JarvisShellCopy = Translations['jarvisShell']
