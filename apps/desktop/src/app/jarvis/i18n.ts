import type { Translations } from '@/i18n'

export const JARVIS_MAIN_VIEWS = ['jarvis', 'tasks', 'memory', 'tools'] as const
export const JARVIS_AUXILIARY_VIEWS = ['settings', 'profile'] as const

export type JarvisMainView = (typeof JARVIS_MAIN_VIEWS)[number]
export type JarvisAuxiliaryView = (typeof JARVIS_AUXILIARY_VIEWS)[number]
export type JarvisShellView = JarvisMainView | JarvisAuxiliaryView

export type JarvisShellCopy = Translations['jarvisShell']
