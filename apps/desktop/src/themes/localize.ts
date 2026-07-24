import type { Translations } from '@/i18n'

type ThemeSummary = { name: string; label: string; description: string }
type BuiltInThemeName = keyof Translations['settings']['appearance']['builtInThemes']

const BUILT_IN_THEME_NAMES = new Set<BuiltInThemeName>(['nous', 'midnight', 'ember', 'mono', 'cyberpunk', 'slate'])

export function localizeTheme<T extends ThemeSummary>(
  theme: T,
  copy: Translations['settings']['appearance']['builtInThemes']
): T {
  if (!BUILT_IN_THEME_NAMES.has(theme.name as BuiltInThemeName)) {
    return theme
  }

  return { ...theme, ...copy[theme.name as BuiltInThemeName] }
}
