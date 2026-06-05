export const DEFAULT_DESKTOP_LANGUAGE = 'en'

export const DESKTOP_LANGUAGES = [
  {
    id: 'en',
    label: 'English',
    nativeLabel: 'English',
    configValue: 'en',
    translationKey: 'settings.appearance.language.english'
  },
  {
    id: 'zh',
    label: 'Simplified Chinese',
    nativeLabel: '简体中文',
    configValue: 'zh',
    translationKey: 'settings.appearance.language.simplifiedChinese'
  }
] as const

export type DesktopLanguage = (typeof DESKTOP_LANGUAGES)[number]['id']

const LANGUAGE_ALIASES: Record<string, DesktopLanguage> = {
  en: 'en',
  'en-us': 'en',
  'en_us': 'en',
  zh: 'zh',
  'zh-cn': 'zh',
  'zh_cn': 'zh',
  'zh-hans': 'zh',
  'zh_hans': 'zh',
  'zh-hans-cn': 'zh',
  'zh_hans_cn': 'zh'
}

export function isDesktopLanguage(value: unknown): value is DesktopLanguage {
  return typeof value === 'string' && DESKTOP_LANGUAGES.some(language => language.id === value)
}

export function normalizeDesktopLanguage(value: unknown): DesktopLanguage {
  if (typeof value !== 'string') {
    return DEFAULT_DESKTOP_LANGUAGE
  }

  return LANGUAGE_ALIASES[value.trim().toLowerCase()] ?? DEFAULT_DESKTOP_LANGUAGE
}

export function isSupportedDesktopLanguageValue(value: unknown): boolean {
  return typeof value === 'string' && LANGUAGE_ALIASES[value.trim().toLowerCase()] != null
}

export function desktopLanguageConfigValue(language: DesktopLanguage): string {
  return DESKTOP_LANGUAGES.find(item => item.id === language)?.configValue ?? DEFAULT_DESKTOP_LANGUAGE
}
