import { atom } from 'nanostores'

export type DesktopLanguage = 'zh' | 'en'

const LANGUAGE_STORAGE_KEY = 'metakina-agent-language'

function initialLanguage(): DesktopLanguage {
  try {
    const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY)
    if (stored === 'zh' || stored === 'en') {
      return stored
    }

    return window.navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en'
  } catch {
    return 'en'
  }
}

export const $desktopLanguage = atom<DesktopLanguage>(initialLanguage())

export function setDesktopLanguage(language: DesktopLanguage) {
  $desktopLanguage.set(language)
  try {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language)
    window.localStorage.setItem('hermes-locale', language)
  } catch {
    // Local storage is a convenience; restricted storage should not break settings.
  }
}
