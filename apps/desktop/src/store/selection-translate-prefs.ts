import { atom } from 'nanostores'

import { normalizeTranslationLanguageCode, type SelectionLanguageCode } from '@/lib/selection-language'

const STORAGE_KEY = 'hermes.desktop.selection-translate.target.v1'
const LEGACY_MODE_STORAGE_KEY = 'hermes.desktop.selection-translate.mode'

export function resolveStoredTranslationTarget(
  storedTarget: string | null,
  legacyMode: string | null,
  browserLanguage: string
): SelectionLanguageCode {
  const target = storedTarget ? normalizeTranslationLanguageCode(storedTarget) : null

  if (target) {
    return target
  }

  if (legacyMode === 'auto') {
    return 'ar'
  }

  const legacyTarget = legacyMode ? normalizeTranslationLanguageCode(legacyMode) : null

  if (legacyTarget) {
    return legacyTarget
  }

  const browserTarget = normalizeTranslationLanguageCode(browserLanguage)

  if (browserTarget) {
    return browserTarget
  }

  return 'en'
}

function loadPreferredTarget(): SelectionLanguageCode {
  if (typeof window === 'undefined') {
    return 'en'
  }

  try {
    return resolveStoredTranslationTarget(
      window.localStorage.getItem(STORAGE_KEY),
      window.localStorage.getItem(LEGACY_MODE_STORAGE_KEY),
      window.navigator.language
    )
  } catch {
    // ignore quota / private mode
  }

  return 'en'
}

export const $selectionTranslatePreferredTarget = atom<SelectionLanguageCode>(loadPreferredTarget())

export function setSelectionTranslatePreferredTarget(target: string) {
  const canonical = normalizeTranslationLanguageCode(target)

  if (!canonical) {
    return
  }

  $selectionTranslatePreferredTarget.set(canonical)

  try {
    window.localStorage.setItem(STORAGE_KEY, canonical)
  } catch {
    // ignore
  }
}
