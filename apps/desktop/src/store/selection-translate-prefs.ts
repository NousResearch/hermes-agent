import { atom } from 'nanostores'

import type { SelectionTranslateMode } from '@/lib/selection-language'

const STORAGE_KEY = 'hermes.desktop.selection-translate.mode'

export const $selectionTranslateMode = atom<SelectionTranslateMode>(loadMode())

function loadMode(): SelectionTranslateMode {
  if (typeof window === 'undefined') {
    return 'auto'
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)

    if (raw === 'auto' || raw === 'ar' || raw === 'en') {
      return raw
    }
  } catch {
    // ignore quota / private mode
  }

  return 'auto'
}

export function setSelectionTranslateMode(mode: SelectionTranslateMode) {
  $selectionTranslateMode.set(mode)

  try {
    window.localStorage.setItem(STORAGE_KEY, mode)
  } catch {
    // ignore
  }
}
