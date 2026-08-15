/**
 * Skill suggestions toggle (ghost suggestion + skill strip).
 *
 * Persisted per-profile in localStorage under `hermes.desktop.` so the
 * setting survives restarts. Default ON — the feature is a discoverability
 * aid and costs nothing until the user pauses typing.
 */
import { atom } from 'nanostores'

const KEY = 'hermes.desktop.skillSuggestions.enabled.v1'

function readInitial(): boolean {
  try {
    const raw = localStorage.getItem(KEY)
    // Absent key = default ON (feature ships enabled).
    return raw === null ? true : raw === '1'
  } catch {
    return true
  }
}

export const $skillSuggestionsEnabled = atom<boolean>(typeof window === 'undefined' ? true : readInitial())

export function setSkillSuggestionsEnabled(enabled: boolean): void {
  $skillSuggestionsEnabled.set(enabled)
  try {
    localStorage.setItem(KEY, enabled ? '1' : '0')
  } catch {
    // localStorage unavailable — setting lives for this session only.
  }
}
