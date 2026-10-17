import { atom } from 'nanostores'

/**
 * Composer spellcheck preferences (#48375).
 *
 * Spellcheck is OFF by default: it flags code, file paths, and slash commands,
 * which is exactly what #44415 disabled it to avoid. Users opt in to get red
 * underlines and right-click suggestions. autoCorrect / autoCapitalize stay OFF
 * regardless — nothing is silently altered, only flagged.
 *
 * `language` is a BCP-47 tag (e.g. "en-US", "de-DE"). Empty = the system
 * locale. It is bridged to Electron's session dictionary (not just the
 * renderer `lang` attribute) because on Windows/Linux Chromium only returns
 * right-click suggestions when the session dictionary for that language is
 * seeded — the lang attribute alone is not enough.
 */
export const $composerSpellcheck = atom(false)
export const $composerSpellcheckLanguage = atom('')

export function normalizeSpellcheckLanguage(value: unknown): string {
  if (typeof value !== 'string') return ''
  const trimmed = value.trim()
  // Loose BCP-47 shape: language[-script][-region] (e.g. en, en-US, zh-Hant-TW).
  return /^[A-Za-z]{2,8}(-[A-Za-z0-9]{2,8})*$/.test(trimmed) ? trimmed : ''
}

export function setComposerEditorFromConfig(value: unknown): void {
  const editor = value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
  $composerSpellcheck.set(editor.spellcheck === true)
  $composerSpellcheckLanguage.set(normalizeSpellcheckLanguage(editor.language))
}
