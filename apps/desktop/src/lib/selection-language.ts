/** Dominant-script heuristic for selection translation direction. */
export type SelectionLanguageCode = 'ar' | 'en'
export type SelectionTranslateMode = 'auto' | SelectionLanguageCode

export function countScriptChars(text: string, script: 'Arabic' | 'Latin'): number {
  const re = script === 'Arabic' ? /\p{Script=Arabic}/gu : /\p{Script=Latin}/gu

  return [...text.matchAll(re)].length
}

/** English-dominant (or neither) → Arabic; Arabic-dominant → English. */
export function detectSelectionLanguage(text: string): SelectionLanguageCode {
  const arabic = countScriptChars(text, 'Arabic')
  const latin = countScriptChars(text, 'Latin')

  return arabic > latin ? 'ar' : 'en'
}

export function resolveTranslateTarget(
  text: string,
  mode: SelectionTranslateMode = 'auto'
): SelectionLanguageCode {
  if (mode === 'ar' || mode === 'en') {
    return mode
  }

  return detectSelectionLanguage(text) === 'ar' ? 'en' : 'ar'
}

export function languageLabel(code: SelectionLanguageCode): string {
  return code === 'ar' ? 'Arabic' : 'English'
}
