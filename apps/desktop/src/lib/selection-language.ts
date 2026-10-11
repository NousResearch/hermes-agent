export type SelectionLanguageCode = string

export interface TranslationLanguageOption {
  code: SelectionLanguageCode
  name: string
}

const ENGLISH_LANGUAGE_NAMES = new Intl.DisplayNames(['en'], { fallback: 'none', type: 'language' })
const MAX_LANGUAGE_TAG_LENGTH = 64

export function normalizeTranslationLanguageCode(value: unknown): SelectionLanguageCode | null {
  if (typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim()

  if (!trimmed || trimmed.length > MAX_LANGUAGE_TAG_LENGTH) {
    return null
  }

  try {
    const locale = new Intl.Locale(trimmed)

    return locale.language && locale.language !== 'und' ? locale.baseName : null
  } catch {
    return null
  }
}

/**
 * Enumerate two-letter base-language suggestions known to Chromium's ICU data.
 * The picker also accepts validated BCP-47 tags for languages and variants that
 * cannot be enumerated by Intl, such as `haw`, `pt-BR`, and `zh-Hant`.
 */
export function listTranslationLanguages(locale: string): TranslationLanguageOption[] {
  const localizedNames = new Intl.DisplayNames([locale], { fallback: 'none', type: 'language' })
  const languages = new Map<SelectionLanguageCode, TranslationLanguageOption>()

  for (let first = 97; first <= 122; first += 1) {
    for (let second = 97; second <= 122; second += 1) {
      const candidate = String.fromCharCode(first, second)
      const canonical = Intl.getCanonicalLocales(candidate)[0]
      const englishName = ENGLISH_LANGUAGE_NAMES.of(canonical)

      if (!englishName || languages.has(canonical)) {
        continue
      }

      languages.set(canonical, {
        code: canonical,
        name: localizedNames.of(canonical) ?? englishName
      })
    }
  }

  return [...languages.values()].sort((left, right) => left.name.localeCompare(right.name, locale))
}

export function languageLabel(code: SelectionLanguageCode, locale = 'en'): string {
  const canonical = normalizeTranslationLanguageCode(code)

  const names =
    locale === 'en' ? ENGLISH_LANGUAGE_NAMES : new Intl.DisplayNames([locale], { fallback: 'none', type: 'language' })

  return canonical ? (names.of(canonical) ?? canonical) : code
}
