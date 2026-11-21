// Pick the Hunspell dictionaries Chromium's spellchecker should load.
//
// Chromium checks text against every dictionary in the list at once, so a
// polyglot user gets one red squiggle for words that are wrong in ALL of their
// languages instead of having to pick a single language.
//
// Sources, in order:
//   1. HERMES_DESKTOP_SPELLCHECK_LANGUAGES — explicit comma/space separated list
//      (e.g. "pt-BR,en-US,es"). When set it is the whole list; nothing else is
//      appended.
//   2. Otherwise the OS preferred languages (Electron's
//      app.getPreferredSystemLanguages(), which follows $LANGUAGE on Linux and
//      the language list in Settings on Windows), then the app locale.
//
// Requested tags are matched against the dictionaries Chromium can actually
// download: exact tag first, then the bare language ("es-MX" -> "es"), then the
// first regional variant of a bare language ("pt" -> "pt-BR"). Unknown tags are
// dropped. With no override and no match we keep the old en-US fallback.

export const SPELLCHECK_LANGUAGES_ENV = 'HERMES_DESKTOP_SPELLCHECK_LANGUAGES'

export interface SpellCheckerLanguageInputs {
  override?: string | null
  preferred?: readonly string[] | null
  locale?: string | null
  available: readonly string[]
}

const FALLBACK = 'en-US'

function normalizeTag(raw: string): string {
  // POSIX locales look like "pt_BR.UTF-8" or "sr_RS@latin"; Chromium wants "pt-BR".
  return raw.trim().split(/[.@]/)[0].replace(/_/g, '-')
}

export function parseSpellCheckerLanguageList(value: string | null | undefined): string[] {
  if (!value) {
    return []
  }

  return value
    .split(/[\s,;:]+/)
    .map(normalizeTag)
    .filter(tag => tag.length > 0 && tag.toLowerCase() !== 'c' && tag.toLowerCase() !== 'posix')
}

export function matchSpellCheckerLanguage(tag: string, available: readonly string[]): string | null {
  const wanted = normalizeTag(tag).toLowerCase()

  if (!wanted) {
    return null
  }

  const byLower = new Map(available.map(lang => [lang.toLowerCase(), lang]))
  const exact = byLower.get(wanted)

  if (exact) {
    return exact
  }

  const base = wanted.split('-')[0]
  const bare = byLower.get(base)

  if (bare) {
    return bare
  }

  return available.find(lang => lang.toLowerCase().startsWith(`${base}-`)) ?? null
}

export function resolveSpellCheckerLanguages(inputs: SpellCheckerLanguageInputs): string[] {
  const overridden = parseSpellCheckerLanguageList(inputs.override)

  const requested =
    overridden.length > 0
      ? overridden
      : [...(inputs.preferred ?? []), ...(inputs.locale ? [inputs.locale] : [])].map(normalizeTag)

  const chosen: string[] = []

  const baseOf = (lang: string): string => lang.toLowerCase().split('-')[0]

  for (const tag of requested) {
    // OS lists pad regional tags with their bare language ("pt-BR", "pt", ...).
    // A bare tag adds nothing once a dictionary for that language is loaded.
    if (!tag.includes('-') && chosen.some(lang => baseOf(lang) === baseOf(tag))) {
      continue
    }

    const match = matchSpellCheckerLanguage(tag, inputs.available)

    if (match && !chosen.includes(match)) {
      chosen.push(match)
    }
  }

  if (chosen.length > 0) {
    return chosen
  }

  return [matchSpellCheckerLanguage(FALLBACK, inputs.available) ?? FALLBACK]
}
