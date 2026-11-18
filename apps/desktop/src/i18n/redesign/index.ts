/**
 * The redesigned shell's wording layer. Classic renders the base catalog exactly as `main` ships
 * it; the redesign renders the base catalog with this overlay merged on top (Sessions → Chats,
 * Artifacts → Library, Profiles → Agents, Cron → Routines, Gateways → Machines). New redesign
 * wording goes in `./<locale>.ts`, never in the base catalog.
 *
 * A locale inherits the English overlay wherever its base still shows the English string (a sparse
 * locale like `ar`, or a pack-only language falling back to English), so an untranslated key reads
 * the redesign's English rather than classic's. Keys the locale translates take only its own
 * overlay.
 */

import { isRecord, mergeTranslations } from '@hermes/shared/i18n'

import { isBundledLocale, TRANSLATIONS } from '../catalog'
import type { BundledLocale, Locale, Translations } from '../types'

import { arRedesign } from './ar'
import { deRedesign } from './de'
import { enRedesign } from './en'
import { esRedesign } from './es'
import { frRedesign } from './fr'
import { jaRedesign } from './ja'
import { ruRedesign } from './ru'
import type { RedesignOverlay } from './types'
import { zhRedesign } from './zh'
import { zhHantRedesign } from './zh-hant'

export type { RedesignOverlay } from './types'

export const REDESIGN_OVERLAYS: Record<BundledLocale, RedesignOverlay> = {
  en: enRedesign,
  zh: zhRedesign,
  'zh-hant': zhHantRedesign,
  ja: jaRedesign,
  ar: arRedesign,
  ru: ruRedesign,
  fr: frRedesign,
  de: deRedesign,
  es: esRedesign
}

/** The slice of `overlay` whose keys `base` still shows exactly as `english` does. */
function untranslatedSlice(overlay: unknown, english: unknown, base: unknown): unknown {
  if (!isRecord(overlay)) {
    return base === english ? overlay : undefined
  }

  const slice: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(overlay)) {
    const picked = untranslatedSlice(
      value,
      isRecord(english) ? english[key] : undefined,
      isRecord(base) ? base[key] : undefined
    )

    if (picked !== undefined) {
      slice[key] = picked
    }
  }

  return Object.keys(slice).length ? slice : undefined
}

const resolved = new Map<Locale, Translations>()

/** `base` (the bundled catalog for `locale`, or English) in the redesign's wording. Memoized per
 *  locale: both inputs are compiled in. */
export function redesignCatalog(locale: Locale, base: Translations): Translations {
  const cached = resolved.get(locale)

  if (cached) {
    return cached
  }

  const english = locale === 'en' ? enRedesign : untranslatedSlice(enRedesign, TRANSLATIONS.en, base)
  const own = locale !== 'en' && isBundledLocale(locale) ? REDESIGN_OVERLAYS[locale] : undefined

  const value = mergeTranslations<Translations>(
    mergeTranslations<Translations>(base, english as RedesignOverlay | undefined),
    own
  )

  resolved.set(locale, value)

  return value
}
