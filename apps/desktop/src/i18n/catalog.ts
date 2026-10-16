import { ar } from './ar'
import { de } from './de'
import { en } from './en'
import { es } from './es'
import { fr } from './fr'
import { ja } from './ja'
import { ko } from './ko'
import { ru } from './ru'
import type { BundledLocale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

/** The catalogs compiled into the app. Runtime-registered languages (plugin
 *  packs, backend `.desktop.yaml` packs) are NOT here — resolve through
 *  `resolveTranslations()` in `./registry`, which layers them over these. */
export const TRANSLATIONS: Record<BundledLocale, Translations> = {
  en,
  zh,
  'zh-hant': zhHant,
  ja,
  ar,
  ru,
  fr,
  de,
  es,
  ko
}

export const BUNDLED_LOCALES = Object.keys(TRANSLATIONS) as readonly BundledLocale[]

export function isBundledLocale(value: unknown): value is BundledLocale {
  return typeof value === 'string' && Object.hasOwn(TRANSLATIONS, value)
}
