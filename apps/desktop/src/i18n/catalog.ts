import { en } from './en'
import { fr } from './fr'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  fr,
  zh,
  'zh-hant': zhHant,
  ja
}
