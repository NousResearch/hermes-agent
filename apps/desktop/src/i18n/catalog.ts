import { en } from './en'
import { de } from './de'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  de,
  zh,
  'zh-hant': zhHant,
  ja
}
