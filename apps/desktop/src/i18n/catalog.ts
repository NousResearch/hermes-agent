import { en } from './en'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { pt } from './pt'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  pt,
  zh,
  'zh-hant': zhHant,
  ja
}
