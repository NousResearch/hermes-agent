import { en } from './en'
import { es } from './es'
import { ja } from './ja'
import { pt } from './pt'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  es,
  ja,
  pt,
  zh,
  'zh-hant': zhHant
}
