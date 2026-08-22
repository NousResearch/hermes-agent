import { ar } from './ar'
import { en } from './en'
import { tr } from './tr'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  tr,
  zh,
  'zh-hant': zhHant,
  ja,
  ar
}
