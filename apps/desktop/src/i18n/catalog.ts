import { en } from './en'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'
import { tr } from './tr'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  tr,
  zh,
  'zh-hant': zhHant,
  ja
}
