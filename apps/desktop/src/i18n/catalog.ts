import { en } from './en'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { ru } from './ru'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh,
  'zh-hant': zhHant,
  ja,
  ru
}
