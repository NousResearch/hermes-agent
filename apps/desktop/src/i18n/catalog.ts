import { ar } from './ar'
import { en } from './en'
import { ja } from './ja'
import { pl } from './pl'
import { ru } from './ru'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  pl,
  zh,
  'zh-hant': zhHant,
  ja,
  ar,
  ru
}
