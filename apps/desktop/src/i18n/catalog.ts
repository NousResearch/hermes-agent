import { ar } from './ar'
import { en, type Translations } from './en'
import { ja } from './ja'
import type { Locale } from './languages'
import { ru } from './ru'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh,
  'zh-hant': zhHant,
  ja,
  ar,
  ru
}
