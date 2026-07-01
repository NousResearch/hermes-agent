import { ar } from './ar'
import { de } from './de'
import { en } from './en'
import { es } from './es'
import { fr } from './fr'
import { hi } from './hi'
import { it } from './it'
import { ja } from './ja'
import { ko } from './ko'
import { ptbr } from './pt-br'
import { ru } from './ru'
import { th } from './th'
import type { Locale, Translations } from './types'
import { vi } from './vi'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh,
  'zh-hant': zhHant,
  ja,
  ko,
  de,
  es,
  fr,
  'pt-br': ptbr,
  ar,
  hi,
  th,
  vi,
  it,
  ru
}
