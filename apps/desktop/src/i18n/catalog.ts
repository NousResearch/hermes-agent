import { en } from './en'
import { zh } from './zh'
import { zhHant } from './zh-hant'
import { ja } from './ja'
import { ko } from './ko'
import { de } from './de'
import { es } from './es'
import { fr } from './fr'
import { ptbr } from './pt-br'
import { ar } from './ar'
import { hi } from './hi'
import { th } from './th'
import { vi } from './vi'
import { it } from './it'
import { ru } from './ru'
import type { Locale, Translations } from './types'

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
