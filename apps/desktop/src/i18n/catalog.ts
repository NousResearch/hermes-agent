import { ar } from './ar'
import { en } from './en'
import { ja } from './ja'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { ptBr } from './pt-br'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  'pt-br': ptBr,
  zh,
  'zh-hant': zhHant,
  ja,
  ar
}
