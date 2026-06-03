import { en } from './en'
import { zh } from './zh'

export const catalogs = {
  en,
  zh
}

export type TranslationKey = keyof typeof en
