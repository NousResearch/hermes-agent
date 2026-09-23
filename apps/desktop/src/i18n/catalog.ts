import { ar } from './ar'
import { en } from './en'
import { ja } from './ja'
import { ru } from './ru'
import { withSecureSecretStorageCopy } from './secure-secret-storage-copy'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

export const TRANSLATIONS: Record<Locale, Translations> = {
  en: withSecureSecretStorageCopy('en', en),
  zh: withSecureSecretStorageCopy('zh', zh),
  'zh-hant': withSecureSecretStorageCopy('zh-hant', zhHant),
  ja: withSecureSecretStorageCopy('ja', ja),
  ar: withSecureSecretStorageCopy('ar', ar),
  ru: withSecureSecretStorageCopy('ru', ru)
}
