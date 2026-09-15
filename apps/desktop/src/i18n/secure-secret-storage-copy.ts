import type { Locale, Translations } from './types'

export const SECURE_SECRET_STORAGE_DESCRIPTIONS: Record<Locale, string> = {
  en: 'Enabled by default. Saved gateway tokens and sign-in credentials are encrypted with the OS keychain (Keychain Access, GNOME Keyring, or Windows DPAPI). Turn this off only as an explicit plaintext escape hatch when the system keychain is unavailable.',
  zh: '默认启用。保存的网关令牌和登录凭据会使用操作系统钥匙串加密（钥匙串访问、GNOME Keyring 或 Windows DPAPI）。仅当系统钥匙串不可用时，才应明确关闭并使用明文应急模式。',
  'zh-hant':
    '預設啟用。已儲存的 Gateway 權杖與登入憑證會使用作業系統鑰匙圈加密（「鑰匙圈存取」、GNOME Keyring 或 Windows DPAPI）。只有在系統鑰匙圈無法使用時，才應明確關閉並使用明文應急模式。',
  ja: '既定で有効です。保存されたゲートウェイトークンとサインイン資格情報は、OS のキーチェーン（キーチェーンアクセス、GNOME Keyring、または Windows DPAPI）で暗号化されます。システムキーチェーンが使用できない場合に限り、明示的な平文の退避手段として無効にしてください。',
  ar: 'مفعّل افتراضيًا. تُشفَّر رموز البوابة وبيانات اعتماد تسجيل الدخول المحفوظة باستخدام سلسلة مفاتيح نظام التشغيل (Keychain Access أو GNOME Keyring أو Windows DPAPI). عطّل هذا الخيار صراحةً فقط كمسار احتياطي بنص صريح عندما لا تتوفر سلسلة مفاتيح النظام.',
  ru: 'Включено по умолчанию. Сохранённые токены шлюза и учётные данные для входа шифруются системным хранилищем ключей (Keychain Access, GNOME Keyring или Windows DPAPI). Отключайте это только явно, как аварийный переход к открытому тексту, если системное хранилище ключей недоступно.'
}

/** Keep policy copy in one product-level owner across every supported locale. */
export function withSecureSecretStorageCopy(locale: Locale, translations: Translations): Translations {
  return {
    ...translations,
    settings: {
      ...translations.settings,
      gateway: {
        ...translations.settings.gateway,
        keychainEncryptionDesc: SECURE_SECRET_STORAGE_DESCRIPTIONS[locale]
      }
    }
  }
}
