import type { TranslationOverrides } from './define-locale'

export const arSelectionTranslate = {
  title: 'ترجمة',
  providerNote: 'يستخدم نموذج Hermes الذي أعددته. قد يغادر النص المحدد هذا الجهاز عبر ذلك المزود.',
  target: 'اللغة المفضلة',
  preferredHint:
    'يُحفظ للترجمات المستقبلية. إذا كان النص يطابق هدفًا غير إنجليزي، يترجمه Hermes إلى الإنجليزية بدلاً منه.',
  searchLanguages: 'ابحث عن لغة…',
  noLanguages: 'لم يتم العثور على لغات.',
  useLanguageTag: (name, tag) => `استخدام ${name} (${tag})`,
  languageTagHint: 'يمكنك أيضًا إدخال وسم لغة مثل pt-BR أو zh-Hant.',
  source: 'النص المحدد',
  translation: 'الترجمة',
  translating: 'جار الترجمة…',
  failed: 'فشلت الترجمة',
  emptyResult: 'أعاد المزود ترجمة فارغة.',
  tooLong: 'حدد ٤٬٠٠٠ حرف أو أقل للترجمة.',
  retry: 'إعادة المحاولة',
  copy: 'نسخ',
  copied: 'تم نسخ الترجمة',
  copyFailed: 'تعذر نسخ الترجمة'
} satisfies TranslationOverrides['selectionTranslate']

export const arSelectionActions = { readAloud: 'قراءة بصوت عال', lookUp: 'بحث', translate: 'ترجمة…', stop: 'إيقاف' }
