import type { TranslationOverrides } from './define-locale'

export const arCommon = {
  common: {
    apply: 'تطبيق',
    back: 'رجوع',
    save: 'حفظ',
    saving: 'جار الحفظ...',
    cancel: 'إلغاء',
    change: 'تغيير',
    choose: 'اختيار',
    clear: 'مسح',
    close: 'إغلاق',
    collapse: 'طي',
    confirm: 'تأكيد',
    connect: 'اتصال',
    connecting: 'جار الاتصال',
    continue: 'متابعة',
    copied: 'تم النسخ',
    copy: 'نسخ',
    copyFailed: 'فشل النسخ',
    delete: 'حذف',
    docs: 'الوثائق',
    done: 'تم',
    error: 'خطأ',
    failed: 'فشل',
    free: 'مجاني',
    loading: 'جار التحميل...',
    notSet: 'غير مضبوط',
    refresh: 'تحديث',
    remove: 'إزالة',
    replace: 'استبدال',
    retry: 'إعادة المحاولة',
    run: 'تشغيل',
    send: 'إرسال',
    set: 'ضبط',
    skip: 'تخطي',
    update: 'تحديث',
    on: 'مفعل',
    off: 'معطل'
  },
  ui: {
    search: {
      clear: 'مسح البحث'
    },
    pagination: {
      label: 'ترقيم الصفحات',
      previous: 'السابق',
      previousAria: 'الصفحة السابقة',
      next: 'التالي',
      nextAria: 'الصفحة التالية'
    },
    sidebar: {
      title: 'الشريط الجانبي',
      description: 'تنقل التطبيق',
      toggle: open => `${open ? 'إظهار' : 'إخفاء'} الشريط الجانبي`
    }
  }
} satisfies Pick<TranslationOverrides, 'common' | 'ui'>
