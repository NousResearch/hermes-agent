import type { TranslationOverrides } from './define-locale'

export const arMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'إعادة تحميل خوادم MCP',
  title: 'إعادة تحميل خوادم MCP؟',
  description:
    'سيؤدي هذا إلى إعادة تشغيل خوادم MCP لهذا الملف الشخصي، وستعيد الرسالة التالية إرسال الموجّه الكامل، مما يستهلك رموز ذاكرة التخزين المؤقت للموجّه.',
  confirm: 'إعادة التحميل الآن',
  success: 'تمت إعادة تحميل خوادم MCP',
  failed: servers => `تمت إعادة تحميل خوادم MCP، لكن هذه لم تتصل: ${servers}`,
  error: 'لم تتم إعادة تحميل خوادم MCP',
  confirmRequired: 'يتطلب /reload-mcp تأكيدًا',
  reloadedAlways: 'تمت إعادة تحميل خوادم MCP · لن يطلب /reload-mcp التأكيد بعد الآن في هذا الملف الشخصي',
  notConnected: servers => `لم يتصل: ${servers}`,
  completionNow: 'إعادة التحميل الآن (تخطي التأكيد مرة واحدة)',
  completionAlways: 'إعادة التحميل الآن وعدم السؤال مجددًا في هذا الملف الشخصي',
  skillsFailed: 'فشلت إعادة تحميل المهارات'
}
