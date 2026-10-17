import type { TranslationOverrides } from './define-locale'

export const arMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'إعادة تحميل خوادم MCP',
  title: 'إعادة تحميل خوادم MCP؟',
  description:
    'سيؤدي هذا إلى مزامنة خوادم MCP لهذا الملف الشخصي مع الإعدادات وتحديث أدوات هذه المحادثة. ستعيد الرسالة التالية إرسال الموجّه الكامل، مما يستهلك رموز ذاكرة التخزين المؤقت للموجّه.',
  confirm: 'إعادة التحميل الآن',
  deferred:
    'تم تحديث خوادم MCP لهذا الملف الشخصي. تحتفظ هذه المحادثة بأدواتها حتى ينتهي دورها قيد التشغيل؛ أعد التحميل بعد ذلك (أو أعد فتح المحادثة).',
  unchanged: 'خوادم MCP محدّثة بالفعل',
  success: 'تمت إعادة تحميل خوادم MCP',
  failed: servers => `تمت إعادة تحميل خوادم MCP، لكن هذه لم تتصل: ${servers}`,
  error: 'لم تتم إعادة تحميل خوادم MCP',
  confirmRequired: 'يتطلب /reload-mcp تأكيدًا',
  reloadedAlways: 'تمت إعادة تحميل خوادم MCP · لن يطلب /reload-mcp التأكيد بعد الآن في هذا الملف الشخصي',
  notConnected: servers => `لم يتصل: ${servers}`,
  completionNow: 'إعادة التحميل الآن (تخطي التأكيد مرة واحدة)',
  completionAlways: 'إعادة التحميل الآن وعدم السؤال مجددًا في هذا الملف الشخصي',
  skillsRemote:
    'تم تحديث قائمة المهارات. تتم إعادة الفحص المباشر على البوابة التي تملك هذا الملف الشخصي، وليس من هذه النافذة.',
  skillsFailed: 'فشلت إعادة تحميل المهارات'
}
