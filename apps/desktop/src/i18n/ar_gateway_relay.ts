import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in ar_settings.ts.
export const arGatewayRelay: GatewayRelayTranslations = {
  title: 'بوابات أخرى',
  intro:
    'افتح بوابة Hermes أخرى من هذا الـ Webapp. تنضم ملفاتها الشخصية ومحادثاتها إلى مبدّل البوابات؛ وتمر الطلبات عبر هذه البوابة موقّعة بمفتاحها الخاص، وتعمل باسم الحساب الذي توافق عليه هناك.',
  empty: 'لا توجد بوابات أخرى بعد.',
  addTitle: 'ربط بوابة',
  urlLabel: 'عنوان الـ Webapp',
  labelLabel: 'الاسم',
  connect: 'ربط',
  connecting: 'جارٍ فتح البوابة الأخرى…',
  connectFailed: 'تعذّر بدء ربط تلك البوابة',
  connected: label => `تم ربط ${label}`,
  denied: 'لم تُربط البوابة الأخرى: رُفض الطلب هناك.',
  failed: detail => `لم تُربط البوابة الأخرى: ${detail}`,
  remove: 'إزالة',
  removeConfirm: label => `إزالة ${label}؟ لن تصل إليها هذه البوابة بعد الآن، وسيُلغى إذنها هناك.`,
  removeFailed: 'تعذّرت إزالة تلك البوابة',
  removeHere: 'إزالة من هنا فقط',
  removeUnconfirmed: label => `إزالة ${label} من هنا فقط؟ يبقى إذنها هناك سارياً حتى تلغيه على ${label}.`,
  test: 'اختبار',
  reachable: 'يمكن الوصول إليها',
  unreachable: error => `لا يمكن الوصول إليها: ${error}`,
  scopes: scopes => `مسموح أيضًا: ${scopes}`,
  coreOnly: 'المحادثات والملفات الشخصية والجلسات والإعدادات فقط',
  grantsTitle: 'بوابات تصل إلى هذه البوابة',
  grantsIntro:
    'بوابات سمحت لها بتمرير الـ Webapp الخاص بك إلى هنا. أي شخص مسجّل الدخول إلى إحداها بحسابك يمكنه استخدام هذا الجهاز من خلالها.',
  grantsEmpty: 'لا تصل أي بوابة أخرى إلى هذه البوابة.',
  grantFrom: (origin, when) => `${origin} · سُمح بها ${when}`,
  revoke: 'إلغاء الإذن',
  revokeConfirm: name => `إلغاء إذن ${name}؟ لن تتمكن من الوصول إلى هذه البوابة بعد الآن، وستُغلق اتصالاتها المفتوحة.`,
  revokeFailed: 'تعذّر إلغاء إذن تلك البوابة',
  loadFailed: 'تعذّر تحميل تمرير البوابات'
}
